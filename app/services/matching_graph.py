"""
Composite matching graph (requirements.md §3).

A LangGraph StateGraph matches ONE client requirement against the global feature
library. Unlike the old single-match version, it can select a COMBINATION of
existing features (`coverage` node), and it explains exact matches.

Flow:
  search -> coverage -> classify -> exact_match            -> END
      │                     │                                (needs_mod)
      │ (no relevant)       └────────────────► modify ⇄ critic -> END
      └─► rephrase (retry once) -> search
      └─► handle_manually -> END

Built incrementally:
  - step 2 (this): state, shared LLM, `search`, `coverage`
  - step 3: `classify`, `exact_match`
  - step 4: `modify` + `critic`
  - step 5: graph wiring + guardrails + public `match_requirement`

Every LLM node uses `.with_structured_output(...)` (schema-safe + LangSmith-traced).
"""

from typing import TypedDict, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from app.config import settings
from app.services.embedding_service import embed_texts
from app.services.qdrant_service import client as qdrant_client, COLLECTION_NAME
from app.models.matching_schemas import (
    CoverageDecision,
    ClassifyVerdict,
    MatchExplanation,
    ModificationOutput,
    CriticVerdict,
)
from app.prompts.prompts import (
    COVERAGE_SYSTEM_PROMPT,
    CLASSIFY_SET_SYSTEM_PROMPT,
    EXACT_MATCH_SYSTEM_PROMPT,
    MODIFY_SET_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    REPHRASE_SYSTEM_PROMPT,
)


TOP_K = 5            # candidates retrieved per requirement (§3.3)
MAX_SEARCH_RETRY = 1  # rephrase -> search retries (§3.5)
MAX_MODIFY_RETRY = 1  # modify -> critic retries (§3.5)


# --------------------------------------------------------------------------- #
# Graph state                                                                 #
# --------------------------------------------------------------------------- #

class MatchState(TypedDict):
    requirement: dict                    # {name, description, domain, ...}

    search_query: str                    # current query (rephrase updates it)
    candidates: list                     # retrieved candidate features
    retry_count: int                     # rephrase retries so far

    selected_features: list              # the covering set chosen by `coverage`
    status: Optional[str]                # exact_match | needs_modification | handle_manually
    explanation: Optional[str]           # exact-match "why it matches"
    modification_needed: Optional[str]   # needs_modification "what to change"
    modify_retry: int                    # modify -> critic retries so far
    modification_valid: Optional[bool]   # critic verdict (routes the modify loop)
    critic_feedback: Optional[str]       # critic's reason, fed back into modify on retry
    confidence: Optional[float]          # one overall score for the matched set
    reasoning: Optional[str]             # coverage reasoning (telemetry)


# --------------------------------------------------------------------------- #
# Shared model                                                                #
# --------------------------------------------------------------------------- #

_llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0,
    api_key=settings.ANTHROPIC_API_KEY,
)
_coverage_llm = _llm.with_structured_output(CoverageDecision)
_classify_llm = _llm.with_structured_output(ClassifyVerdict)
_exact_llm = _llm.with_structured_output(MatchExplanation)
_modify_llm = _llm.with_structured_output(ModificationOutput)
_critic_llm = _llm.with_structured_output(CriticVerdict)

# Rephrase returns a plain sentence (no schema) and uses a little temperature so
# the retry query is worded differently enough to surface new candidates.
_rephrase_llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0.4,
    api_key=settings.ANTHROPIC_API_KEY,
)


def _selected_text(features: list) -> str:
    """Render the selected feature set for a prompt (one line each)."""
    return "\n".join(
        f"- name: {f['name']} | domain: {f['domain']} | description: {f['description']}"
        for f in features
    )


# --------------------------------------------------------------------------- #
# Node: search (retrieve top-K candidates)                                    #
# --------------------------------------------------------------------------- #

async def search(state: MatchState) -> dict:
    query = state.get("search_query") or state["requirement"]["description"]

    try:
        query_vector = (await embed_texts([query]))[0]
        hits = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=TOP_K,          # global search across all projects (§1)
        ).points
    except Exception:
        # Embedding or Qdrant failure (e.g. a network timeout) -> no candidates,
        # which routes this ONE requirement to rephrase/handle_manually instead
        # of crashing the whole batch.
        return {"candidates": []}

    candidates = [
        {
            "feature_id": h.payload.get("feature_id"),
            "name": h.payload.get("name", ""),
            "description": h.payload.get("description", ""),
            "domain": h.payload.get("domain", ""),
            "score": h.score,
        }
        for h in hits
    ]
    return {"candidates": candidates}


# --------------------------------------------------------------------------- #
# Node: coverage (single vs composite selection)                              #
# --------------------------------------------------------------------------- #

async def coverage(state: MatchState) -> dict:
    candidates = state["candidates"]
    if not candidates:
        return {"selected_features": []}

    req = state["requirement"]
    candidates_text = "\n".join(
        f"[{i}] name: {c['name']} | domain: {c['domain']} | description: {c['description']}"
        for i, c in enumerate(candidates)
    )
    user = (
        f"Client requirement:\n"
        f"name: {req.get('name', '')}\n"
        f"description: {req['description']}\n\n"
        f"Candidate features:\n{candidates_text}"
    )

    try:
        decision: CoverageDecision = await _coverage_llm.ainvoke(
            [SystemMessage(COVERAGE_SYSTEM_PROMPT), HumanMessage(user)]
        )
    except Exception:
        # Fallback: treat as no match -> routes to rephrase / handle_manually.
        return {"selected_features": []}

    if not decision.relevant or not decision.selected_ids:
        return {"selected_features": [], "reasoning": decision.reasoning}

    selected = [candidates[i] for i in decision.selected_ids if 0 <= i < len(candidates)]
    return {
        "selected_features": selected,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
    }


# --------------------------------------------------------------------------- #
# Node: classify (exact_match vs needs_modification, over the whole set)       #
# --------------------------------------------------------------------------- #

async def classify(state: MatchState) -> dict:
    req = state["requirement"]
    user = (
        f"Client requirement:\n"
        f"domain: {req.get('domain', '')}\n"
        f"description: {req['description']}\n\n"
        f"Existing feature(s) we built:\n{_selected_text(state['selected_features'])}"
    )

    try:
        verdict: ClassifyVerdict = await _classify_llm.ainvoke(
            [SystemMessage(CLASSIFY_SET_SYSTEM_PROMPT), HumanMessage(user)]
        )
    except Exception:
        # Fallback: safer to route to modify (describe the gap) than to crash.
        return {"status": "needs_modification"}

    # coverage already set the one overall `confidence`; classify only sets status.
    return {"status": verdict.status}


# --------------------------------------------------------------------------- #
# Node: exact_match (write the client-facing "why it already matches")         #
# --------------------------------------------------------------------------- #

async def exact_match(state: MatchState) -> dict:
    req = state["requirement"]
    user = (
        f"Client requirement:\n{req['description']}\n\n"
        f"Existing feature(s):\n{_selected_text(state['selected_features'])}"
    )

    try:
        result: MatchExplanation = await _exact_llm.ainvoke(
            [SystemMessage(EXACT_MATCH_SYSTEM_PROMPT), HumanMessage(user)]
        )
        explanation = result.explanation
    except Exception:
        explanation = "This requirement is already covered by existing feature(s)."

    return {"explanation": explanation}


# --------------------------------------------------------------------------- #
# Node: modify (describe what must change; retries incorporate critic feedback)#
# --------------------------------------------------------------------------- #

async def modify(state: MatchState) -> dict:
    req = state["requirement"]
    feedback = state.get("critic_feedback")  # set by critic only when it rejected

    user = (
        f"Client requirement:\n"
        f"domain: {req.get('domain', '')}\n"
        f"description: {req['description']}\n\n"
        f"Existing feature(s) we built:\n{_selected_text(state['selected_features'])}"
    )
    if feedback:
        # This run is a RETRY — show the critic's reason so we fix that gap.
        user += (
            f"\n\nA reviewer REJECTED your previous description for this reason:\n"
            f'"{feedback}"\n'
            f"Write an improved description that fixes exactly this."
        )

    try:
        result: ModificationOutput = await _modify_llm.ainvoke(
            [SystemMessage(MODIFY_SET_SYSTEM_PROMPT), HumanMessage(user)]
        )
        modification = result.modification
    except Exception:
        modification = ""  # empty -> critic rejects; retry is capped so no infinite loop

    update = {"modification_needed": modification}
    if feedback:
        # Count this attempt only when it was a retry (mirrors the rephrase counter).
        update["modify_retry"] = state.get("modify_retry", 0) + 1
    return update


# --------------------------------------------------------------------------- #
# Node: critic (reject vacuous modifications; its reason feeds the retry)      #
# --------------------------------------------------------------------------- #

async def critic(state: MatchState) -> dict:
    req = state["requirement"]
    user = (
        f"Client requirement:\n{req['description']}\n\n"
        f"Existing feature(s):\n{_selected_text(state['selected_features'])}\n\n"
        f"Proposed modification description:\n{state.get('modification_needed', '')}"
    )

    try:
        verdict: CriticVerdict = await _critic_llm.ainvoke(
            [SystemMessage(CRITIC_SYSTEM_PROMPT), HumanMessage(user)]
        )
    except Exception:
        # Fallback: accept, so a critic error never loops forever.
        return {"modification_valid": True, "critic_feedback": None}

    return {
        "modification_valid": verdict.valid,
        "critic_feedback": None if verdict.valid else verdict.reason,
    }


def route_after_critic(state: MatchState) -> str:
    """Accept a valid modification; otherwise retry modify once, then give up."""
    if state.get("modification_valid"):
        return "END"
    if state.get("modify_retry", 0) < MAX_MODIFY_RETRY:
        return "modify"
    return "END"  # retries exhausted -> keep the best modification we produced


# --------------------------------------------------------------------------- #
# Node: rephrase (reword the query and search again, once)                    #
# --------------------------------------------------------------------------- #

async def rephrase(state: MatchState) -> dict:
    req = state["requirement"]
    original = state.get("search_query") or req["description"]

    try:
        resp = await _rephrase_llm.ainvoke(
            [SystemMessage(REPHRASE_SYSTEM_PROMPT), HumanMessage(f"Requirement: {original}")]
        )
        rephrased = resp.content.strip() if isinstance(resp.content, str) else original
    except Exception:
        rephrased = original  # fall back to the original query; retry_count still ticks up

    return {"search_query": rephrased, "retry_count": state.get("retry_count", 0) + 1}


# --------------------------------------------------------------------------- #
# Node: handle_manually (nothing reusable found)                              #
# --------------------------------------------------------------------------- #

def handle_manually(state: MatchState) -> dict:
    return {
        "status": "handle_manually",
        "selected_features": [],
        "explanation": None,
        "modification_needed": None,
        "confidence": None,
    }


# --------------------------------------------------------------------------- #
# Routers                                                                     #
# --------------------------------------------------------------------------- #

def route_after_coverage(state: MatchState) -> str:
    """Something covered it -> classify; nothing -> rephrase once, else give up."""
    if state.get("selected_features"):
        return "classify"
    if state.get("retry_count", 0) < MAX_SEARCH_RETRY:
        return "rephrase"
    return "handle_manually"


def route_after_classify(state: MatchState) -> str:
    if state.get("status") == "needs_modification":
        return "modify"
    return "exact_match"


# --------------------------------------------------------------------------- #
# Graph wiring                                                                 #
# --------------------------------------------------------------------------- #

_graph = StateGraph(MatchState)

_graph.add_node("search", search)
_graph.add_node("coverage", coverage)
_graph.add_node("rephrase", rephrase)
_graph.add_node("classify", classify)
_graph.add_node("exact_match", exact_match)
_graph.add_node("modify", modify)
_graph.add_node("critic", critic)
_graph.add_node("handle_manually", handle_manually)

_graph.set_entry_point("search")
_graph.add_edge("search", "coverage")

_graph.add_conditional_edges(
    "coverage",
    route_after_coverage,
    {
        "classify": "classify",
        "rephrase": "rephrase",
        "handle_manually": "handle_manually",
    },
)
_graph.add_edge("rephrase", "search")

_graph.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "exact_match": "exact_match",
        "modify": "modify",
    },
)
_graph.add_edge("exact_match", END)

_graph.add_edge("modify", "critic")
_graph.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "modify": "modify",
        "END": END,
    },
)

_graph.add_edge("handle_manually", END)

matching_app = _graph.compile()


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #

async def match_requirement(requirement: dict) -> dict:
    """Match ONE requirement against the library; returns a flat result dict."""
    initial_state: MatchState = {
        "requirement": requirement,
        "search_query": "",
        "candidates": [],
        "retry_count": 0,
        "selected_features": [],
        "status": None,
        "explanation": None,
        "modification_needed": None,
        "modify_retry": 0,
        "modification_valid": None,
        "critic_feedback": None,
        "confidence": None,
        "reasoning": None,
    }

    # recursion_limit is a backstop; the retry counters are the real guardrails.
    try:
        final_state = await matching_app.ainvoke(initial_state, config={"recursion_limit": 25})
    except Exception:
        # One requirement failing (timeout, etc.) must not sink the whole batch;
        # degrade it to handle_manually and let the rest proceed.
        final_state = {}

    status = final_state.get("status") or "handle_manually"
    return {
        "requirement_name": requirement.get("name", ""),
        "requirement_description": requirement.get("description", ""),
        "status": status,
        "matched_features": final_state.get("selected_features") or [],
        "explanation": final_state.get("explanation"),
        "modification_needed": final_state.get("modification_needed"),
        "confidence": final_state.get("confidence"),
    }
