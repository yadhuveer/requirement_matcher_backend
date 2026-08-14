"""
Agentic feature extraction (requirements.md §2).

A LangChain v1 tool-calling agent extracts features from ONE document chunk and
self-checks with a verifier, re-extracting (bounded) when the verifier finds
missed or wrongly-split features.

This module is built up in steps:
  - step 2 (this): shared LLM, chunk state, and the `extract_features` tool
  - step 3: the `verify_extraction` tool
  - step 4: the round-cap guardrail middleware
  - step 5: create_agent wiring + the public runner

Design note — why the chunk lives in *state*, not in a tool argument:
the chunk is bulk input data. Making the agent LLM copy the whole chunk into its
tool call would be wasteful and lossy. Instead the chunk sits in the agent's
state and is *injected* into the tools via `ToolRuntime` (hidden from the model).
"""

import asyncio

from typing_extensions import NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain.tools import tool, ToolRuntime
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from app.config import settings
from app.models.extraction_schemas import FeatureList, VerifyVerdict


# --------------------------------------------------------------------------- #
# Shared model                                                                #
# --------------------------------------------------------------------------- #
# temperature=0 → deterministic extraction. with_structured_output(FeatureList)
# forces schema-valid output (a FeatureList instance) — no ```json parsing.

_llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0,
    api_key=settings.ANTHROPIC_API_KEY,
)
_extraction_llm = _llm.with_structured_output(FeatureList)


# --------------------------------------------------------------------------- #
# Agent state                                                                 #
# --------------------------------------------------------------------------- #
# Extends the built-in AgentState (which already carries `messages`). We add the
# current chunk, the latest extracted features, and a round counter that the
# guardrail middleware (step 4) reads to enforce the max-2-rounds cap.

class ExtractionState(AgentState):
    chunk_text: str                     # the chunk being processed (set at invoke time)
    features: NotRequired[list]         # latest extracted features (list of dicts)
    round: NotRequired[int]             # how many extraction rounds have run


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
# Adapted from the original single-shot prompt. The "return ONLY JSON" formatting
# rules are gone — structured outputs handle that now. Granularity rules stay,
# because the verifier and this prompt must share one idea of "one feature".

EXTRACTION_SYSTEM_PROMPT = """You are analyzing documentation for a software project that has already been built and delivered. Extract the distinct FEATURES that were implemented, and describe each in plain, non-technical language.

These descriptions are later matched against how NEW CLIENTS describe what they want — in everyday language ("a way for students to be grouped and given lessons"), NOT technical terms ("JWT auth", "CRUD"). So each description must capture what the feature DOES FOR USERS, in plain language, understandable on its own.

For each feature: a short `name` (3-6 words), a self-contained plain-language `description` (the most important field — include the domain context so the meaning is unambiguous), the `domain` (business area), and any `tech_details` mentioned (empty string if none).

Granularity — this is critical:
- Each feature = exactly ONE capability a client could ask for, build, or remove on its own.
- Test: "Could a client plausibly want one of these WITHOUT the other?" If yes → separate features. If no → one feature.
- Do NOT over-split: the create/edit/delete variations of managing one kind of thing are ONE feature, not three.
- Do NOT bundle several independently-requestable capabilities into one feature.

Extract only FEATURES — a concrete capability the software provides to a user or operator (something a client could ask you to build as a single line item). Judge WHAT the software does, not HOW WELL it does it.

Do NOT extract:
- Non-functional / quality requirements — statements about how well the system performs rather than what it does: performance / response time, scalability / load, availability / uptime, reliability, security & encryption standards, regulatory or industry compliance, data retention, disaster recovery, compatibility, accessibility, maintainability, observability. (Typical phrasing: "shall respond within X", "shall handle N users", "X% uptime", "shall comply with <standard>".) These are constraints ON features, not features.
- High-level objectives / goals / summary statements — broad umbrella items that restate, at a high level, capabilities described in detail elsewhere, without themselves being one concrete thing to build.
- Document scaffolding — background, scope, stakeholders, timelines, budget, assumptions, constraints, risks, glossary, revision history, traceability tables.

Test each candidate: "Could a client ask us to BUILD this as one distinct thing they would use?" A quality target, a compliance obligation, or a broad business goal is NOT a feature — skip it. (A security or access capability a user or admin actually operates — signing in, managing permissions, enabling two-factor — IS a feature; the exclusion is for quality targets and compliance obligations, not for security capabilities.)

If a section contains no features, extract nothing from it. Never invent features not supported by the text."""


# --------------------------------------------------------------------------- #
# Tool: extract_features                                                      #
# --------------------------------------------------------------------------- #

@tool
async def extract_features(runtime: ToolRuntime, feedback: str = "") -> Command:
    """Extract the distinct product features from the current document section.

    Call this first with no feedback. If the verifier reports problems, call it
    AGAIN and pass those problems as `feedback` so they get fixed (e.g. features
    that were missed, or features that were wrongly split apart).
    """
    chunk: str = runtime.state["chunk_text"]
    round_no: int = runtime.state.get("round", 0) + 1

    user = f"Document section:\n---\n{chunk}\n---"
    if feedback:
        user += (
            "\n\nA previous extraction of this same section had the problems below. "
            "Produce a corrected, complete feature list that fixes them:\n"
            f"{feedback}"
        )

    result: FeatureList = await _extraction_llm.ainvoke(
        [SystemMessage(EXTRACTION_SYSTEM_PROMPT), HumanMessage(user)]
    )
    features = [f.model_dump() for f in result.features]

    # Write the features + round into state so the verifier (step 3) can read
    # them; the ToolMessage is what the agent LLM sees.
    return Command(
        update={
            "features": features,
            "round": round_no,
            "messages": [
                ToolMessage(
                    content=f"Extracted {len(features)} feature(s) on round {round_no}.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


# --------------------------------------------------------------------------- #
# Tool: verify_extraction                                                     #
# --------------------------------------------------------------------------- #
# The verifier does TWO checks on the CURRENT chunk only (requirements.md §2.2):
#   1. completeness   — any capability in the text that was NOT extracted?
#   2. local over-split — any single capability wrongly broken into several?
# It deliberately does NOT touch cross-section duplicates (that is dedupe's job).

_verify_llm = _llm.with_structured_output(VerifyVerdict)

VERIFY_SYSTEM_PROMPT = """You are checking whether a feature extraction from a document section is complete and correctly split. You are given the original section text and the features extracted from it. Judge TWO things:

1. COMPLETENESS — is any distinct FEATURE (a concrete capability the software DOES for a user or operator) present in the text but MISSING from the extracted features? List each as a `missed` hint. Judge WHAT the software does, not HOW WELL — see the exclusions below for what does NOT count as a feature.

2. OVER-SPLITTING — was any single capability wrongly broken into multiple features? Test: "Could a client plausibly want one of these WITHOUT the other?" If NO, they are one capability and were wrongly split — list them together in `wrongly_split`. (The create/edit/delete variations of managing one kind of thing are ONE feature.)

Do NOT flag as missing:
- Non-functional / quality requirements (performance, scalability, uptime, reliability, security or encryption standards, compliance, data retention, disaster recovery, compatibility, accessibility, maintainability, observability), high-level objectives / goals / summary statements, or document scaffolding (background, scope, stakeholders, timelines, budget, assumptions, risks, glossary, revision history, traceability tables). These are correctly EXCLUDED from features — never report them as missed.
- Duplicates across sections — a separate dedupe step handles those.
- Features that are correctly separate (a client could genuinely want one without the other).
- Anything not actually supported by the text.

Set `clean=true` ONLY if there is nothing missed AND nothing wrongly split; then both lists must be empty. Be strict, but do not invent problems — if the extraction is genuinely good, say it is clean."""


@tool
async def verify_extraction(runtime: ToolRuntime) -> str:
    """Check the current extraction against the section text. Reports whether any
    feature was MISSED or WRONGLY SPLIT.

    Call this after extract_features. If it reports problems, call extract_features
    again and pass the reported problems as `feedback`. If it says the extraction
    is clean, you are done.
    """
    chunk: str = runtime.state["chunk_text"]
    features: list = runtime.state.get("features", [])

    features_text = (
        "\n".join(f"- {f['name']}: {f['description']}" for f in features)
        or "(no features extracted)"
    )
    user = (
        f"Document section:\n---\n{chunk}\n---\n\n"
        f"Features extracted from it:\n{features_text}"
    )

    verdict: VerifyVerdict = await _verify_llm.ainvoke(
        [SystemMessage(VERIFY_SYSTEM_PROMPT), HumanMessage(user)]
    )

    if verdict.clean:
        return "VERIFICATION CLEAN — nothing missed, nothing wrongly split. Extraction is complete; stop."

    # Human-readable feedback the agent can pass straight back into extract_features.
    lines = ["VERIFICATION FOUND ISSUES — call extract_features again with this feedback:"]
    if verdict.missed:
        lines.append("Missed capabilities (add these):")
        lines += [f"  - {m.hint}" for m in verdict.missed]
    if verdict.wrongly_split:
        lines.append("Wrongly split (merge each group into ONE feature):")
        lines += [
            f"  - {', '.join(w.feature_names)} — {w.reason}"
            for w in verdict.wrongly_split
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Guardrail middleware: cap extraction rounds (requirements.md §2.3)           #
# --------------------------------------------------------------------------- #
# The extract tool increments state["round"] each time it runs. This middleware
# is the HARD stop: before every model turn, if we've already run `max_rounds`
# extractions, we jump straight to the end — the model cannot start another round
# no matter what it decides. We hold the reins, not the model. The latest extract's
# features are already in state, so stopping here keeps them (best-effort, §2.3).

class RoundCapMiddleware(AgentMiddleware):
    def __init__(self, max_rounds: int = 2) -> None:
        super().__init__()
        self.max_rounds = max_rounds

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        if state.get("round", 0) >= self.max_rounds:
            return {"jump_to": "end"}
        return None


# --------------------------------------------------------------------------- #
# The agent + public runner (step 5)                                          #
# --------------------------------------------------------------------------- #

AGENT_SYSTEM_PROMPT = """You extract software features from ONE document section, and you MUST verify your work. The section text is already available to your tools — you do not need it in the conversation.

Follow this loop:
1. Call `extract_features` (no feedback) for an initial feature list.
2. Call `verify_extraction` to check it.
3. If verification reports issues, call `extract_features` AGAIN, passing the reported issues verbatim as `feedback`. Then call `verify_extraction` once more.
4. Stop as soon as verification is clean (or you are told the round limit is reached). Reply with a one-line summary.

Do not call tools more than necessary. If the first extraction verifies clean, stop immediately."""

_extraction_agent = create_agent(
    model=_llm,
    tools=[extract_features, verify_extraction],
    middleware=[RoundCapMiddleware(max_rounds=2)],
    state_schema=ExtractionState,
    system_prompt=AGENT_SYSTEM_PROMPT,
)


async def _run_agent_on_chunk(chunk: str) -> list[dict]:
    """Run the extract → verify → (re-extract) agent on a single chunk and
    return its final feature list (list of dicts)."""
    final_state = await _extraction_agent.ainvoke(
        {
            "chunk_text": chunk,
            "round": 0,
            "messages": [HumanMessage("Process the document section now.")],
        }
    )
    return final_state.get("features", [])


async def extract_features_from_chunks(chunks: list[str]) -> list[dict]:
    """Public entry point — same signature as the old feature_extractor.

    Runs the extraction agent on every chunk with bounded concurrency and returns
    all features flattened. Cross-chunk duplicates are handled later by dedupe.
    """
    max_concurrent = 3
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_limit(chunk: str) -> list[dict]:
        async with semaphore:
            return await _run_agent_on_chunk(chunk)

    results = await asyncio.gather(
        *[run_with_limit(c) for c in chunks],
        return_exceptions=True,
    )

    all_features: list[dict] = []
    for r in results:
        if isinstance(r, Exception):
            # One bad chunk shouldn't sink the whole document.
            continue
        all_features.extend(r)
    return all_features
