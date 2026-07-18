import json
from typing import TypedDict, Optional
from anthropic import AsyncAnthropic

from app.config import settings
from app.services.embedding_service import embed_texts
from app.services.qdrant_service import client as qdrant_client, COLLECTION_NAME
from app.prompts.prompts import NODE1_RELEVANCE_SYSTEM_PROMPT ,REPHRASE_SYSTEM_PROMPT , CLASSIFY_SYSTEM_PROMPT,MODIFICATION_SYSTEM_PROMPT
from langgraph.graph import StateGraph, END

anthropic_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

MAX_FEATURE_RETRY=1


class MatchState(TypedDict):
    requirement:dict

    search_query:str
    candidates:list
    relevant_match:Optional[dict]
    retry_count:int

    status:Optional[str]
    modification_needed: Optional[str]
    confidence: Optional[float]


async def search_and_judge(state:MatchState) -> dict:
    query = state.get("search_query") or state["requirement"]["description"]

    query_vector = (await embed_texts([query]))[0]
    
    hits = qdrant_client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=3,
    ).points

    candidates= [{
        "feature_id": h.payload.get("feature_id"),
        "name":h.payload.get("name",""),
        "description":h.payload.get("description",""),
        "domain":h.payload.get("domain",""),
        "score":h.score,
    } for h in hits
    ]

    if not candidates:
        return {"candidates":[], "relevant_match":None}
    
    req = state["requirement"]

    candidates_text = "\n".join(
        f"[{i}] name: {c['name']} | domain: {c['domain']} | description :{c['description']}"
        for i,c in enumerate(candidates)
    )

    user_msg = (
        f"Client requirement:\n"
        f"name: {req.get('name','')}\n"
        f"description: {req['description']}\n\n"
        f"Candidate features:\n{candidates_text}"
    )

    response = await anthropic_client.messages.create(
        model=settings.LLM_MODEL,
        max_tokens=500,
        temperature=0,
        system=NODE1_RELEVANCE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],

    )

    raw = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()
 
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        verdict = {"relevant": False}
 
    if verdict.get("relevant") and "matched_index" in verdict:
        idx = verdict["matched_index"]
        if 0 <= idx < len(candidates):
            return {
                "candidates": candidates,
                "relevant_match": candidates[idx],
                "confidence": verdict.get("confidence"),
            }
 
    
    return {"candidates": candidates, "relevant_match": None}


def route_after_search(state:MatchState)->str:
    if state.get("relevant_match") is not None:
        return "classify"
    
    if state.get("retry_count") < MAX_FEATURE_RETRY:
        return "rephrase"
    
    return "handle_manually"



async def rephrase_query(state:MatchState) ->dict:
    req= state["requirement"]
    original = state.get("search_query") or req["description"]

    response = await anthropic_client.messages.create(
        model=settings.LLM_MODEL,
        max_tokens=300,
        temperature=0.3,
        system=REPHRASE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Requirement: {original}"}],
    )

    rephrased = response.content[0].text.strip()

    return {
        "search_query": rephrased,
        "retry_count": state.get("retry_count",0) +1,
    }

async def classify_match(state:MatchState) -> dict:
    req = state["requirement"]
    match = state["relevant_match"]

    user_msg = (
        f"Client requirement:\n"
        f"domain:{req.get('domain','')}\n"
        f"description:{req['description']}\n\n"
        f"Exsisting feature we built:\n"
        f"domain: {match.get('domain','')}\n"
        f"description: {match.get('description','')}"
    )


    response = await anthropic_client.messages.create(
        model=settings.LLM_MODEL,
        max_tokens=300,
        temperature=0,
        system=CLASSIFY_SYSTEM_PROMPT,
        messages=[{"role":"user","content":user_msg}]
    )

    
    raw = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()

    try:
        verdict = json.loads(raw)
        status = verdict.get("status")
        if status not in ("exact_match", "needs_modification"):
            status = "needs_modification"
        return {
            "status":status,
            "confidence":verdict.get("confidence", state.get("confidence"))
        }
    except json.JSONDecodeError:
        return {"status":"needs_modification"}

def route_after_classify(state:MatchState) ->str:
    if state.get("status")== "needs_modification":
        return "describe_modification"
    return "END"

async def describe_modification(state:MatchState) -> dict:
    req = state["requirement"]
    match = state["relevant_match"]

    user_msg = (
        f"Client requirement:\n"
        f"domain: {req.get('domain','')}\n"
        f"description: {req['description']}\n\n"
        f"Exsisting feature we built:\n"
        f"domain: {match.get('domain','')}\n"
        f"description: {match.get('description','')}"
    )

    response =  await anthropic_client.messages.create(
        model=settings.LLM_MODEL,
        max_tokens=400,
        temperature=0,
        system=MODIFICATION_SYSTEM_PROMPT,
        messages=[{"role":"user", "content":user_msg}] 
    )

    modification=response.content[0].text.strip()
    return {"modification_needed":modification}

def mark_handle_manually(state:MatchState) -> dict:
    return {
        "status": "handle_manually",
        "relevant_match":None,
        "modification_needed":None
    }





_graph = StateGraph(MatchState)

_graph.add_node("search", search_and_judge)
_graph.add_node("rephrase", rephrase_query)
_graph.add_node("classify", classify_match)
_graph.add_node("describe_modification", describe_modification)
_graph.add_node("handle_manually", mark_handle_manually)

_graph.set_entry_point("search")

_graph.add_conditional_edges(
    "search",
    route_after_search,
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
        "describe_modification": "describe_modification",
        "END": END,
    },
)

_graph.add_edge("describe_modification", END)
_graph.add_edge("handle_manually", END)

matching_app = _graph.compile()


async def match_requirement(requirement: dict) -> dict:
    initial_state = {
        "requirement": requirement,
        "search_query": "",
        "candidates": [],
        "relevant_match": None,
        "retry_count": 0,
        "status": None,
        "modification_needed": None,
        "confidence": None,
    }
    final_state = await matching_app.ainvoke(initial_state)

    return {
        "requirement_name": requirement.get("name", ""),
        "requirement_description": requirement.get("description", ""),
        "status": final_state.get("status"),
        "matched_feature": final_state.get("relevant_match"),
        "modification_needed": final_state.get("modification_needed"),
        "confidence": final_state.get("confidence"),
    }