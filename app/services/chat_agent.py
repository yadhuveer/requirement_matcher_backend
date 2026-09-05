"""
Chat agent for one analysis (chatagent.md).

Phase 2: a per-analysis conversational assistant that answers doubts (grounded in
the analysis data) and EDITS a requirement's status / explanation / modification via
a tool — streamed token-by-token over SSE. The user tags exactly one requirement per
message; that requirement is the focus and the edit target.

Long-term memory (conversation summary, episodic summary) is added in later phases;
for now working memory = the recent messages of the session.

Everything here is additive and read/writes only the chat + new_features tables.
"""

import json
from typing import Optional, Literal, AsyncGenerator

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from sqlalchemy.orm import Session, joinedload

from app.config import settings
from app.models.db_models import (
    NewProject,
    NewFeature,
    NewFeatureMatch,
    ChatSession,
    ChatMessage,
)
from app.services.requirement_service import get_analysis_results


MODEL = "claude-sonnet-4-6"
ALLOWED_STATUSES = ("exact_match", "needs_modification", "handle_manually")
MAX_TOOL_LOOPS = 4          # safety cap on the reply→tool→reply loop

# Memory budget (chatagent.md §4). Token-based, not message-count: the window is
# huge, but every turn re-sends context, so we keep per-turn input small.
WORKING_TOKENS_LIMIT = 4000   # raw recent-message window before we summarise
KEEP_RECENT_MESSAGES = 4      # always kept verbatim after a fold-in (~2 turns)
SUMMARY_TOKEN_CAP = 1000      # conversation summary size cap
MESSAGE_FETCH_CAP = 200       # hard cap on rows pulled per turn

def _approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) — good enough for budget thresholds."""
    return len(text) // 4


# --------------------------------------------------------------------------- #
# Tool the model can call (executed by us, with the DB session in scope)       #
# --------------------------------------------------------------------------- #

class update_requirement(BaseModel):
    """Update the TAGGED requirement. Include ONLY the field(s) you want to change.
    Changing `status` alone updates just the status and KEEPS the existing explanation
    and modification (the user may switch the status back). Only set `explanation` or
    `modification` when the user asked to change that text (optionally giving a hint of
    how it should read)."""

    requirement_id: int = Field(description="id of the requirement to update (the tagged one).")
    status: Optional[Literal["exact_match", "needs_modification", "handle_manually"]] = Field(
        default=None, description="new status, only if the user wants to change it."
    )
    explanation: Optional[str] = Field(
        default=None, description="new exact-match explanation, only if the user asked to change it."
    )
    modification: Optional[str] = Field(
        default=None, description="new 'what to change' text, only if the user asked to change it."
    )


_llm = ChatAnthropic(
    model=MODEL,
    temperature=0,
    max_tokens=2000,          # replies are short
    api_key=settings.ANTHROPIC_API_KEY,
)
_llm_with_tools = _llm.bind_tools([update_requirement])


SYSTEM_PROMPT = """You are the assistant for ONE requirement-matching analysis. An agency matched a new client's requirements against features it has already built; each requirement is labelled exact_match, needs_modification, or handle_manually (build from scratch).

You can do THREE things and nothing else:
1. Answer greetings briefly and naturally.
2. Answer the user's questions/doubts about THIS analysis — a requirement, its verdict, or the existing feature(s) it matched — using ONLY the analysis data given to you below. Describe the existing feature(s) ACCURATELY, exactly as given: do NOT rename them, and do NOT invent or embellish what they do. If a feature was built in a different context, it is fine to say so.
3. Edit a requirement when the user asks, by calling the `update_requirement` tool.

If a message is not a greeting and not about this analysis / requirement matching, say you can only help with this analysis and do not answer it.

The user tags exactly ONE requirement with each message; it is shown below as the TAGGED REQUIREMENT and is the focus of the message and the target of any edit — use its id when calling the tool.

Editing rules:
- To change status, call the tool with just `status`. Do NOT wipe the explanation or modification when only the status changes.
- Only change the explanation/modification text when the user asks; if they hint how it should read, follow the hint. Keep new text grounded and accurate.
- After an edit, briefly confirm in plain language what you changed.

Keep replies short and clear."""


# --------------------------------------------------------------------------- #
# Analysis snapshot (grounding context)                                        #
# --------------------------------------------------------------------------- #

def build_snapshot(db: Session, project: NewProject, tagged_requirement_id: Optional[int]) -> str:
    """Compact index of every requirement + the tagged requirement in full.
    Kept small so per-turn context stays cheap even for large analyses."""
    features = (
        db.query(NewFeature)
        .filter(NewFeature.project_id == project.id)
        .order_by(NewFeature.id)
        .all()
    )

    lines = [f"ANALYSIS: {project.project_name}", "", "ALL REQUIREMENTS (id | status | name):"]
    for f in features:
        lines.append(f"[{f.id}] {f.match_status or 'unknown'} | {f.requirement_name or 'Untitled'}")

    tagged = next((f for f in features if f.id == tagged_requirement_id), None) if tagged_requirement_id else None
    if tagged is not None:
        matches = (
            db.query(NewFeatureMatch)
            .options(joinedload(NewFeatureMatch.completed_feature))
            .filter(NewFeatureMatch.new_feature_id == tagged.id)
            .all()
        )
        matched_text = "\n".join(
            f"  - {m.completed_feature.name}: {m.completed_feature.description}"
            for m in matches
            if m.completed_feature is not None
        ) or "  (none)"

        lines += [
            "",
            "TAGGED REQUIREMENT (the user's message is about THIS one; it is the edit target):",
            f"id: {tagged.id}",
            f"name: {tagged.requirement_name}",
            f"description: {tagged.requirement_description}",
            f"status: {tagged.match_status}",
            "matched existing features:",
            matched_text,
            f"explanation (why exact_match): {tagged.explanation or '(none)'}",
            f"modification_needed (what to change): {tagged.modification_needed or '(none)'}",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The edit tool's actual effect (validated, scoped to this analysis)           #
# --------------------------------------------------------------------------- #

def _apply_update(db: Session, project_id: int, args: dict) -> dict:
    rid = args.get("requirement_id")
    req = (
        db.query(NewFeature)
        .filter(NewFeature.id == rid, NewFeature.project_id == project_id)
        .first()
    )
    if req is None:
        return {"ok": False, "error": "requirement not found in this analysis"}

    status = args.get("status")
    if status is not None:
        if status not in ALLOWED_STATUSES:
            return {"ok": False, "error": f"invalid status; allowed: {ALLOWED_STATUSES}"}
        req.match_status = status
    # Only touch explanation / modification when explicitly provided (keep otherwise).
    if args.get("explanation") is not None:
        req.explanation = args["explanation"]
    if args.get("modification") is not None:
        req.modification_needed = args["modification"]

    db.commit()
    return {"ok": True, "requirement_id": rid, "name": req.requirement_name, "status": req.match_status}


# --------------------------------------------------------------------------- #
# Session + message persistence                                                #
# --------------------------------------------------------------------------- #

def create_session(db: Session, project_id: int) -> ChatSession:
    s = ChatSession(project_id=project_id)
    db.add(s)
    db.commit()
    db.refresh(s)
    return s

def save_message(db: Session, session_id: int, role: str, content: str) -> None:
    db.add(ChatMessage(session_id=session_id, role=role, content=content))
    db.commit()

def load_recent_messages(db: Session, session_id: int, limit: int = MESSAGE_FETCH_CAP) -> list[ChatMessage]:
    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(rows))   # back to chronological order


# --------------------------------------------------------------------------- #
# Conversation summary — rolling, bounded, recency-weighted (chatagent.md §5.1) #
# --------------------------------------------------------------------------- #
# When the raw recent-message window exceeds WORKING_TOKENS_LIMIT, the older
# messages are folded into the session's summary and dropped from the window.
# The new summary FOLDS IN the old one but represents newer content in more
# detail, compressing the older summary further to stay inside the cap.

_summary_llm = ChatAnthropic(
    model=MODEL,
    temperature=0,
    max_tokens=SUMMARY_TOKEN_CAP + 200,
    api_key=settings.ANTHROPIC_API_KEY,
)

SUMMARY_SYSTEM_PROMPT = f"""You maintain a running summary of a conversation between a user and an assistant about a requirement-matching analysis.

You are given the PREVIOUS SUMMARY (may be empty) and the OLDER MESSAGES now being folded into it. Produce ONE updated summary that replaces the previous one.

Rules:
- Stay under roughly {SUMMARY_TOKEN_CAP} tokens. This is a hard budget.
- Give MORE detail to the newer content (the messages being folded in) and compress the older previous-summary content further to make room. When you must shrink, shorten the older material first — but never silently drop an important fact.
- Keep concrete specifics that matter later: which requirements were discussed, what the user asked for, what was changed and why, and any stated preferences.
- Drop greetings, pleasantries, and repetition.
- Write plain prose or terse bullets. No preamble — output only the summary."""


async def _summarise(previous_summary: str, older: list[ChatMessage]) -> str:
    convo = "\n".join(f"{m.role}: {m.content}" for m in older)
    user = (
        f"PREVIOUS SUMMARY:\n{previous_summary or '(none)'}\n\n"
        f"OLDER MESSAGES BEING FOLDED IN:\n{convo}"
    )
    resp = await _summary_llm.ainvoke(
        [SystemMessage(SUMMARY_SYSTEM_PROMPT), HumanMessage(user)]
    )
    text = resp.content if isinstance(resp.content, str) else _chunk_text(resp)
    return (text or "").strip()


async def _maybe_summarise(db: Session, session: ChatSession, history: list[ChatMessage]) -> tuple[str, list[ChatMessage]]:
    """Return (summary, working_window). Folds older messages into the session
    summary when the raw window is over budget. Never raises — on failure the
    conversation simply continues with the existing summary and window."""
    summary = session.summary or ""
    total = sum(_approx_tokens(m.content) for m in history)
    if total <= WORKING_TOKENS_LIMIT or len(history) <= KEEP_RECENT_MESSAGES:
        return summary, history

    older = history[:-KEEP_RECENT_MESSAGES]
    recent = history[-KEEP_RECENT_MESSAGES:]
    try:
        new_summary = await _summarise(summary, older)
        if new_summary:
            session.summary = new_summary
            db.commit()
            return new_summary, recent
    except Exception:
        db.rollback()
    return summary, history


# --------------------------------------------------------------------------- #
# Episodic memory — one rolling summary per ANALYSIS (chatagent.md §5.2)        #
# --------------------------------------------------------------------------- #
# Shared across ALL chat sessions of an analysis: this is what lets a brand-new
# session answer "what did we change / decide earlier?" without any transcript.
# Written on every edit; for non-edit turns a small LLM judgment decides. Folded
# in recency-first and re-compressed when it exceeds its cap.

EPISODIC_TOKEN_CAP = 1500

_episodic_llm = ChatAnthropic(
    model=MODEL,
    temperature=0,
    max_tokens=EPISODIC_TOKEN_CAP + 200,
    api_key=settings.ANTHROPIC_API_KEY,
)

EPISODIC_SYSTEM_PROMPT = f"""You maintain the long-term memory of an analysis: the important things that have happened across ALL chat sessions about it.

You are given the CURRENT MEMORY (may be empty) and a NEW EVENT from the latest exchange. Produce ONE updated memory that replaces the current one.

Rules:
- Stay under roughly {EPISODIC_TOKEN_CAP} tokens. This is a hard budget.
- Record only things that matter later: edits made (what changed, on which requirement, and why) and firm user decisions or preferences. Never record greetings, small talk, or ordinary questions that changed nothing.
- Give MORE detail to the new event and compress older entries further to stay in budget. When you must shrink, compress the oldest material first — never silently drop that something happened.
- Keep it factual and specific (name the requirement). No preamble — output only the memory."""


class _ImportanceVerdict(BaseModel):
    important: bool = Field(
        description="True only if this exchange contains a firm decision or preference worth remembering long-term. False for greetings, small talk, and ordinary questions that changed nothing."
    )
    note: str = Field(
        default="", description="If important, one short sentence recording it. Else empty."
    )


_importance_llm = _llm.with_structured_output(_ImportanceVerdict)


async def _judge_importance(user_message: str, assistant_text: str) -> Optional[str]:
    """Small LLM judgment: is this non-edit turn worth remembering? Returns a note or None."""
    try:
        verdict: _ImportanceVerdict = await _importance_llm.ainvoke(
            [
                SystemMessage(
                    "Decide whether an exchange about a requirement-matching analysis is worth "
                    "remembering long-term. Only firm decisions or stated preferences qualify. "
                    "Greetings, small talk, and ordinary questions that changed nothing do NOT."
                ),
                HumanMessage(f"User: {user_message}\n\nAssistant: {assistant_text}"),
            ]
        )
        if verdict.important and verdict.note.strip():
            return verdict.note.strip()
    except Exception:
        pass
    return None


async def _update_episodic(db: Session, project: NewProject, event: str) -> None:
    """Fold one event into the analysis's rolling episodic memory. Never raises."""
    try:
        current = project.episodic_summary or ""
        # Below cap: append directly (cheap, keeps facts verbatim).
        if _approx_tokens(current) + _approx_tokens(event) <= EPISODIC_TOKEN_CAP:
            project.episodic_summary = (current + "\n" + event).strip() if current else event
        else:
            resp = await _episodic_llm.ainvoke(
                [
                    SystemMessage(EPISODIC_SYSTEM_PROMPT),
                    HumanMessage(f"CURRENT MEMORY:\n{current or '(none)'}\n\nNEW EVENT:\n{event}"),
                ]
            )
            text = resp.content if isinstance(resp.content, str) else _chunk_text(resp)
            if text and text.strip():
                project.episodic_summary = text.strip()
        db.commit()
    except Exception:
        db.rollback()


# --------------------------------------------------------------------------- #
# Streaming one turn                                                           #
# --------------------------------------------------------------------------- #

def _chunk_text(chunk) -> str:
    """Pull the human-visible text out of a streamed chunk (string or block list)."""
    c = getattr(chunk, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for block in c:
            if isinstance(block, dict) and block.get("type") == "text":
                out.append(block.get("text", ""))
            elif isinstance(block, str):
                out.append(block)
        return "".join(out)
    return ""


async def stream_turn(
    db: Session,
    project: NewProject,
    session_id: int,
    user_message: str,
    tagged_requirement_id: Optional[int],
) -> AsyncGenerator[dict, None]:
    """Run one chat turn, yielding events: {type:'token'|'status'|'done', ...}."""
    history = load_recent_messages(db, session_id)         # BEFORE saving the new one
    save_message(db, session_id, "user", user_message)

    # Fold older messages into the session summary when the window is over budget.
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    summary, window = (session.summary or "", history)
    if session is not None:
        summary, window = await _maybe_summarise(db, session, history)

    system_text = SYSTEM_PROMPT + "\n\n---\nCURRENT ANALYSIS DATA:\n" + build_snapshot(
        db, project, tagged_requirement_id
    )
    # Episodic memory: what happened across ALL sessions of this analysis. This is
    # what lets a brand-new chat answer "what did we change earlier?".
    if project.episodic_summary:
        system_text += (
            "\n\n---\nWHAT HAS HAPPENED EARLIER IN THIS ANALYSIS (across all chats):\n"
            + project.episodic_summary
        )
    if summary:
        system_text += "\n\n---\nSUMMARY OF EARLIER CONVERSATION:\n" + summary

    messages = [SystemMessage(system_text)]
    for m in window:
        messages.append(HumanMessage(m.content) if m.role == "user" else AIMessage(m.content))
    messages.append(HumanMessage(user_message))

    changed = False
    text_parts: list[str] = []
    edit_notes: list[str] = []   # what the edits did, for episodic memory

    for _ in range(MAX_TOOL_LOOPS):
        full = None
        try:
            async for chunk in _llm_with_tools.astream(messages):
                full = chunk if full is None else full + chunk
                t = _chunk_text(chunk)
                if t:
                    text_parts.append(t)
                    yield {"type": "token", "text": t}
        except Exception:
            yield {"type": "token", "text": "\n\n(Sorry — something went wrong. Please try again.)"}
            break

        if full is None:
            break
        messages.append(full)

        tool_calls = getattr(full, "tool_calls", None) or []
        if not tool_calls:
            break

        yield {"type": "status", "text": "updating requirement…"}
        for tc in tool_calls:
            if tc.get("name") == "update_requirement":
                args = tc.get("args", {})
                res = _apply_update(db, project.id, args)
                if res.get("ok"):
                    changed = True
                    # Note exactly what changed, for the episodic memory.
                    fields = [f for f in ("status", "explanation", "modification") if args.get(f) is not None]
                    edit_notes.append(
                        f"Changed {', '.join(fields) or 'fields'} on requirement "
                        f"'{res.get('name')}' (id {res.get('requirement_id')})"
                        + (f"; status is now {args['status']}" if args.get("status") else "")
                        + f". User asked: \"{user_message}\""
                    )
            else:
                res = {"ok": False, "error": "unknown tool"}
            messages.append(ToolMessage(content=json.dumps(res), tool_call_id=tc.get("id", "")))
        # loop again so the model can produce its confirmation text

    assistant_text = "".join(text_parts).strip() or "(no reply)"
    save_message(db, session_id, "assistant", assistant_text)

    # Episodic memory: always record edits; otherwise let a small judgment decide.
    if edit_notes:
        await _update_episodic(db, project, " ".join(edit_notes))
    else:
        note = await _judge_importance(user_message, assistant_text)
        if note:
            await _update_episodic(db, project, note)

    results = None
    if changed:
        data = get_analysis_results(db, project.id, project.user_id)
        results = data["results"] if data else None

    yield {"type": "done", "changed": changed, "results": results}
