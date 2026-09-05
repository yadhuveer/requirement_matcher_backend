"""
Chat endpoints for the per-analysis assistant (chatagent.md).

Additive — mounted under the same /api/analyse prefix, but only handles the
/{project_id}/chat/... paths. Every call is scoped to the signed-in user and the
analysis they own. The message endpoint streams the reply over SSE.
"""

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.services.auth_dependency import get_current_user
from app.models.db_models import User, NewProject, ChatSession, ChatMessage
from app.services.chat_agent import stream_turn, create_session, load_recent_messages

router = APIRouter(prefix="/api/analyse")


class ChatMessageIn(BaseModel):
    message: str
    requirement_id: Optional[int] = None   # the tagged requirement, if any


def _own_project(db: Session, project_id: int, user_id: int) -> NewProject:
    project = (
        db.query(NewProject)
        .filter(NewProject.id == project_id, NewProject.user_id == user_id)
        .first()
    )
    if project is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return project


# NOTE: literal paths ("session", "sessions") are declared BEFORE the
# "{session_id}" param routes so they match first.

@router.post("/{project_id}/chat/session")
async def start_chat_session(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _own_project(db, project_id, current_user.id)
    s = create_session(db, project_id)
    return {"session_id": s.id}


@router.get("/{project_id}/chat/sessions")
async def list_chat_sessions(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _own_project(db, project_id, current_user.id)
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.project_id == project_id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )
    return {
        "sessions": [
            {"session_id": s.id, "created_at": s.created_at.isoformat() if s.created_at else None}
            for s in sessions
        ]
    }


@router.get("/{project_id}/chat/{session_id}")
async def get_chat_messages(
    project_id: int,
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _own_project(db, project_id, current_user.id)
    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.project_id == project_id)
        .first()
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    msgs = load_recent_messages(db, session_id, limit=1000)
    return {
        "messages": [
            {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat() if m.created_at else None}
            for m in msgs
        ]
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/{project_id}/chat/{session_id}")
async def post_chat_message(
    project_id: int,
    session_id: int,
    body: ChatMessageIn,
    current_user: User = Depends(get_current_user),
):
    """Stream the assistant's reply (SSE): status → token… → done{changed, results?}."""
    user_id = current_user.id

    async def event_gen():
        # Fresh session for the (possibly multi-second) stream, closed at the end.
        db = SessionLocal()
        try:
            project = (
                db.query(NewProject)
                .filter(NewProject.id == project_id, NewProject.user_id == user_id)
                .first()
            )
            if project is None:
                yield _sse("error", {"detail": "Analysis not found"})
                return
            session = (
                db.query(ChatSession)
                .filter(ChatSession.id == session_id, ChatSession.project_id == project_id)
                .first()
            )
            if session is None:
                yield _sse("error", {"detail": "Chat not found"})
                return
            async for ev in stream_turn(db, project, session_id, body.message, body.requirement_id):
                yield _sse(ev.get("type", "message"), ev)
        finally:
            db.close()

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
