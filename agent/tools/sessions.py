"""ChatSession metadata helpers — the thin layer between the agent layer
and the ``chat_sessions`` table.

Concept: each LangGraph ``thread_id`` (the AsyncSqliteSaver key, persisted
in `data/langgraph.sqlite`) is paired with one row in this table that
records ``user_id``, ``tenant``, a display ``title``, and ``last_msg_at``.
The frontend uses these rows to power the ChatGPT-style session sidebar.

The actual conversation state still lives in the LangGraph checkpoint;
this table is metadata only. Deleting a session means clearing both:
the row here, plus the checkpoint (handled by the API endpoint).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from rag.db.base import SessionLocal
from rag.db.models import ChatSession


_TITLE_MAX = 40


def _iso_utc(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _to_dict(s: ChatSession) -> dict[str, Any]:
    return {
        "thread_id": s.thread_id,
        "user_id": s.user_id,
        "tenant": s.tenant,
        "title": s.title,
        "first_msg_at": _iso_utc(s.first_msg_at),
        "last_msg_at": _iso_utc(s.last_msg_at),
    }


def touch_session(
    *,
    thread_id: str,
    user_id: str,
    tenant: str,
    first_message: str | None = None,
) -> dict[str, Any]:
    """Upsert: create the session if missing, otherwise bump ``last_msg_at``.

    On creation, ``title`` is auto-populated from ``first_message[:40]``
    so the sidebar shows something more useful than "(新会话)" once the
    user has typed even one query.
    """
    with SessionLocal() as s:
        existing = s.get(ChatSession, thread_id)
        if existing is None:
            title = (first_message or "").strip()[:_TITLE_MAX] or "(新会话)"
            row = ChatSession(
                thread_id=thread_id,
                user_id=user_id,
                tenant=tenant,
                title=title,
            )
            s.add(row)
            s.commit()
            s.refresh(row)
            return _to_dict(row)
        # Keep title as-is (user may have renamed it). Just bump last_msg_at.
        existing.last_msg_at = datetime.now(timezone.utc)
        s.commit()
        s.refresh(existing)
        return _to_dict(existing)


def list_sessions_for_user(user_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
    """Newest first, capped. Most callers only need the last 50 anyway."""
    with SessionLocal() as s:
        rows = s.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.last_msg_at.desc())
            .limit(max(1, min(int(limit), 200)))
        ).scalars().all()
        return [_to_dict(r) for r in rows]


def get_session(thread_id: str) -> dict[str, Any] | None:
    with SessionLocal() as s:
        row = s.get(ChatSession, thread_id)
        return _to_dict(row) if row else None


def rename_session(thread_id: str, *, title: str) -> dict[str, Any] | None:
    """User-initiated rename. Title is clamped to 80 chars (column limit)."""
    title = (title or "").strip()[:80]
    if not title:
        return None
    with SessionLocal() as s:
        row = s.get(ChatSession, thread_id)
        if row is None:
            return None
        row.title = title
        s.commit()
        s.refresh(row)
        return _to_dict(row)


def delete_session(thread_id: str) -> bool:
    """Delete the metadata row only. The caller (API endpoint) is responsible
    for also clearing the LangGraph checkpoint and NULL-ing
    ``complaints.thread_id`` so deletion is consistent across systems."""
    with SessionLocal() as s:
        row = s.get(ChatSession, thread_id)
        if row is None:
            return False
        s.delete(row)
        s.commit()
        return True
