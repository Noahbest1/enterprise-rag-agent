"""Complaint DB tools: create / escalate / list.

SLA defaults (hours) keyed on severity. Production tunes these per tenant
contract. Returning plain dicts so graph state stays JSON-serialisable.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from rag.db.base import SessionLocal
from rag.db.models import Complaint, ComplaintReply


def _iso_utc(dt: datetime | None) -> str | None:
    """Always-UTC ISO-8601. SQLite stores DateTime(timezone=True) values
    naive; if we just call ``isoformat()`` the frontend's ``new Date(...)``
    interprets the result as LOCAL time and the display drifts by the
    user's timezone offset (we hit a ~1h skew when admin and user replies
    interleaved, because optimistic inserts used ``Date.now()`` while server
    data was reparsed as local). Force a ``+00:00`` suffix so JS Date
    parsing is unambiguous on every browser."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


SLA_HOURS = {"high": 1, "medium": 4, "low": 24}


def _complaint_to_dict(c: Complaint) -> dict:
    return {
        "id": c.id,
        "user_id": c.user_id,
        "tenant": c.tenant,
        # The ChatSession that filed this complaint (NULL if filed before
        # sessions existed, or if the parent session was deleted). Frontend
        # uses this to scope SSE complaint_reply events to the right session.
        "thread_id": c.thread_id,
        "order_id": c.order_id,
        "topic": c.topic,
        "severity": c.severity,
        "content_hash": c.content_hash,
        "status": c.status,
        "escalated": bool(c.escalated),
        "assigned_to": c.assigned_to,
        "sla_due_at": _iso_utc(c.sla_due_at),
        "created_at": _iso_utc(c.created_at),
    }


def _pick_human_agent(tenant: str) -> str:
    # Mock routing; production hits the CRM assignment API. Two fake
    # senior agents per tenant so repeated escalations don't all pile up
    # on the same person in the demo.
    pool = {
        "jd": ["jd-cs-senior-A", "jd-cs-senior-B"],
        "taobao": ["tb-cs-senior-A", "tb-cs-senior-B"],
    }.get(tenant, ["cs-senior-A"])
    # deterministic-but-rotating: based on current second so two calls in a
    # row usually hit different agents without needing external state.
    idx = datetime.now(timezone.utc).second % len(pool)
    return pool[idx]


def create_complaint(
    *,
    user_id: str | None,
    tenant: str,
    order_id: str | None,
    topic: str,
    severity: str,
    content: str,
) -> dict:
    """Insert one complaint row. If severity==high, auto-escalate."""
    now = datetime.now(timezone.utc)
    sla_hours = SLA_HOURS.get(severity, SLA_HOURS["low"])
    sla_due = now + timedelta(hours=sla_hours)

    escalated = severity == "high"
    assigned = _pick_human_agent(tenant) if escalated else None
    status = "escalated" if escalated else "open"

    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    with SessionLocal() as s:
        c = Complaint(
            user_id=user_id,
            tenant=tenant,
            order_id=order_id,
            topic=topic,
            severity=severity,
            content_hash=content_hash,
            status=status,
            escalated=escalated,
            assigned_to=assigned,
            sla_due_at=sla_due,
        )
        s.add(c)
        s.commit()
        s.refresh(c)
        return _complaint_to_dict(c)


def reopen_complaint(complaint_id: int) -> dict | None:
    """Re-activate a previously-closed complaint.

    Restores the appropriate "active" state based on severity:
      - severity=high → status=escalated + escalated=true + freshly assigned
        to a human CS agent + 1h SLA. (Mirrors create_complaint's escalation
        logic, since the original severity hasn't changed.)
      - severity=medium/low → status=open + escalated=false + new SLA window.

    Returns None if the complaint doesn't exist; raises ValueError if the
    complaint isn't currently closed (avoid double-reopen mid-conversation).
    """
    now = datetime.now(timezone.utc)
    with SessionLocal() as s:
        c = s.get(Complaint, complaint_id)
        if c is None:
            return None
        if c.status != "closed":
            raise ValueError(f"complaint {complaint_id} is not closed (status={c.status})")
        if c.severity == "high":
            c.status = "escalated"
            c.escalated = True
            # Re-pick a human agent — original assignee may have moved on.
            c.assigned_to = _pick_human_agent(c.tenant)
            c.sla_due_at = now + timedelta(hours=SLA_HOURS["high"])
        else:
            c.status = "open"
            c.escalated = False
            sla_hours = SLA_HOURS.get(c.severity, SLA_HOURS["low"])
            c.sla_due_at = now + timedelta(hours=sla_hours)
        s.commit()
        s.refresh(c)
        return _complaint_to_dict(c)


def escalate_complaint(complaint_id: int, *, assigned_to: str | None = None) -> dict | None:
    with SessionLocal() as s:
        c = s.get(Complaint, complaint_id)
        if c is None:
            return None
        c.escalated = True
        c.status = "escalated"
        c.assigned_to = assigned_to or _pick_human_agent(c.tenant)
        s.commit()
        s.refresh(c)
        return _complaint_to_dict(c)


def get_complaint(complaint_id: int) -> dict | None:
    with SessionLocal() as s:
        c = s.get(Complaint, complaint_id)
        return _complaint_to_dict(c) if c else None


def list_complaints_for_user(user_id: str, *, limit: int = 10) -> list[dict]:
    with SessionLocal() as s:
        rows = s.execute(
            select(Complaint).where(Complaint.user_id == user_id)
            .order_by(Complaint.created_at.desc()).limit(limit)
        ).scalars().all()
        return [_complaint_to_dict(c) for c in rows]


def list_complaints(
    *,
    tenant: str | None = None,
    status: str | None = None,
    only_escalated: bool = False,
    limit: int = 50,
) -> list[dict]:
    """Admin-facing query -- newest first. Filterable by tenant / status /
    only_escalated. The admin dashboard uses this to populate its queue."""
    with SessionLocal() as s:
        stmt = select(Complaint)
        if tenant:
            stmt = stmt.where(Complaint.tenant == tenant)
        if status:
            stmt = stmt.where(Complaint.status == status)
        if only_escalated:
            stmt = stmt.where(Complaint.escalated == True)  # noqa: E712
        stmt = stmt.order_by(
            Complaint.created_at.desc(),
            Complaint.id.desc(),  # tiebreaker when two rows land in the same second
        ).limit(max(1, min(int(limit), 200)))
        rows = s.execute(stmt).scalars().all()
        return [_complaint_to_dict(c) for c in rows]


def claim_complaint(complaint_id: int, *, assigned_to: str) -> dict | None:
    """Admin marks themselves as the owner of a complaint. Flips status to
    ``escalated`` if it was still ``open`` (taking a low/medium ticket off
    the auto-SLA queue). Returns None if the row doesn't exist."""
    with SessionLocal() as s:
        c = s.get(Complaint, complaint_id)
        if c is None:
            return None
        c.assigned_to = assigned_to
        c.escalated = True
        if c.status == "open":
            c.status = "escalated"
        s.commit()
        s.refresh(c)
        return _complaint_to_dict(c)


def add_reply(
    complaint_id: int,
    *,
    content: str,
    author_label: str,
    author_kind: str = "admin",
) -> dict | None:
    """Persist one admin (or system) reply on the thread. Returns the
    created reply dict + the parent complaint's user_id so the caller can
    publish a targeted SSE event to that user."""
    content = content.strip()
    if not content:
        return None
    with SessionLocal() as s:
        c = s.get(Complaint, complaint_id)
        if c is None:
            return None
        reply = ComplaintReply(
            complaint_id=complaint_id,
            author_kind=author_kind,
            author_label=author_label,
            content=content,
        )
        s.add(reply)
        s.commit()
        s.refresh(reply)
        return {
            "id": reply.id,
            "complaint_id": reply.complaint_id,
            "author_kind": reply.author_kind,
            "author_label": reply.author_label,
            "content": reply.content,
            "created_at": _iso_utc(reply.created_at),
            "user_id": c.user_id,
            "tenant": c.tenant,
        }


def list_replies(complaint_id: int) -> list[dict]:
    with SessionLocal() as s:
        rows = s.execute(
            select(ComplaintReply)
            .where(ComplaintReply.complaint_id == complaint_id)
            .order_by(ComplaintReply.created_at.asc())
        ).scalars().all()
        return [
            {
                "id": r.id,
                "complaint_id": r.complaint_id,
                "author_kind": r.author_kind,
                "author_label": r.author_label,
                "content": r.content,
                "created_at": _iso_utc(r.created_at),
            }
            for r in rows
        ]
