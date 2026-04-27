"""Admin-side routes: list / claim / reply on complaint tickets.

Resume-project auth posture: when ``REQUIRE_API_KEY=false`` (dev default)
any caller can hit these. In prod, a real deployment would gate admin
routes on an ``admin`` scope inside the api_keys row. Out of scope for
this step; the structure is ready for it (``require_api_key`` is already
a Depends here so adding a scope check is a one-liner later).

Each reply publishes a ``complaint_reply`` event onto the in-process user
bus so the target user's live AgentChat injects a system bubble in real
time. Persistence + event publish are sequential: if the event publish
fails (subscriber queue full etc.) the reply is still in the DB.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from agent.tools.complaints import (
    add_reply,
    claim_complaint,
    get_complaint,
    list_complaints,
    list_replies,
)

from .audit import hash16, record_audit
from .auth import AuthContext, require_api_key
from .user_events import publish_user_event


router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/complaints")
def admin_list_complaints(
    tenant: str | None = None,
    status: str | None = None,
    only_escalated: bool = False,
    limit: int = 50,
    auth: AuthContext = Depends(require_api_key),
):
    """List complaints for the admin queue, newest first. When the caller
    is authed, ``tenant`` is forced to their api-key tenant so one tenant's
    admin can't peek at another tenant's tickets."""
    effective_tenant = tenant
    if auth.is_authenticated:
        effective_tenant = auth.tenant_id

    items = list_complaints(
        tenant=effective_tenant,
        status=status,
        only_escalated=only_escalated,
        limit=limit,
    )
    return {"count": len(items), "items": items}


@router.get("/complaints/{complaint_id}")
def admin_get_complaint(
    complaint_id: int,
    auth: AuthContext = Depends(require_api_key),
):
    """Fetch a single complaint + its reply history."""
    c = get_complaint(complaint_id)
    if c is None:
        raise HTTPException(status_code=404, detail="complaint not found")
    if auth.is_authenticated and c.get("tenant") != auth.tenant_id:
        raise HTTPException(status_code=404, detail="complaint not found")
    replies = list_replies(complaint_id)
    return {"complaint": c, "replies": replies}


class ClaimComplaintRequest(BaseModel):
    assigned_to: str = Field(..., min_length=1, max_length=64)


@router.post("/complaints/{complaint_id}/claim")
def admin_claim_complaint(
    complaint_id: int,
    req: ClaimComplaintRequest,
    auth: AuthContext = Depends(require_api_key),
):
    """Admin takes ownership. Status flips to ``escalated`` (if still open)
    + ``assigned_to`` is set. Writes a ``complaint_claimed`` audit row."""
    before = get_complaint(complaint_id)
    if before is None:
        raise HTTPException(status_code=404, detail="complaint not found")
    if auth.is_authenticated and before.get("tenant") != auth.tenant_id:
        raise HTTPException(status_code=404, detail="complaint not found")

    updated = claim_complaint(complaint_id, assigned_to=req.assigned_to)
    assert updated is not None  # got here, so row exists

    record_audit(
        event_type="complaint_claimed",
        tenant_id=updated.get("tenant"),
        user_id=None,  # admin action, not tied to end-user PII
        extra={
            "complaint_id": complaint_id,
            "assigned_to": req.assigned_to,
            "previous_status": before.get("status"),
        },
    )
    return {"ok": True, "complaint": updated}


class ReplyComplaintRequest(BaseModel):
    author_label: str = Field(..., min_length=1, max_length=64)
    content: str = Field(..., min_length=1, max_length=4000)


@router.post("/complaints/{complaint_id}/reply")
def admin_reply_to_complaint(
    complaint_id: int,
    req: ReplyComplaintRequest,
    auth: AuthContext = Depends(require_api_key),
):
    """Admin posts a reply. Persists the reply + publishes a
    ``complaint_reply`` event on the user bus + writes a
    ``complaint_replied`` audit row (content hashed, never raw)."""
    parent = get_complaint(complaint_id)
    if parent is None:
        raise HTTPException(status_code=404, detail="complaint not found")
    if auth.is_authenticated and parent.get("tenant") != auth.tenant_id:
        raise HTTPException(status_code=404, detail="complaint not found")
    # Defense in depth: the UI hides the reply box on closed complaints,
    # but a direct API call shouldn't sneak a message into a ticket the
    # user explicitly closed.
    if parent.get("status") == "closed":
        raise HTTPException(status_code=409, detail="complaint is closed")

    reply = add_reply(
        complaint_id,
        content=req.content,
        author_label=req.author_label,
        author_kind="admin",
    )
    if reply is None:
        raise HTTPException(status_code=400, detail="empty reply or complaint gone")

    target_user = reply.get("user_id")
    delivered = 0
    if target_user:
        delivered = publish_user_event(
            target_user,
            "complaint_reply",
            {
                "complaint_id": complaint_id,
                "reply_id": reply["id"],
                "author_label": reply["author_label"],
                "content": reply["content"],
                "created_at": reply["created_at"],
                # The session this complaint was filed in. Frontend SSE
                # handler uses it to decide whether to render the reply in
                # the current session or just toast (cross-session reply).
                # NULL when the parent session was deleted, in which case
                # the reply has no home — frontend just toasts.
                "thread_id": parent.get("thread_id"),
            },
        )

    record_audit(
        event_type="complaint_replied",
        tenant_id=parent.get("tenant"),
        user_id=None,
        extra={
            "complaint_id": complaint_id,
            "reply_id": reply["id"],
            "author_label": req.author_label,
            "content_hash": hash16(req.content),
            "user_id_hash": hash16(target_user) if target_user else None,
            "subscribers_notified": delivered,
        },
    )
    return {
        "ok": True,
        "reply": reply,
        "subscribers_notified": delivered,
    }
