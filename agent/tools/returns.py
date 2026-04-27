"""Return / refund business logic.

Rules implemented today:
- 7-day no-reason window from placed_at. "refunded" orders can't be
  returned again. "cancelled" orders can't be returned.
- Eligibility message is in user language (matches the tenant's persona).
- Refund amount equals order total minus (if any) shipping fee placeholder.
- A ``ReturnRequest`` row is persisted on successful creation.

Real platforms layer far more nuance (shipping insurance, damage triage,
cross-store policies). We keep it simple but realistic enough that the
Summary LLM has clean JSON-shaped facts to narrate.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from rag.db.base import SessionLocal
from rag.db.models import Order, ReturnRequest


NO_REASON_WINDOW_DAYS = 7


def _within_no_reason_window(order: Order) -> tuple[bool, int]:
    placed = order.placed_at
    if placed.tzinfo is None:
        placed = placed.replace(tzinfo=timezone.utc)
    elapsed = datetime.now(timezone.utc) - placed
    days_left = NO_REASON_WINDOW_DAYS - elapsed.days
    return days_left > 0, max(days_left, 0)


def check_eligibility(order_id: str) -> dict[str, Any]:
    with SessionLocal() as s:
        order = s.get(Order, order_id)
        if order is None:
            return {"ok": False, "reason": "order_not_found"}

        if order.status == "refunded":
            return {"ok": False, "reason": "already_refunded", "order_id": order.id}
        if order.status == "cancelled":
            return {"ok": False, "reason": "order_cancelled", "order_id": order.id}

        in_window, days_left = _within_no_reason_window(order)
        if not in_window:
            return {
                "ok": False,
                "reason": "out_of_no_reason_window",
                "order_id": order.id,
                "placed_at": order.placed_at.isoformat(),
            }

        # Simple refund: full order total. Real platforms subtract shipping
        # if buyer paid; for mock we always return full.
        return {
            "ok": True,
            "order_id": order.id,
            "tenant": order.tenant,
            "days_left_in_window": days_left,
            "refund_cents": order.total_cents,
            "item_titles": [it.title for it in order.items],
            "current_status": order.status,
        }


def create_return_request(order_id: str, kind: str, reason: str) -> dict[str, Any]:
    """Persist the request. Does not flip the underlying order's status --
    that happens in an async workflow in a real system."""
    if kind not in {"refund", "return", "exchange", "price_protect"}:
        return {"ok": False, "reason": "invalid_kind", "kind": kind}

    with SessionLocal() as s:
        order = s.get(Order, order_id)
        if order is None:
            return {"ok": False, "reason": "order_not_found"}
        req = ReturnRequest(
            order_id=order.id,
            tenant=order.tenant,
            kind=kind,
            reason=reason.strip() or "用户未提供原因",
            status="pending",
            refund_cents=order.total_cents if kind in ("refund", "return") else None,
        )
        s.add(req)
        s.commit()
        return {
            "ok": True,
            "request_id": req.id,
            "status": req.status,
            "kind": req.kind,
            "order_id": req.order_id,
            "refund_cents": req.refund_cents,
            "created_at": req.created_at.isoformat() if req.created_at else None,
        }


def reopen_return_request(request_id: int) -> dict[str, Any] | None:
    """Reverse a previously-cancelled return: flip status back to ``pending``
    and bump ``updated_at``. Refuses non-cancelled rows so an already-active
    request can't be silently re-shuffled mid-conversation.

    Mirrors the complaint reopen pattern: same shape (None for missing,
    ValueError for wrong-state) so the API endpoint can fan out the same
    error codes (404 vs 409).
    """
    with SessionLocal() as s:
        row = s.get(ReturnRequest, request_id)
        if row is None:
            return None
        if row.status != "cancelled":
            raise ValueError(
                f"return request {request_id} is not cancelled (status={row.status})"
            )
        row.status = "pending"
        s.commit()
        s.refresh(row)
        return {
            "ok": True,
            "request_id": row.id,
            "status": row.status,
            "kind": row.kind,
            "order_id": row.order_id,
            "refund_cents": row.refund_cents,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }


def list_requests_for_order(order_id: str) -> list[dict[str, Any]]:
    with SessionLocal() as s:
        from sqlalchemy import select
        rows = s.execute(
            select(ReturnRequest).where(ReturnRequest.order_id == order_id)
        ).scalars().all()
        return [
            {
                "request_id": r.id,
                "kind": r.kind,
                "status": r.status,
                "reason": r.reason,
                "refund_cents": r.refund_cents,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
