"""Order + logistics lookup tools (DB-backed).

Thin wrappers over SQLAlchemy so the specialist nodes stay short. Return
plain dicts -- never ORM objects -- so the state graph can serialise them.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import desc, select

from rag.db.base import SessionLocal
from rag.db.models import Order, User


# Order statuses that the user can self-cancel without involving CS.
# Once the order leaves the warehouse (shipped / delivered) cancellation
# is no longer "free" — the user has to go through the complaint /
# refund flow with human agents.
CANCELLABLE_STATUSES = {"placed", "paid"}


def cancel_order(order_id: str) -> dict:
    """Self-service cancel: only allowed before shipment.

    Returns:
        ok=True path: {"ok": True, "order_id", "previous_status",
        "current_status": "cancelled", "refunded_cents"}
        ok=False path: {"ok": False, "reason", ...}

    The 4 reason codes mirror the eligibility ones so the front-end can
    use the same renderer:
    - "order_not_found"
    - "already_cancelled"
    - "already_refunded"
    - "not_cancellable_after_shipping" — past the self-service window;
      the user should escalate to a human via complaint.
    """
    with SessionLocal() as s:
        o = s.get(Order, order_id)
        if o is None:
            return {"ok": False, "reason": "order_not_found"}
        if o.status == "cancelled":
            return {"ok": False, "reason": "already_cancelled", "order_id": o.id}
        if o.status == "refunded":
            return {"ok": False, "reason": "already_refunded", "order_id": o.id}
        if o.status not in CANCELLABLE_STATUSES:
            return {
                "ok": False,
                "reason": "not_cancellable_after_shipping",
                "order_id": o.id,
                "current_status": o.status,
            }

        previous_status = o.status
        o.status = "cancelled"
        # In a real platform the refund timing would depend on payment
        # method. For demo we just record cents so callers can show
        # "全额退款 ¥XXX" feedback.
        refunded_cents = o.total_cents
        s.commit()
        return {
            "ok": True,
            "order_id": o.id,
            "previous_status": previous_status,
            "current_status": "cancelled",
            "refunded_cents": refunded_cents,
            "cancelled_at": datetime.now(timezone.utc).isoformat(),
        }


def _order_to_dict(o: Order) -> dict:
    return {
        "id": o.id,
        "user_id": o.user_id,
        "tenant": o.tenant,
        "status": o.status,
        "total_yuan": o.total_cents / 100.0,
        "currency": o.currency,
        "tracking_no": o.tracking_no,
        "carrier": o.carrier,
        "placed_at": o.placed_at.isoformat() if o.placed_at else None,
        "items": [
            {
                "sku": it.sku,
                "title": it.title,
                "qty": it.qty,
                "unit_price_yuan": it.unit_price_cents / 100.0,
            }
            for it in o.items
        ],
    }


def get_order(order_id: str) -> dict | None:
    with SessionLocal() as s:
        o = s.get(Order, order_id)
        return _order_to_dict(o) if o else None


def list_user_orders(user_id: str, tenant: str, limit: int = 10) -> list[dict]:
    with SessionLocal() as s:
        rows = s.execute(
            select(Order)
            .where(Order.user_id == user_id, Order.tenant == tenant)
            .order_by(desc(Order.placed_at))
            .limit(limit)
        ).scalars().all()
        return [_order_to_dict(o) for o in rows]


def find_recent_order_with_keyword(user_id: str, tenant: str, keyword: str) -> dict | None:
    """Naive LIKE against order_items.title -- good enough for agent demo.

    In production this would be a dedicated search index, but for Order
    lookup by product mention ("上周买的 MacBook") LIKE is fine at demo scale.
    """
    kw = (keyword or "").strip()
    if not kw:
        return None
    orders = list_user_orders(user_id, tenant, limit=50)
    for o in orders:
        for it in o["items"]:
            if kw.lower() in it["title"].lower():
                return o
    return None


# --- Logistics ---
# In real life this would hit a carrier API (SF / YTO / ZT / JD express...).
# For demo we synthesise believable status rows from stored fields.


def track_package(tracking_no: str, carrier: str | None = None) -> dict | None:
    """Return mock tracking timeline."""
    from datetime import datetime, timedelta, timezone

    with SessionLocal() as s:
        match = s.execute(
            select(Order).where(Order.tracking_no == tracking_no)
        ).scalar_one_or_none()
    if match is None:
        return None

    now = datetime.now(timezone.utc)
    placed = match.placed_at or (now - timedelta(days=2))

    status = match.status
    timeline = [
        {"ts": placed.isoformat(), "event": "已揽收"},
        {"ts": (placed + timedelta(hours=6)).isoformat(), "event": f"已发往{match.carrier or '分拨中心'}"},
    ]
    if status in ("shipped", "delivered", "refunded"):
        timeline.append({"ts": (placed + timedelta(hours=20)).isoformat(), "event": "运输中"})
    if status in ("delivered",):
        timeline.append({"ts": (placed + timedelta(days=2)).isoformat(), "event": "已签收"})
    if status == "refunded":
        timeline.append({"ts": (placed + timedelta(days=3)).isoformat(), "event": "退回仓库,已退款"})

    eta = None
    if status in ("placed",):
        eta = (now + timedelta(days=2)).isoformat()
    elif status == "shipped":
        eta = (now + timedelta(days=1)).isoformat()

    return {
        "tracking_no": tracking_no,
        "carrier": match.carrier or carrier,
        "status": status,
        "eta": eta,
        "timeline": timeline,
    }
