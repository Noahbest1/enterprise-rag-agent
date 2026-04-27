"""Invoice lookup + request tools (DB-backed).

Same pattern as agent/tools/orders.py: return plain dicts so the graph
state stays JSON-serialisable. One invoice per order in this demo —
production would allow split invoices (e.g. per-item for B2B customers).
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from rag.db.base import SessionLocal
from rag.db.models import Invoice, Order


def _invoice_to_dict(inv: Invoice) -> dict:
    # Dynamic download URL: once status is 'issued', link points at our real
    # reportlab-rendered PDF endpoint. Persisted ``download_url`` (e.g. seeded
    # mock URLs) is honoured when present, else we compute one on the fly.
    download_url = inv.download_url
    if inv.status == "issued" and not download_url:
        download_url = f"/invoice/{inv.id}.pdf"
    return {
        "id": inv.id,
        "order_id": inv.order_id,
        "tenant": inv.tenant,
        "title": inv.title,
        "tax_id": inv.tax_id,
        "invoice_type": inv.invoice_type,
        "amount_yuan": inv.amount_cents / 100.0,
        "status": inv.status,
        "download_url": download_url,
        "requested_at": inv.requested_at.isoformat() if inv.requested_at else None,
        "issued_at": inv.issued_at.isoformat() if inv.issued_at else None,
    }


def get_invoice_by_order(order_id: str) -> dict | None:
    with SessionLocal() as s:
        inv = s.execute(
            select(Invoice).where(Invoice.order_id == order_id).order_by(Invoice.id.desc())
        ).scalar_one_or_none()
        return _invoice_to_dict(inv) if inv else None


def request_invoice(
    *, order_id: str, title: str, tax_id: str | None = None,
    invoice_type: str = "electronic",
) -> dict:
    """Create a new invoice in ``requested`` state (or surface the existing
    one if it's already issued / requested, to avoid dup-submit).

    Cancelled invoices DON'T count as a dup — if the user (or admin)
    cancelled an earlier invoice on this order, they're allowed to file a
    fresh request. Without this exclusion the cancelled row would block
    the new one forever, leaving the user with no way to recover.
    """
    with SessionLocal() as s:
        existing = s.execute(
            select(Invoice).where(
                Invoice.order_id == order_id,
                Invoice.status != "cancelled",
            )
        ).scalar_one_or_none()
        if existing is not None:
            return {**_invoice_to_dict(existing), "duplicate": True}

        order = s.get(Order, order_id)
        if order is None:
            return {"error": "order_not_found", "order_id": order_id}
        if order.status not in ("delivered", "shipped"):
            return {
                "error": "order_not_eligible",
                "order_id": order_id,
                "current_status": order.status,
                "message": "仅已发货 / 已签收的订单可开票。",
            }

        inv = Invoice(
            order_id=order_id,
            tenant=order.tenant,
            title=title,
            tax_id=tax_id,
            invoice_type=invoice_type,
            amount_cents=order.total_cents,
            status="requested",
            requested_at=datetime.now(timezone.utc),
        )
        s.add(inv)
        s.commit()
        s.refresh(inv)
        return _invoice_to_dict(inv)


def list_invoices_for_user(user_id: str, *, limit: int = 10) -> list[dict]:
    with SessionLocal() as s:
        rows = s.execute(
            select(Invoice).join(Order, Invoice.order_id == Order.id)
            .where(Order.user_id == user_id)
            .order_by(Invoice.requested_at.desc())
            .limit(limit)
        ).scalars().all()
        return [_invoice_to_dict(inv) for inv in rows]
