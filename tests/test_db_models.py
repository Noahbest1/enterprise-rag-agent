"""Schema + CRUD tests for the DB models."""
from __future__ import annotations

from datetime import datetime, timezone

from rag.db.models import Feedback, Order, OrderItem, User


def test_tables_exist(db_session):
    from sqlalchemy import inspect
    insp = inspect(db_session.bind)
    names = set(insp.get_table_names())
    # alembic_version isn't there because tests bypass alembic and use create_all
    assert {"users", "orders", "order_items", "feedback", "kb_metadata"} <= names


def test_user_order_items_roundtrip(db_session):
    user = User(id="u1", tenant="jd", display_name="test user")
    db_session.add(user)

    order = Order(
        id="O1",
        user_id="u1",
        tenant="jd",
        status="shipped",
        total_cents=12345,
        placed_at=datetime.now(timezone.utc),
    )
    order.items = [
        OrderItem(sku="sku1", title="Widget", qty=2, unit_price_cents=5000),
        OrderItem(sku="sku2", title="Gadget", qty=1, unit_price_cents=2345),
    ]
    db_session.add(order)
    db_session.commit()

    reloaded = db_session.get(Order, "O1")
    assert reloaded is not None
    assert reloaded.user.id == "u1"
    assert len(reloaded.items) == 2
    assert sum(it.qty * it.unit_price_cents for it in reloaded.items) == 12345


def test_feedback_insert(db_session):
    f = Feedback(
        trace_id="tr1",
        kb_id="jd_demo",
        query="Q",
        answer="A",
        verdict="up",
    )
    db_session.add(f)
    db_session.commit()
    assert f.id is not None
    assert f.created_at is not None


def test_feedback_verdict_strict_usage(db_session):
    """Verdict column is a plain string at the DB layer, but the API enforces up/down.
    Make sure invalid verdicts still insert at the ORM level (protected only at API)."""
    f = Feedback(trace_id="tr2", kb_id="kb", query="q", answer="a", verdict="maybe")
    db_session.add(f)
    db_session.commit()
    assert f.id is not None
