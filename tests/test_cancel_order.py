"""Order cancellation (self-service before shipment) tests.

Status-aware actions:
- placed / paid → user-cancelable, full refund, no human needed.
- shipped / delivered / refunded / cancelled → 409 with a specific reason
  so the front-end can fall back to the complaint-escalation flow.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent.tools.orders import cancel_order
from rag.db.base import SessionLocal
from rag.db.models import Order


@pytest.fixture()
def seeded(seeded_db):
    return seeded_db


# ---------- helper unit tests ----------

def test_cancel_order_succeeds_for_placed(seeded):
    r = cancel_order("JD20260422789")  # status=placed in seed
    assert r["ok"] is True
    assert r["previous_status"] == "placed"
    assert r["current_status"] == "cancelled"
    assert r["refunded_cents"] > 0


def test_cancel_order_persists_status_change(seeded):
    cancel_order("JD20260422789")
    with SessionLocal() as s:
        o = s.get(Order, "JD20260422789")
        assert o.status == "cancelled"


def test_cancel_order_rejects_shipped(seeded):
    r = cancel_order("JD20260418123")  # status=shipped
    assert r["ok"] is False
    assert r["reason"] == "not_cancellable_after_shipping"
    assert r["current_status"] == "shipped"


def test_cancel_order_rejects_delivered(seeded):
    r = cancel_order("JD20260420456")  # status=delivered
    assert r["ok"] is False
    assert r["reason"] == "not_cancellable_after_shipping"


def test_cancel_order_rejects_refunded(seeded):
    r = cancel_order("TB20260421002")  # status=refunded
    assert r["ok"] is False
    assert r["reason"] == "already_refunded"


def test_cancel_order_idempotent_already_cancelled(seeded):
    cancel_order("JD20260422789")
    r = cancel_order("JD20260422789")  # try again
    assert r["ok"] is False
    assert r["reason"] == "already_cancelled"


def test_cancel_order_unknown_id(seeded):
    r = cancel_order("DOES_NOT_EXIST")
    assert r["ok"] is False
    assert r["reason"] == "order_not_found"


# ---------- endpoint tests ----------

@pytest.fixture
def client():
    from rag_api.main import app
    return TestClient(app)


def test_cancel_order_endpoint_happy_path(client, seeded):
    resp = client.post("/agent/actions/cancel-order", json={"order_id": "JD20260422789"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["result"]["current_status"] == "cancelled"


def test_cancel_order_endpoint_409_on_shipped(client, seeded):
    resp = client.post("/agent/actions/cancel-order", json={"order_id": "JD20260418123"})
    assert resp.status_code == 409
    assert resp.json()["detail"] == "not_cancellable_after_shipping"


def test_cancel_order_endpoint_writes_audit(client, seeded):
    """Successful cancel must leave an audit row with the previous status
    + refund cents (hash-only blast radius story)."""
    from rag.db.models import AuditLog

    client.post("/agent/actions/cancel-order", json={"order_id": "JD20260422789"})

    with SessionLocal() as s:
        rows = s.query(AuditLog).filter(AuditLog.event_type == "order_cancelled").all()
        assert len(rows) == 1
        extra = rows[0].extra or {}
        assert extra.get("previous_status") == "placed"
        assert extra.get("refunded_cents", 0) > 0


# ---------- eligibility escalate_action mapping ----------

def test_eligibility_returns_escalate_action_per_reason(seeded):
    """check_eligibility now returns escalate_action telling the front-end
    which button to render. Status-aware mapping."""
    from agent.tools.returns import check_eligibility

    # placed → cancel
    r = check_eligibility("JD20260422789")
    assert r["ok"] is False
    assert r["reason"] == "not_yet_shipped"
    assert r["escalate_action"] == "cancel_order"

    # shipped → escalate intercept
    r = check_eligibility("JD20260418123")
    assert r["ok"] is False
    assert r["reason"] == "not_yet_delivered"
    assert r["escalate_action"] == "escalate_intercept"

    # delivered + in window → self-service
    r = check_eligibility("JD20260420456")
    assert r["ok"] is True
    assert r["escalate_action"] == "self_service_return"

    # refunded → no action available
    r = check_eligibility("TB20260421002")
    assert r["ok"] is False
    assert r["escalate_action"] is None
