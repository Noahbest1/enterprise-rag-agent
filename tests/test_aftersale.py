"""AfterSale specialist + return request workflow."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent.specialists.aftersale import aftersale_node
from agent.state import AgentState
from agent.tools.returns import (
    check_eligibility,
    create_return_request,
    list_requests_for_order,
)
from rag.db.base import SessionLocal
from rag.db.models import Order


@pytest.fixture()
def seeded(seeded_db):
    return seeded_db


def test_check_eligibility_ok_for_recent_delivered(seeded):
    # delivered, 3 days old → estimated signed = today → window has full 7 days
    r = check_eligibility("JD20260420456")
    assert r["ok"] is True
    assert r["days_left_in_window"] >= 1
    assert r["refund_cents"] > 0


def test_check_eligibility_rejects_refunded(seeded):
    r = check_eligibility("TB20260421002")  # status=refunded
    assert r["ok"] is False
    assert r["reason"] == "already_refunded"


def test_check_eligibility_shipped_not_yet_delivered(seeded):
    """Shipped (in transit) — window hasn't started, separate reason from
    'past the window' so the UI can show an accurate message."""
    r = check_eligibility("JD20260418123")  # status=shipped
    assert r["ok"] is False
    assert r["reason"] == "not_yet_delivered"


def test_check_eligibility_placed_not_yet_shipped(seeded):
    """Placed but not shipped — even earlier than not_yet_delivered."""
    r = check_eligibility("JD20260422789")  # status=placed
    assert r["ok"] is False
    assert r["reason"] == "not_yet_shipped"


def test_check_eligibility_rejects_out_of_window(seeded):
    """Past the 7-day window from estimated sign date (placed_at + 3 days)."""
    with SessionLocal() as s:
        o = s.get(Order, "JD20260420456")  # already delivered
        # Backdate so estimated sign was > 7 days ago.
        o.placed_at = datetime.now(timezone.utc) - timedelta(days=15)
        s.commit()

    r = check_eligibility("JD20260420456")
    assert r["ok"] is False
    assert r["reason"] == "out_of_no_reason_window"


def test_check_eligibility_window_uses_signed_estimate_not_placed(seeded):
    """Order placed 9 days ago + delivered: with the OLD logic this would be
    out_of_window (placed + 7 = -2 days left). With the new logic, estimated
    signed = placed + 3 = 6 days ago, days_left = 7 - 6 = 1 → still in
    window. This is the bug the user observed and we just fixed."""
    with SessionLocal() as s:
        o = s.get(Order, "JD20260420456")
        o.placed_at = datetime.now(timezone.utc) - timedelta(days=9)
        s.commit()

    r = check_eligibility("JD20260420456")
    assert r["ok"] is True, f"expected eligible but got {r}"
    assert r["days_left_in_window"] >= 1


def test_create_return_request_persists(seeded):
    r = create_return_request("JD20260420456", "refund", "试用后发现不合适")
    assert r["ok"] is True
    assert r["status"] == "pending"

    rows = list_requests_for_order("JD20260420456")
    assert len(rows) == 1
    assert rows[0]["kind"] == "refund"
    assert rows[0]["reason"] == "试用后发现不合适"


def test_reopen_return_request_flips_cancelled_back_to_pending(seeded):
    """Same reversibility pattern as complaints: a cancelled return can
    be reopened so the user doesn't get stuck on a dead row."""
    from agent.tools.returns import reopen_return_request
    from rag.db.models import ReturnRequest

    r = create_return_request("JD20260420456", "refund", "试一下")
    assert r["ok"]
    rid = r["request_id"]
    # Cancel
    with SessionLocal() as s:
        s.get(ReturnRequest, rid).status = "cancelled"
        s.commit()
    # Reopen
    reopened = reopen_return_request(rid)
    assert reopened["status"] == "pending"
    assert reopened["request_id"] == rid


def test_reopen_return_request_rejects_non_cancelled(seeded):
    from agent.tools.returns import reopen_return_request
    r = create_return_request("JD20260420456", "refund", "still active")
    try:
        reopen_return_request(r["request_id"])
    except ValueError as e:
        assert "not cancelled" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_reopen_return_request_endpoint(seeded):
    from fastapi.testclient import TestClient
    from rag.db.models import ReturnRequest
    from rag_api.main import app

    client = TestClient(app)
    r = create_return_request("JD20260420456", "refund", "endpoint test")
    rid = r["request_id"]

    # 409 on non-cancelled
    resp = client.post("/agent/actions/reopen-return-request", json={"request_id": rid})
    assert resp.status_code == 409

    # Cancel + reopen via endpoint
    with SessionLocal() as s:
        s.get(ReturnRequest, rid).status = "cancelled"
        s.commit()
    resp = client.post("/agent/actions/reopen-return-request", json={"request_id": rid})
    assert resp.status_code == 200
    assert resp.json()["request"]["status"] == "pending"

    # 404 on missing
    resp = client.post("/agent/actions/reopen-return-request", json={"request_id": 99999})
    assert resp.status_code == 404


def test_create_return_request_rejects_invalid_kind(seeded):
    r = create_return_request("JD20260420456", "banana_split", "nope")
    assert r["ok"] is False
    assert r["reason"] == "invalid_kind"


def test_aftersale_node_end_to_end_return(seeded):
    """AfterSale is dry-run: eligibility only, no row in return_requests."""
    state: AgentState = {
        "tenant": "jd",
        "user_id": "jd-demo-user",
        "messages": [{"role": "user", "content": "我要退货"}],
        "plan": [
            {
                "step_id": 1,
                "agent": "aftersale",
                "query": "我要把 iPhone 订单退货",
                "depends_on": [],
                "status": "pending",
            }
        ],
        "current_step": 0,
        "step_results": {},
        "entities": {},
        "trace": [],
    }
    patch = aftersale_node(state)
    res = patch["step_results"][1]
    assert res["agent"] == "aftersale"
    assert res["order_id"] == "JD20260420456"
    assert res["kind"] == "return"
    assert res["eligibility"]["ok"] is True
    assert res["request"] is None
    assert list_requests_for_order("JD20260420456") == []
    # entities carried forward, but no last_return_request_id because nothing was created
    assert patch["entities"]["last_order_id"] == "JD20260420456"
    assert patch["entities"]["last_aftersale_kind"] == "return"
    assert "last_return_request_id" not in patch["entities"]


def test_aftersale_node_surfaces_existing_pending_request(seeded):
    """If the UI button already created a pending R-N, re-asking surfaces it
    instead of showing the confirm button again."""
    # Simulate the button having run earlier in the session
    created = create_return_request("JD20260420456", "return", "之前点了按钮")
    assert created["ok"] is True
    rid = created["request_id"]

    state: AgentState = {
        "tenant": "jd",
        "user_id": "jd-demo-user",
        "messages": [{"role": "user", "content": "刚才那个退货申请怎么样了"}],
        "plan": [
            {"step_id": 1, "agent": "aftersale", "query": "iPhone 那单退货情况", "depends_on": [], "status": "pending"},
        ],
        "current_step": 0,
        "step_results": {},
        "entities": {},
        "trace": [],
    }
    patch = aftersale_node(state)
    res = patch["step_results"][1]
    assert res["request"] is not None
    assert res["request"]["request_id"] == rid
    assert res["request"]["status"] == "pending"
    # Still only one row -- we did NOT double-create
    assert len(list_requests_for_order("JD20260420456")) == 1
    assert patch["entities"]["last_return_request_id"] == rid


def test_aftersale_node_uses_entities_from_prior_step(seeded):
    """If OrderAgent already stashed last_order_id, AfterSale honours it."""
    state: AgentState = {
        "tenant": "jd",
        "user_id": "jd-demo-user",
        "messages": [{"role": "user", "content": "退款"}],
        "plan": [
            {"step_id": 1, "agent": "aftersale", "query": "退款吧", "depends_on": [], "status": "pending"},
        ],
        "current_step": 0,
        "step_results": {},
        "entities": {"last_order_id": "JD20260418123"},  # seeded from a prior turn
        "trace": [],
    }
    patch = aftersale_node(state)
    res = patch["step_results"][1]
    assert res["order_id"] == "JD20260418123"
    # "退款吧" → kind == "refund" via keyword matcher
    assert res["kind"] == "refund"


def test_aftersale_abstain_when_no_order(seeded):
    """Need to ensure no orders for this user so aftersale truly abstains."""
    # ghost user id won't match any seeded orders
    state: AgentState = {
        "tenant": "jd",
        "user_id": "ghost-user-no-orders",
        "messages": [{"role": "user", "content": "退货"}],
        "plan": [
            {"step_id": 1, "agent": "aftersale", "query": "退货", "depends_on": [], "status": "pending"},
        ],
        "current_step": 0,
        "step_results": {},
        "entities": {},
        "trace": [],
    }
    patch = aftersale_node(state)
    res = patch["step_results"][1]
    assert res["abstain"] is True
