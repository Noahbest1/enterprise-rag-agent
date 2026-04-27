"""End-to-end test of the agent's Order + Logistics specialists against real DB.

Bypasses the Planner LLM by constructing the plan by hand, then invokes each
specialist node in sequence. This lets us assert deterministic behaviour
without spending LLM tokens.
"""
from __future__ import annotations

import pytest

from agent.specialists.logistics import logistics_node
from agent.specialists.order import order_node
from agent.state import AgentState


@pytest.fixture()
def base_state(seeded_db) -> AgentState:
    return {
        "tenant": "jd",
        "user_id": "jd-demo-user",
        "messages": [{"role": "user", "content": "我 MacBook 订单物流"}],
        "plan": [
            {
                "step_id": 1,
                "agent": "order",
                "query": "查找最近的 MacBook 订单",
                "depends_on": [],
                "status": "pending",
            },
            {
                "step_id": 2,
                "agent": "logistics",
                "query": "根据上一步订单查询物流",
                "depends_on": [1],
                "status": "pending",
            },
        ],
        "current_step": 0,
        "step_results": {},
        "entities": {},
        "trace": [],
    }


def _apply(state: AgentState, patch: dict) -> AgentState:
    merged = {**state}
    for k, v in patch.items():
        if k == "trace":
            merged["trace"] = [*(state.get("trace") or []), *v]
        elif k == "step_results":
            merged["step_results"] = v  # already includes prior results
        elif k == "entities":
            merged["entities"] = {**(state.get("entities") or {}), **v}
        else:
            merged[k] = v
    return merged


def test_order_node_finds_macbook(base_state):
    state = base_state
    patch = order_node(state)
    assert 1 in patch["step_results"]
    res = patch["step_results"][1]
    assert res["abstain"] is False
    assert res["mode"] in ("by_keyword", "by_id", "recent")
    assert res["orders"], "orders should not be empty"
    top = res["orders"][0]
    assert "MacBook" in top["items"][0]["title"]
    # Entity carried forward
    assert patch["entities"]["last_order_id"] == top["id"]
    assert patch["entities"]["last_tracking_no"] == top["tracking_no"]


def test_logistics_uses_entities_from_order(base_state):
    # Simulate: order step already ran, entities carry the tracking_no.
    state = _apply(
        base_state,
        order_node(base_state),
    )
    state["current_step"] = 1

    patch = logistics_node(state)
    assert 2 in patch["step_results"]
    res = patch["step_results"][2]
    assert res["abstain"] is False
    assert "tracking_info" in res
    info = res["tracking_info"]
    assert info["status"] in ("placed", "shipped", "delivered", "refunded")
    assert info["timeline"], "timeline should have events"


def test_order_node_returns_recent_when_no_keyword(base_state):
    base_state["plan"][0]["query"] = "最近的订单"
    patch = order_node(base_state)
    res = patch["step_results"][1]
    assert res["mode"] == "recent"
    assert len(res["orders"]) >= 1


def test_order_node_abstains_for_unknown_tenant():
    state: AgentState = {
        "tenant": "jd",
        "user_id": "ghost-user-no-data",
        "messages": [{"role": "user", "content": "查订单"}],
        "plan": [
            {"step_id": 1, "agent": "order", "query": "查订单", "depends_on": [], "status": "pending"},
        ],
        "current_step": 0,
        "step_results": {},
        "entities": {},
        "trace": [],
    }
    patch = order_node(state)
    res = patch["step_results"][1]
    assert res["abstain"] is True
    assert res["orders"] == []


def test_logistics_abstains_without_tracking_info(base_state):
    base_state["plan"][0]["agent"] = "logistics"
    base_state["plan"][0]["query"] = "没有单号"
    base_state["plan"] = [base_state["plan"][0]]
    patch = logistics_node(base_state)
    res = patch["step_results"][1]
    assert res["abstain"] is True
