"""Recommend specialist."""
from __future__ import annotations

import pytest

from agent.specialists.recommend import recommend_node
from agent.state import AgentState
from agent.tools.recommend import refresh_catalogue, similar_products


@pytest.fixture()
def seeded(seeded_db):
    # Make sure recommend's lru_cache is repopulated against the seeded DB.
    refresh_catalogue()
    yield seeded_db
    refresh_catalogue()


def test_similar_products_returns_candidates(seeded):
    out = similar_products("MacBook Pro", top_k=3)
    assert len(out) > 0
    assert all("sku" in r and "similarity" in r for r in out)


def test_similar_products_exclude_self(seeded):
    # Find the macbook sku first
    all_out = similar_products("MacBook Pro", top_k=10)
    mac_sku = next(r["sku"] for r in all_out if "MacBook" in r["title"])

    filtered = similar_products("MacBook Pro", top_k=10, exclude_sku=mac_sku)
    assert all(r["sku"] != mac_sku for r in filtered)


def test_recommend_node_uses_entities(seeded):
    state: AgentState = {
        "tenant": "jd",
        "user_id": "jd-demo-user",
        "messages": [{"role": "user", "content": "有类似的吗"}],
        "plan": [{"step_id": 1, "agent": "recommend", "query": "类似的商品", "depends_on": [], "status": "pending"}],
        "current_step": 0,
        "step_results": {},
        "entities": {"last_item_title": "Apple MacBook Pro 14 M4 Pro"},
        "trace": [],
    }
    patch = recommend_node(state)
    res = patch["step_results"][1]
    assert res["abstain"] is False
    assert len(res["items"]) > 0
    assert res["anchor"] == "Apple MacBook Pro 14 M4 Pro"


def test_recommend_node_falls_back_to_query(seeded):
    state: AgentState = {
        "tenant": "jd",
        "user_id": "jd-demo-user",
        "messages": [{"role": "user", "content": "手机有什么好的"}],
        "plan": [{"step_id": 1, "agent": "recommend", "query": "推荐个手机", "depends_on": [], "status": "pending"}],
        "current_step": 0,
        "step_results": {},
        "entities": {},
        "trace": [],
    }
    patch = recommend_node(state)
    res = patch["step_results"][1]
    assert res["abstain"] is False
    assert len(res["items"]) > 0
