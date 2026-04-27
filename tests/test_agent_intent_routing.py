"""Agent-path intent routing tests.

The RAG path already has 3-way intent routing (see test_intent_routing.py).
This file mirrors that for the Agent path: meta-questions ("我之前问的是
哪个订单") must NOT reach the LangGraph planner — they should be answered
directly from conversation history.

Without this fix, the planner routes by keyword ("订单") and the order
specialist's default "list recent orders" behavior fires, producing the
hallucinated "您之前咨询的是这些订单" response.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    current = None
    for line in body.splitlines():
        if not line:
            current = None
            continue
        if line.startswith("event:"):
            current = line[len("event:"):].strip()
        elif line.startswith("data:"):
            if current is None:
                continue
            events.append((current, json.loads(line[len("data:"):].strip())))
    return events


@pytest.fixture(autouse=True)
def _reset_db():
    from rag.db.base import Base, engine
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture
def client():
    from rag_api.main import app
    return TestClient(app)


def _install_meta_mocks(monkeypatch, conversation_to_inject):
    """Pretend a checkpoint exists with prior conversation, so has_prior=True
    and the meta path can fire. Bypasses real LangGraph + LLM."""
    from rag_api import agent_routes as ar

    class _FakeSnapshot:
        def __init__(self, values):
            self.values = values

    class _FakeGraph:
        async def aget_state(self, config):
            return _FakeSnapshot({
                "tenant": "jd",
                "user_id": "jd-demo-user",
                "messages": conversation_to_inject,
                "entities": {"last_order_id": "JD20260420456"},
            })

        async def astream(self, *args, **kwargs):
            # Should NOT be called on meta path — assertion enforced in tests.
            raise AssertionError("astream was called on a meta-intent turn")
            yield  # pragma: no cover -- generator marker

    monkeypatch.setattr(ar, "build_graph", lambda **kw: _FakeGraph())

    # Stub touch_session to avoid DB writes
    import agent.tools.sessions as sessions_mod
    monkeypatch.setattr(sessions_mod, "touch_session", lambda **kw: None)


def test_agent_meta_query_skips_graph(client, monkeypatch):
    """The killer scenario: user asks 'what order did I ask about earlier'
    AFTER prior turns about something unrelated. Without intent routing the
    order specialist would list ALL the user's orders (wrong). With it,
    the meta handler answers from conversation history."""
    prior_convo = [
        {"role": "user", "content": "PLUS 会员价格"},
        {"role": "assistant", "content": "PLUS 年费 198 元。"},
    ]
    _install_meta_mocks(monkeypatch, prior_convo)

    # Stub the meta LLM call so test is deterministic
    from rag.answer import meta_answer
    async def fake_answer_meta(query, conversation, language):
        return "你之前没有提过具体订单,只问了 PLUS 会员价格。"
    monkeypatch.setattr(meta_answer, "answer_meta", fake_answer_meta)
    # Patch the import used inside agent_routes too (it imports the module function directly)
    import rag_api.agent_routes as ar
    monkeypatch.setattr(ar, "answer_meta", fake_answer_meta)

    resp = client.post("/agent/chat", json={
        "query": "我之前问的是哪个订单",
        "tenant": "jd",
        "user_id": "jd-demo-user",
        "thread_id": "test-thread-1",
    })
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    types = [e[0] for e in events]

    # Must NOT see plan / specialist_start / specialist_done — graph was bypassed.
    assert "plan" not in types
    assert "specialist_start" not in types
    assert "specialist_done" not in types

    # Must see an answer event with intent=meta + empty citations.
    answer_events = [e for e in events if e[0] == "answer"]
    assert len(answer_events) == 1
    payload = answer_events[0][1]
    assert payload["intent"] == "meta"
    assert payload["citations"] == []
    assert "PLUS" in payload["text"] or "没有" in payload["text"]


def test_agent_chitchat_skips_graph(client, monkeypatch):
    """Greeting should NOT trigger Planner → policy_qa pipeline.

    Without intent routing the planner uses the keyword '客服' to maybe
    classify as complaint or routes to policy_qa — both heavier than needed.
    """
    _install_meta_mocks(monkeypatch, [])  # no prior conversation needed for chitchat

    from rag.answer import meta_answer
    async def fake_answer_chitchat(query, language):
        return "你好,我是企业知识库 + 智能客服助手。"
    monkeypatch.setattr(meta_answer, "answer_chitchat", fake_answer_chitchat)
    import rag_api.agent_routes as ar
    monkeypatch.setattr(ar, "answer_chitchat", fake_answer_chitchat)

    resp = client.post("/agent/chat", json={
        "query": "你好",
        "tenant": "jd",
        "user_id": "jd-demo-user",
        "thread_id": "test-thread-2",
    })
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    types = [e[0] for e in events]
    assert "plan" not in types
    assert "specialist_start" not in types

    answer_events = [e for e in events if e[0] == "answer"]
    assert len(answer_events) == 1
    assert answer_events[0][1]["intent"] == "chitchat"


def test_agent_meta_without_history_short_circuits_with_fallback(client, monkeypatch):
    """Meta keyword + NO prior conversation → still short-circuit (don't
    run the graph), but return the meta handler's polite "no prior turns"
    fallback. This is the bug we hit when a fresh session got
    "我之前问的是哪个订单" routed to the order specialist, which listed
    all orders and the LLM narrated "you previously asked about these"
    (a hallucination).
    """
    from rag_api import agent_routes as ar
    # Snapshot empty → has_prior=False, conversation=[]
    class _FakeGraph:
        async def aget_state(self, config):
            return None

        async def astream(self, *args, **kwargs):
            raise AssertionError("astream must NOT be called on meta path")
            yield  # pragma: no cover

    monkeypatch.setattr(ar, "build_graph", lambda **kw: _FakeGraph())
    import agent.tools.sessions as sessions_mod
    monkeypatch.setattr(sessions_mod, "touch_session", lambda **kw: None)

    # Stub the meta handler to return a known fallback string.
    async def fake_answer_meta(query, conversation, language):
        # Real handler returns "请重新提问" when conversation is empty.
        return "这是新会话,你之前没有问过别的内容。"
    monkeypatch.setattr(ar, "answer_meta", fake_answer_meta)

    resp = client.post("/agent/chat", json={
        "query": "刚刚说的是什么",
        "tenant": "jd",
        "user_id": "jd-demo-user",
        "thread_id": "test-thread-3",
    })
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    types = [e[0] for e in events]
    assert "plan" not in types  # planner did NOT run
    assert "specialist_start" not in types
    answer_events = [e for e in events if e[0] == "answer"]
    assert len(answer_events) == 1
    payload = answer_events[0][1]
    assert payload["intent"] == "meta"
    assert "新会话" in payload["text"] or "之前" in payload["text"]
