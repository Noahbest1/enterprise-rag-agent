"""SSE agent streaming endpoint: event shape + ordering."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(seeded_db):
    from rag_api.main import app
    return TestClient(app)


def _parse_sse(text: str) -> list[dict]:
    """Tiny SSE parser: return [{"event":..., "data": raw_json_str}]."""
    events = []
    cur: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("event: "):
            cur["event"] = line[len("event: "):]
        elif line.startswith("data: "):
            cur.setdefault("data", "")
            cur["data"] += line[len("data: "):]
        elif line == "":
            if cur:
                events.append(cur)
                cur = {}
    if cur:
        events.append(cur)
    return events


def test_agent_chat_streams_expected_events(client):
    """The SSE stream emits agent_start, plan, specialist_start/done, error OR answer, done."""
    # We hit a deterministic order flow so no LLM quirks. An Order-only query
    # still needs the Planner LLM for decomposition -- skip if no key.
    import os
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.skip("LLM required for planner in SSE test")

    body = {"tenant": "jd", "query": "查我最近的订单", "user_id": "jd-demo-user"}
    with client.stream("POST", "/agent/chat", json=body) as r:
        assert r.status_code == 200, r.read()
        raw = b"".join(list(r.iter_bytes())).decode("utf-8")

    events = _parse_sse(raw)
    names = [e.get("event") for e in events]
    assert names[0] == "agent_start"
    assert names[-1] in ("done", "error")
    assert "plan" in names
    # either answer or error must be present before done
    assert any(n in names for n in ("answer", "error"))


def test_agent_chat_rejects_empty_query(client):
    r = client.post("/agent/chat", json={"query": "", "tenant": "jd"})
    assert r.status_code == 400


def test_agent_chat_rejects_bad_tenant(client):
    # The endpoint accepts but streams an error event
    body = {"tenant": "nope", "query": "hi"}
    with client.stream("POST", "/agent/chat", json=body) as r:
        raw = b"".join(list(r.iter_bytes())).decode("utf-8")
    events = _parse_sse(raw)
    assert any(e.get("event") == "error" for e in events)


def test_get_variant_works(client):
    import os
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.skip("LLM required")
    r = client.get("/agent/chat", params={"query": "查订单", "tenant": "jd"})
    assert r.status_code == 200
