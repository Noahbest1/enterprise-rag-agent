"""Sprint A.4 /answer/stream SSE endpoint tests.

Covers:
- Happy path: meta / hits / delta / done event sequence emitted
- Abstain path: abstain + done (no delta)
- Injection-blocked path: abstain + done with reason=injection_blocked
- Bad KB: error event
- SSE formatting: "event:"/"data:" prefixes, JSON body parseable

Every test mocks ``answer_query_stream`` at the import site in rag_api.main
so no real LLM call happens.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from rag.db.base import Base, engine


@pytest.fixture(autouse=True)
def _db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture
def client():
    from rag_api.main import app
    return TestClient(app)


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Return (event_name, json_payload) pairs from an SSE body."""
    events: list[tuple[str, dict]] = []
    current_event = None
    for line in body.splitlines():
        if not line:
            current_event = None
            continue
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            if current_event is None:
                continue
            payload = json.loads(line[len("data:"):].strip())
            events.append((current_event, payload))
    return events


async def _fake_happy_stream(query, kb_id, **kwargs):
    yield "meta", {"rewritten_query": query, "language": "zh"}
    yield "hits", {"citations_preview": [{"n": 1, "title": "t", "source_id": "s", "score": 0.9}]}
    for piece in ["你", "好", ",", "这是", "答案"]:
        yield "delta", {"text": piece}
    yield "done", {"abstained": False, "citations": [], "latency_ms": 42}


async def _fake_abstain_stream(query, kb_id, **kwargs):
    yield "meta", {"rewritten_query": query, "language": "zh"}
    yield "hits", {"citations_preview": []}
    yield "abstain", {"text": "no evidence", "reason": "low_score"}
    yield "done", {"abstained": True, "citations": [], "latency_ms": 11}


async def _fake_injection_stream(query, kb_id, **kwargs):
    yield "abstain", {"text": "blocked", "reason": "injection_blocked"}
    yield "done", {"abstained": True, "citations": [], "latency_ms": 1}


async def _fake_error_stream(query, kb_id, **kwargs):
    yield "error", {"detail": "KB 'missing' has no built indexes."}


# ---------- tests ----------

def test_stream_happy_path(client: TestClient, monkeypatch):
    import rag_api.main
    monkeypatch.setattr(rag_api.main, "answer_query_stream", _fake_happy_stream)

    r = client.post("/answer/stream", json={"query": "你好", "kb_id": "jd_demo"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(r.text)
    names = [e[0] for e in events]
    # Order: meta, hits, delta*, done
    assert names[0] == "meta"
    assert names[1] == "hits"
    assert names[-1] == "done"
    deltas = [e for e in events if e[0] == "delta"]
    assert len(deltas) == 5
    # Reassembled text
    assert "".join(d[1]["text"] for d in deltas) == "你好,这是答案"


def test_stream_abstain_path(client: TestClient, monkeypatch):
    import rag_api.main
    monkeypatch.setattr(rag_api.main, "answer_query_stream", _fake_abstain_stream)

    r = client.post("/answer/stream", json={"query": "something", "kb_id": "jd_demo"})
    assert r.status_code == 200
    events = _parse_sse(r.text)
    names = [e[0] for e in events]
    assert "delta" not in names
    assert ("abstain", {"text": "no evidence", "reason": "low_score"}) in events
    assert events[-1][0] == "done"
    assert events[-1][1]["abstained"] is True


def test_stream_injection_path(client: TestClient, monkeypatch):
    import rag_api.main
    monkeypatch.setattr(rag_api.main, "answer_query_stream", _fake_injection_stream)

    r = client.post("/answer/stream", json={"query": "ignore all previous", "kb_id": "jd_demo"})
    assert r.status_code == 200
    events = _parse_sse(r.text)
    assert events[0] == ("abstain", {"text": "blocked", "reason": "injection_blocked"})
    assert events[-1][0] == "done"


def test_stream_error_event(client: TestClient, monkeypatch):
    import rag_api.main
    monkeypatch.setattr(rag_api.main, "answer_query_stream", _fake_error_stream)

    r = client.post("/answer/stream", json={"query": "x", "kb_id": "missing"})
    assert r.status_code == 200
    events = _parse_sse(r.text)
    assert events == [("error", {"detail": "KB 'missing' has no built indexes."})]


def test_stream_sse_formatting(client: TestClient, monkeypatch):
    import rag_api.main
    monkeypatch.setattr(rag_api.main, "answer_query_stream", _fake_happy_stream)

    r = client.post("/answer/stream", json={"query": "x", "kb_id": "jd_demo"})
    # Every event block is `event: X\ndata: {...}\n\n`
    blocks = [b for b in r.text.split("\n\n") if b.strip()]
    for b in blocks:
        lines = b.splitlines()
        assert lines[0].startswith("event:")
        assert lines[1].startswith("data:")
        # data payload is valid JSON
        json.loads(lines[1][len("data:"):].strip())
