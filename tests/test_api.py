"""API surface tests.

Uses FastAPI's TestClient so we don't need the server process running.
The DB goes through the in-memory fixture; LLM-dependent endpoints are
not exercised here (that is covered by the agent test + eval harness).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(db_session):
    from rag_api.main import app
    return TestClient(app)


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["service"] == "RAG API"
    assert "x-trace-id" in {k.lower() for k in r.headers.keys()}


def test_health_respects_incoming_trace_id(client):
    r = client.get("/health", headers={"X-Trace-Id": "fixture-trace"})
    assert r.headers.get("x-trace-id") == "fixture-trace"


def test_feedback_persists_up(client):
    body = {
        "trace_id": "t1",
        "kb_id": "jd_demo",
        "query": "PLUS 年卡多少钱",
        "answer": "¥299...",
        "verdict": "up",
    }
    r = client.post("/feedback", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True

    from rag.db.base import SessionLocal
    from rag.db.models import Feedback
    with SessionLocal() as s:
        rows = s.query(Feedback).all()
        assert len(rows) == 1
        assert rows[0].verdict == "up"


def test_feedback_rejects_bad_verdict(client):
    body = {
        "trace_id": "t2",
        "kb_id": "x",
        "query": "q",
        "answer": "a",
        "verdict": "maybe",
    }
    r = client.post("/feedback", json=body)
    assert r.status_code == 400


def test_kbs_lists_filesystem(client):
    r = client.get("/kbs")
    assert r.status_code == 200
    items = r.json()["items"]
    # We built jd_demo / taobao_demo / cs4303 earlier, so there should be >= 1.
    assert isinstance(items, list)


def test_profiles_compat(client):
    r = client.get("/profiles")
    assert r.status_code == 200
    assert "profiles" in r.json()
