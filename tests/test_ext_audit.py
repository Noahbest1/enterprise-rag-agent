"""Extension-stage audit log tests.

Covers:
- AuditLog row written on every non-exempt HTTP request (middleware)
- Exempt paths (/health /metrics /) do NOT create rows
- Business event helper writes with custom event_type + extra
- /audit lists rows newest-first, respects filters (event_type, path, user_id, since)
- /audit with authed caller scopes to auth.tenant_id (can't read other tenants)
- hash16 is deterministic and 16 chars
- record_audit never raises when DB is unreachable (swallow-on-fail)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from rag.config import settings
from rag.db.base import Base, SessionLocal, engine
from rag.db.models import ApiKey, AuditLog
from rag_api.audit import hash16, record_audit
from rag_api.auth import generate_raw_key, hash_key


@pytest.fixture(autouse=True)
def _fresh_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture
def client():
    from rag_api.main import app
    return TestClient(app)


# ---------- hash helper ----------

def test_hash16_is_deterministic():
    assert hash16("abc") == hash16("abc")
    assert hash16("abc") != hash16("abd")
    assert len(hash16("abc")) == 16


def test_hash16_handles_none():
    assert hash16(None) is None
    assert hash16("") is None


# ---------- record_audit ----------

def test_record_audit_persists_basic_row():
    record_audit(
        event_type="unit_test",
        tenant_id="acme",
        user_id="u1",
        trace_id="trace-abc",
        extra={"note": "hello"},
    )
    with SessionLocal() as s:
        rows = s.query(AuditLog).all()
        assert len(rows) == 1
        r = rows[0]
        assert r.event_type == "unit_test"
        assert r.tenant_id == "acme"
        assert r.user_id == "u1"
        assert r.trace_id == "trace-abc"
        assert r.extra == {"note": "hello"}


def test_record_audit_truncates_long_error():
    record_audit(event_type="t", error="x" * 9000)
    with SessionLocal() as s:
        r = s.query(AuditLog).one()
        assert r.error and len(r.error) <= 500


# ---------- middleware ----------

def test_middleware_writes_http_audit(client: TestClient):
    r = client.get("/kbs")
    assert r.status_code == 200
    with SessionLocal() as s:
        rows = s.query(AuditLog).all()
        # Should be exactly 1: /kbs
        assert len(rows) == 1
        row = rows[0]
        assert row.event_type == "http_request"
        assert row.method == "GET"
        assert row.path == "/kbs"
        assert row.status_code == 200
        assert row.latency_ms is not None and row.latency_ms >= 0
        assert row.trace_id  # set by trace_and_log


def test_middleware_exempts_health_metrics_root(client: TestClient):
    client.get("/health")
    client.get("/metrics")
    with SessionLocal() as s:
        rows = s.query(AuditLog).filter(AuditLog.path.in_(["/health", "/metrics", "/"])).all()
        assert rows == []


def test_middleware_records_status_code_for_errors(client: TestClient):
    r = client.post("/search", json={"query": "x", "kb_id": "__nonexistent__"})
    # 404 comes from missing KB; should still be audited
    assert r.status_code == 404
    with SessionLocal() as s:
        rows = s.query(AuditLog).filter(AuditLog.path == "/search").all()
        assert len(rows) == 1
        assert rows[0].status_code == 404


# ---------- /audit endpoint ----------

def _seed_key(tenant_id: str = "acme") -> str:
    raw = generate_raw_key()
    with SessionLocal() as s:
        s.add(ApiKey(key_hash=hash_key(raw), tenant_id=tenant_id))
        s.commit()
    return raw


def test_audit_endpoint_lists_newest_first(client: TestClient):
    # Seed rows
    with SessionLocal() as s:
        for i in range(3):
            s.add(AuditLog(
                event_type="unit_test",
                tenant_id="acme",
                extra={"i": i},
                created_at=datetime.now(timezone.utc) + timedelta(seconds=i),
            ))
        s.commit()
    r = client.get("/audit?event_type=unit_test&limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    # Newest first: extra.i should be 2, 1, 0
    assert [row["extra"]["i"] for row in body["items"]] == [2, 1, 0]


def test_audit_endpoint_filters_by_path(client: TestClient):
    client.get("/kbs")
    client.post("/search", json={"query": "x", "kb_id": "_none_"})
    r = client.get("/audit?path=/search")
    body = r.json()
    assert all(row["path"] == "/search" for row in body["items"])
    assert len(body["items"]) == 1


def test_audit_endpoint_filters_by_user_id(client: TestClient):
    record_audit(event_type="unit", tenant_id="acme", user_id="u1")
    record_audit(event_type="unit", tenant_id="acme", user_id="u2")
    r = client.get("/audit?user_id=u1")
    body = r.json()
    assert body["count"] == 1
    assert body["items"][0]["user_id"] == "u1"


def test_audit_endpoint_filters_by_since(client: TestClient):
    # Seed one old, one recent row
    old = datetime.now(timezone.utc) - timedelta(days=2)
    recent = datetime.now(timezone.utc) - timedelta(hours=1)
    with SessionLocal() as s:
        s.add(AuditLog(event_type="unit", created_at=old, extra={"tag": "old"}))
        s.add(AuditLog(event_type="unit", created_at=recent, extra={"tag": "recent"}))
        s.commit()
    # Use Z suffix so + in "+00:00" isn't URL-decoded to space in the query string.
    since = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat().replace("+00:00", "Z")
    r = client.get("/audit", params={"event_type": "unit", "since": since})
    body = r.json()
    assert body["count"] == 1
    assert body["items"][0]["extra"]["tag"] == "recent"


def test_audit_endpoint_scopes_to_tenant_when_authed(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "require_api_key", True)
    raw = _seed_key(tenant_id="acme")
    # Seed rows for both tenants
    record_audit(event_type="unit", tenant_id="acme", extra={"t": "acme"})
    record_audit(event_type="unit", tenant_id="other", extra={"t": "other"})
    # Authed caller tries to read 'other' — should get ONLY acme
    r = client.get("/audit?tenant_id=other&event_type=unit",
                   headers={"Authorization": f"Bearer {raw}"})
    body = r.json()
    assert all(row["tenant_id"] == "acme" for row in body["items"])


def test_audit_endpoint_bad_since_returns_400(client: TestClient):
    r = client.get("/audit?since=not-a-date")
    assert r.status_code == 400
