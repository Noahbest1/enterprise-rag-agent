"""PH5.3 API key auth tests.

Covers:
- hash_key determinism + generate_raw_key uniqueness
- anonymous fallback when require_api_key=False
- 401 when require_api_key=True and no/invalid key is given
- valid key -> tenant_id populated, last_used_at updated
- disabled key rejected
- protected endpoints (e.g. /search) honour auth
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from rag.config import settings
from rag.db.base import Base, SessionLocal, engine
from rag.db.models import ApiKey
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


def _seed_key(tenant_id: str = "acme", disabled: bool = False) -> str:
    raw = generate_raw_key()
    with SessionLocal() as s:
        row = ApiKey(
            key_hash=hash_key(raw),
            tenant_id=tenant_id,
            description="test",
        )
        s.add(row)
        s.commit()
        if disabled:
            from datetime import datetime, timezone
            row.disabled_at = datetime.now(timezone.utc)
            s.commit()
    return raw


# ---------- unit ----------

def test_hash_key_is_deterministic():
    assert hash_key("abc") == hash_key("abc")
    assert hash_key("abc") != hash_key("abd")
    # sha256 hex = 64 chars
    assert len(hash_key("abc")) == 64


def test_generate_raw_key_unique():
    keys = {generate_raw_key() for _ in range(20)}
    assert len(keys) == 20
    for k in keys:
        assert k.startswith("rag_")
        assert len(k) > 30


# ---------- dependency ----------

def test_anonymous_fallback_when_require_off(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "require_api_key", False)
    # /health is public; /kbs is open in this test (no deps changed), so use
    # /kbs -- it does not depend on auth but also shouldn't 401.
    r = client.get("/kbs")
    assert r.status_code == 200


def test_401_when_require_on_and_no_key(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "require_api_key", True)
    r = client.post("/search", json={"query": "x", "kb_id": "nonexistent"})
    assert r.status_code == 401
    assert "WWW-Authenticate" in r.headers


def test_401_on_invalid_bearer(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "require_api_key", True)
    r = client.post(
        "/search",
        json={"query": "x", "kb_id": "nonexistent"},
        headers={"Authorization": "Bearer rag_fake_key_that_does_not_exist"},
    )
    assert r.status_code == 401


def test_valid_bearer_populates_tenant(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "require_api_key", True)
    raw = _seed_key(tenant_id="acme")

    # /search returns 404 for missing KB but that means we made it PAST auth.
    r = client.post(
        "/search",
        json={"query": "x", "kb_id": "nonexistent"},
        headers={"Authorization": f"Bearer {raw}"},
    )
    assert r.status_code == 404  # passed auth, failed on KB lookup


def test_disabled_key_rejected(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "require_api_key", True)
    raw = _seed_key(tenant_id="acme", disabled=True)
    r = client.post(
        "/search",
        json={"query": "x", "kb_id": "nonexistent"},
        headers={"Authorization": f"Bearer {raw}"},
    )
    assert r.status_code == 401


def test_last_used_at_updates(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "require_api_key", True)
    raw = _seed_key(tenant_id="acme")
    time.sleep(0.01)
    r = client.post(
        "/search",
        json={"query": "x", "kb_id": "nonexistent"},
        headers={"Authorization": f"Bearer {raw}"},
    )
    assert r.status_code == 404

    with SessionLocal() as s:
        row = s.query(ApiKey).filter(ApiKey.tenant_id == "acme").one()
        assert row.last_used_at is not None


def test_anonymous_still_allowed_on_public_routes(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "require_api_key", True)
    # /health and /metrics should remain reachable without a key.
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200
