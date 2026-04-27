"""4th-layer Agent memory: long-term user preferences across sessions.

Covers:
- set / get / list / delete helpers
- key normalisation (lowercase, trimmed, ≤64 chars)
- value rejection on empty
- upsert: same (user, key) updates instead of duplicating
- isolation per user
- ``render_for_planner`` produces a compact one-line-per-pref summary
  (Planner injects this into its system prompt as the 4th memory layer)
- API endpoints round-trip
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent.tools.preferences import (
    delete_preference,
    get_preference,
    list_preferences,
    render_for_planner,
    set_preference,
)
from rag.db.base import Base, SessionLocal, engine
from rag.db.models import User, UserPreference
from rag_api.main import app


@pytest.fixture(autouse=True)
def _fresh_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    # Seed the user rows we'll write preferences against (FK CASCADE).
    with SessionLocal() as s:
        s.add(User(id="alice", tenant="jd", display_name="Alice"))
        s.add(User(id="bob", tenant="taobao", display_name="Bob"))
        s.commit()
    yield


@pytest.fixture()
def client():
    return TestClient(app)


def test_set_and_get_preference():
    p = set_preference("alice", "preferred_carrier", "jd-express")
    assert p["key"] == "preferred_carrier"
    assert p["value"] == "jd-express"
    assert p["source"] == "user"

    fetched = get_preference("alice", "preferred_carrier")
    assert fetched["value"] == "jd-express"


def test_key_is_normalised_lowercase_trimmed():
    set_preference("alice", "  Preferred_Carrier  ", "jd-express")
    fetched = get_preference("alice", "preferred_carrier")
    assert fetched is not None


def test_set_preference_rejects_empty():
    with pytest.raises(ValueError):
        set_preference("alice", "", "jd")
    with pytest.raises(ValueError):
        set_preference("alice", "carrier", "")


def test_upsert_updates_in_place():
    set_preference("alice", "carrier", "jd-express")
    set_preference("alice", "carrier", "sf-express")
    items = list_preferences("alice")
    assert len(items) == 1
    assert items[0]["value"] == "sf-express"


def test_isolation_per_user():
    set_preference("alice", "carrier", "jd-express")
    set_preference("bob", "carrier", "sf-express")
    assert get_preference("alice", "carrier")["value"] == "jd-express"
    assert get_preference("bob", "carrier")["value"] == "sf-express"


def test_delete_preference():
    set_preference("alice", "carrier", "jd-express")
    assert delete_preference("alice", "carrier") is True
    assert get_preference("alice", "carrier") is None
    assert delete_preference("alice", "carrier") is False  # idempotent


def test_render_for_planner_empty_when_no_prefs():
    assert render_for_planner("alice") == ""


def test_render_for_planner_compact_summary():
    set_preference("alice", "preferred_carrier", "jd-express")
    set_preference("alice", "lang_preference", "zh-CN", source="inferred")
    summary = render_for_planner("alice")
    # Surfaces both preferences with source attribution.
    assert "preferred_carrier" in summary
    assert "jd-express" in summary
    assert "lang_preference" in summary
    assert "zh-CN" in summary
    assert "user" in summary or "inferred" in summary


def test_render_for_planner_caps_at_max_items():
    for i in range(10):
        set_preference("alice", f"k{i}", f"v{i}")
    summary = render_for_planner("alice", max_items=3)
    # Mentions the overflow count.
    assert "另有" in summary or "..." in summary


# ---------- API endpoints ----------

def test_endpoint_list_preferences(client):
    set_preference("alice", "carrier", "jd-express")
    r = client.get("/users/alice/preferences")
    assert r.status_code == 200
    data = r.json()
    assert data["user_id"] == "alice"
    assert any(p["key"] == "carrier" for p in data["items"])


def test_endpoint_set_preference(client):
    r = client.put("/users/alice/preferences",
                   json={"key": "lang", "value": "en"})
    assert r.status_code == 200
    assert r.json()["preference"]["value"] == "en"


def test_endpoint_set_preference_rejects_bad_source(client):
    r = client.put("/users/alice/preferences",
                   json={"key": "lang", "value": "en", "source": "evil"})
    assert r.status_code == 400


def test_endpoint_set_preference_rejects_empty_key(client):
    r = client.put("/users/alice/preferences",
                   json={"key": "", "value": "x"})
    assert r.status_code == 400


def test_endpoint_delete_preference(client):
    set_preference("alice", "carrier", "jd-express")
    r = client.delete("/users/alice/preferences/carrier")
    assert r.status_code == 200
    r2 = client.delete("/users/alice/preferences/carrier")
    assert r2.status_code == 404
