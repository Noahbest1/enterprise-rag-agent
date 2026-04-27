"""Model A: ChatSession metadata + thread_id-scoped tickets.

Covers:
- touch_session creates on first call, bumps last_msg_at on subsequent
- title auto-populated from first message, capped at 40 chars
- title preserved on later touches (rename is explicit only)
- list_sessions_for_user returns newest-first, scoped per user
- rename / delete helpers
- GET /users/{id}/sessions endpoint
- PATCH /sessions/{id} title rename
- DELETE /sessions/{id} clears row + NULLs complaints.thread_id (tickets
  are user property and survive session deletion)
- POST /agent/actions/submit-complaint stores thread_id when supplied
- GET /users/{id}/inbox?thread_id=... filters to that session + NULL
  thread_id complaints (legacy / unassigned)
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from agent.tools.complaints import create_complaint
from agent.tools.sessions import (
    delete_session,
    get_session,
    list_sessions_for_user,
    rename_session,
    touch_session,
)
from rag.db.base import Base, SessionLocal, engine
from rag.db.models import ChatSession, Complaint
from rag_api.main import app


@pytest.fixture(autouse=True)
def _fresh_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture()
def client():
    return TestClient(app)


# ---------- touch_session ----------

def test_touch_session_creates_on_first_call():
    s = touch_session(thread_id="t-1", user_id="alice", tenant="jd",
                      first_message="查我最近的 iPhone 订单")
    assert s["thread_id"] == "t-1"
    assert s["user_id"] == "alice"
    assert s["tenant"] == "jd"
    assert s["title"] == "查我最近的 iPhone 订单"
    assert s["first_msg_at"] is not None
    assert s["last_msg_at"] is not None


def test_touch_session_truncates_long_title_to_40_chars():
    long_msg = "这是一个非常非常非常非常非常非常非常非常非常非常非常非常非常长的查询语句"
    s = touch_session(thread_id="t-2", user_id="bob", tenant="taobao",
                      first_message=long_msg)
    assert len(s["title"]) <= 40
    assert s["title"] == long_msg[:40]


def test_touch_session_falls_back_to_default_title_when_empty():
    s = touch_session(thread_id="t-3", user_id="carol", tenant="jd",
                      first_message="")
    assert s["title"] == "(新会话)"


def test_touch_session_preserves_title_on_subsequent_calls():
    """The first message becomes the title. Subsequent messages should NOT
    overwrite it (the user might rename later)."""
    touch_session(thread_id="t-4", user_id="alice", tenant="jd",
                  first_message="查订单")
    s2 = touch_session(thread_id="t-4", user_id="alice", tenant="jd",
                       first_message="不同的第二条消息")
    assert s2["title"] == "查订单"  # not the second message


def test_touch_session_bumps_last_msg_at():
    s1 = touch_session(thread_id="t-5", user_id="alice", tenant="jd",
                       first_message="hi")
    time.sleep(1.1)  # SQLite CURRENT_TIMESTAMP has 1s precision
    s2 = touch_session(thread_id="t-5", user_id="alice", tenant="jd")
    t1 = datetime.fromisoformat(s1["last_msg_at"])
    t2 = datetime.fromisoformat(s2["last_msg_at"])
    assert t2 > t1


# ---------- list_sessions_for_user ----------

def test_list_sessions_newest_first():
    touch_session(thread_id="t-old", user_id="alice", tenant="jd",
                  first_message="old")
    time.sleep(1.1)
    touch_session(thread_id="t-new", user_id="alice", tenant="jd",
                  first_message="new")
    items = list_sessions_for_user("alice")
    assert [it["thread_id"] for it in items] == ["t-new", "t-old"]


def test_list_sessions_isolates_per_user():
    touch_session(thread_id="t-a", user_id="alice", tenant="jd",
                  first_message="a")
    touch_session(thread_id="t-b", user_id="bob", tenant="jd",
                  first_message="b")
    a = list_sessions_for_user("alice")
    b = list_sessions_for_user("bob")
    assert {it["thread_id"] for it in a} == {"t-a"}
    assert {it["thread_id"] for it in b} == {"t-b"}


# ---------- rename / delete ----------

def test_rename_session():
    touch_session(thread_id="t-r", user_id="alice", tenant="jd",
                  first_message="default")
    s = rename_session("t-r", title="My iPhone return ticket")
    assert s["title"] == "My iPhone return ticket"


def test_rename_session_rejects_empty_title():
    touch_session(thread_id="t-r2", user_id="alice", tenant="jd",
                  first_message="x")
    assert rename_session("t-r2", title="   ") is None


def test_rename_missing_session_returns_none():
    assert rename_session("nonexistent", title="anything") is None


def test_delete_session():
    touch_session(thread_id="t-d", user_id="alice", tenant="jd",
                  first_message="bye")
    assert delete_session("t-d") is True
    assert get_session("t-d") is None
    assert delete_session("t-d") is False  # idempotent on missing


# ---------- GET /users/{id}/sessions ----------

def test_endpoint_lists_sessions(client):
    touch_session(thread_id="t-e1", user_id="alice", tenant="jd",
                  first_message="first")
    time.sleep(1.1)
    touch_session(thread_id="t-e2", user_id="alice", tenant="jd",
                  first_message="second")
    r = client.get("/users/alice/sessions")
    assert r.status_code == 200
    data = r.json()
    assert data["user_id"] == "alice"
    assert [it["thread_id"] for it in data["items"]] == ["t-e2", "t-e1"]


def test_endpoint_empty_when_user_has_no_sessions(client):
    r = client.get("/users/no-such-user/sessions")
    assert r.status_code == 200
    assert r.json()["items"] == []


# ---------- PATCH /sessions/{id} ----------

def test_patch_rename_endpoint(client):
    touch_session(thread_id="t-p", user_id="alice", tenant="jd",
                  first_message="default")
    r = client.patch("/sessions/t-p", json={"title": "renamed by user"})
    assert r.status_code == 200
    assert r.json()["session"]["title"] == "renamed by user"


def test_patch_rename_404_on_missing(client):
    r = client.patch("/sessions/nope", json={"title": "x"})
    assert r.status_code == 404


# ---------- DELETE /sessions/{id} ----------

def test_delete_session_unlinks_complaints(client):
    """Tickets created in a session should survive session deletion;
    their thread_id just becomes NULL."""
    touch_session(thread_id="t-del", user_id="alice", tenant="jd",
                  first_message="going to file a complaint")
    c = create_complaint(user_id="alice", tenant="jd", order_id=None,
                         topic="other", severity="medium",
                         content="some complaint")
    # Manually link to the session (simulating the submit-complaint flow).
    with SessionLocal() as s:
        row = s.get(Complaint, c["id"])
        row.thread_id = "t-del"
        s.commit()

    r = client.delete("/sessions/t-del")
    assert r.status_code == 200

    # Session row gone.
    assert get_session("t-del") is None
    # Complaint row still exists.
    with SessionLocal() as s:
        row = s.get(Complaint, c["id"])
        assert row is not None
        # But thread_id is NULL now.
        assert row.thread_id is None


def test_delete_session_404_on_missing(client):
    r = client.delete("/sessions/never-existed")
    assert r.status_code == 404


# ---------- submit-complaint stores thread_id ----------

def test_submit_complaint_stores_thread_id(client):
    touch_session(thread_id="t-sub", user_id="alice", tenant="jd",
                  first_message="prep")
    r = client.post("/agent/actions/submit-complaint", json={
        "severity": "medium", "topic": "delivery",
        "user_id": "alice", "tenant": "jd",
        "order_id": None,
        "content": "物流太慢",
        "thread_id": "t-sub",
    })
    assert r.status_code == 200
    cid = r.json()["complaint"]["id"]
    with SessionLocal() as s:
        row = s.get(Complaint, cid)
        assert row.thread_id == "t-sub"


def test_submit_complaint_thread_id_is_optional(client):
    """Legacy callers without thread_id still work; the row's column stays NULL."""
    r = client.post("/agent/actions/submit-complaint", json={
        "severity": "medium", "topic": "other",
        "user_id": "alice", "tenant": "jd",
        "content": "no session involved",
    })
    assert r.status_code == 200
    cid = r.json()["complaint"]["id"]
    with SessionLocal() as s:
        row = s.get(Complaint, cid)
        assert row.thread_id is None


# ---------- inbox?thread_id= filter ----------

def test_complaint_dict_includes_thread_id():
    """``_complaint_to_dict`` must surface ``thread_id`` so the admin reply
    endpoint can include it in the SSE event payload (frontend uses it to
    decide whether to render the reply in the current session)."""
    from agent.tools.complaints import get_complaint
    c = create_complaint(user_id="alice", tenant="jd", order_id=None,
                         topic="other", severity="low", content="x")
    with SessionLocal() as s:
        s.get(Complaint, c["id"]).thread_id = "t-test"
        s.commit()
    fetched = get_complaint(c["id"])
    assert fetched["thread_id"] == "t-test"


def test_admin_reply_sse_payload_includes_thread_id(client, monkeypatch):
    """Regression for the cross-session leak bug: admin replies on a
    complaint must broadcast the parent's thread_id, so the user's
    frontend doesn't dump the reply into whichever session is active."""
    captured: list[dict] = []
    from rag_api import user_events

    def spy_publish(user_id, event_type, data):
        captured.append({"user_id": user_id, "event": event_type, "data": data})
        return 1
    monkeypatch.setattr(user_events, "publish_user_event", spy_publish)
    monkeypatch.setattr("rag_api.admin_routes.publish_user_event", spy_publish)

    # Set up: complaint linked to session t-A
    c = create_complaint(user_id="alice", tenant="jd", order_id=None,
                         topic="delivery", severity="high", content="hi")
    with SessionLocal() as s:
        s.get(Complaint, c["id"]).thread_id = "t-A"
        s.commit()

    r = client.post(f"/admin/complaints/{c['id']}/reply",
                    json={"author_label": "admin-X", "content": "回复"})
    assert r.status_code == 200

    assert len(captured) == 1
    payload = captured[0]["data"]
    assert payload["complaint_id"] == c["id"]
    assert payload["thread_id"] == "t-A"  # the bug fix


def test_admin_reply_sse_payload_thread_id_is_null_for_orphaned_complaint(client, monkeypatch):
    """When the parent session was deleted, the complaint's thread_id is
    NULL — the SSE event should propagate that NULL so the frontend can
    show a 'session deleted, see admin' toast instead of dumping into the
    current session."""
    captured: list[dict] = []
    from rag_api import user_events

    def spy_publish(user_id, event_type, data):
        captured.append({"event": event_type, "data": data})
        return 1
    monkeypatch.setattr(user_events, "publish_user_event", spy_publish)
    monkeypatch.setattr("rag_api.admin_routes.publish_user_event", spy_publish)

    # Complaint with NULL thread_id (orphaned)
    c = create_complaint(user_id="alice", tenant="jd", order_id=None,
                         topic="other", severity="low", content="legacy")
    # thread_id stays NULL by default

    r = client.post(f"/admin/complaints/{c['id']}/reply",
                    json={"author_label": "admin-X", "content": "回复"})
    assert r.status_code == 200
    assert captured[0]["data"]["thread_id"] is None


def test_inbox_filter_by_thread_id_scopes_strictly(client):
    """Inbox with ?thread_id= returns ONLY complaints from that session.
    NULL-thread_id legacy complaints don't leak into every session anymore
    (the old behaviour confused users — '本会话工单' shouldn't include
    cross-session orphans). Legacy complaints are still visible via the
    unfiltered inbox call and via admin."""
    # complaint A: in session t-1
    cA = create_complaint(user_id="alice", tenant="jd", order_id=None,
                          topic="delivery", severity="medium", content="A")
    with SessionLocal() as s:
        s.get(Complaint, cA["id"]).thread_id = "t-1"
        s.commit()
    # complaint B: in session t-2
    cB = create_complaint(user_id="alice", tenant="jd", order_id=None,
                          topic="quality", severity="medium", content="B")
    with SessionLocal() as s:
        s.get(Complaint, cB["id"]).thread_id = "t-2"
        s.commit()
    # complaint C: legacy, no session
    cC = create_complaint(user_id="alice", tenant="jd", order_id=None,
                          topic="other", severity="low", content="C")

    # Filter to t-1 → see ONLY A. C (legacy) and B (other session) excluded.
    r = client.get("/users/alice/inbox", params={"thread_id": "t-1"})
    ids = {it["complaint_id"] for it in r.json()["items"]}
    assert ids == {cA["id"]}

    # Filter to t-2 → see ONLY B.
    r = client.get("/users/alice/inbox", params={"thread_id": "t-2"})
    ids = {it["complaint_id"] for it in r.json()["items"]}
    assert ids == {cB["id"]}

    # No filter → see all three (legacy still reachable for admin / global view).
    r = client.get("/users/alice/inbox")
    ids = {it["complaint_id"] for it in r.json()["items"]}
    assert ids == {cA["id"], cB["id"], cC["id"]}
