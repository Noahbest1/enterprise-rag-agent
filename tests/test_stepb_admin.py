"""Step B: admin routes (list / get / claim / reply) + audit writes.

Verifies:
- GET /admin/complaints returns existing complaints, filterable.
- GET /admin/complaints/{id} returns complaint + empty reply list initially.
- POST /admin/complaints/{id}/claim flips assigned_to + status, writes audit.
- POST /admin/complaints/{id}/reply persists reply, writes audit with hashed
  content (raw content NEVER lands in the audit row's extra).
- POST /admin/complaints/{id}/reply on a missing ticket returns 404.
- Reply publishes an event on the in-process user bus (caller sees
  subscribers_notified count when a live subscriber exists).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent.tools.complaints import create_complaint
from rag.db.base import Base, SessionLocal, engine
from rag.db.models import AuditLog, ComplaintReply
from rag_api.main import app
from rag_api.user_events import active_subscriber_count, publish_user_event


@pytest.fixture(autouse=True)
def _fresh_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture()
def client():
    return TestClient(app)


def _make_complaint(user_id: str = "jd-demo-user", tenant: str = "jd",
                    severity: str = "medium") -> dict:
    return create_complaint(
        user_id=user_id,
        tenant=tenant,
        order_id=None,
        topic="quality",
        severity=severity,
        content="物流太慢了",
    )


def test_admin_list_complaints_returns_newest_first(client):
    c1 = _make_complaint(severity="low")
    c2 = _make_complaint(severity="high")  # more recent
    resp = client.get("/admin/complaints")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    # newest first
    assert data["items"][0]["id"] == c2["id"]
    assert data["items"][1]["id"] == c1["id"]


def test_admin_list_complaints_filter_only_escalated(client):
    low = _make_complaint(severity="low")        # NOT escalated
    high = _make_complaint(severity="high")      # escalated auto
    resp = client.get("/admin/complaints?only_escalated=true")
    assert resp.status_code == 200
    ids = [x["id"] for x in resp.json()["items"]]
    assert high["id"] in ids
    assert low["id"] not in ids


def test_admin_get_complaint_with_replies(client):
    c = _make_complaint()
    resp = client.get(f"/admin/complaints/{c['id']}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["complaint"]["id"] == c["id"]
    assert data["replies"] == []


def test_admin_get_complaint_404(client):
    resp = client.get("/admin/complaints/99999")
    assert resp.status_code == 404


def test_admin_claim_complaint_flips_status_and_audits(client):
    c = _make_complaint(severity="low")
    assert c["status"] == "open"
    resp = client.post(f"/admin/complaints/{c['id']}/claim",
                       json={"assigned_to": "jd-cs-senior-A"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["complaint"]["assigned_to"] == "jd-cs-senior-A"
    assert body["complaint"]["status"] == "escalated"

    # audit row
    with SessionLocal() as s:
        from sqlalchemy import select
        rows = s.execute(
            select(AuditLog).where(AuditLog.event_type == "complaint_claimed")
        ).scalars().all()
    assert len(rows) == 1
    assert rows[0].extra["complaint_id"] == c["id"]
    assert rows[0].extra["assigned_to"] == "jd-cs-senior-A"
    assert rows[0].extra["previous_status"] == "open"


def test_admin_claim_complaint_404(client):
    resp = client.post("/admin/complaints/99999/claim",
                       json={"assigned_to": "jd-cs-senior-A"})
    assert resp.status_code == 404


def test_admin_reply_persists_and_audits_hashed(client):
    c = _make_complaint()
    raw_reply = "您好 已经帮您催促仓库今天发货"
    resp = client.post(f"/admin/complaints/{c['id']}/reply",
                       json={"author_label": "jd-cs-senior-A", "content": raw_reply})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["reply"]["content"] == raw_reply
    assert body["reply"]["author_label"] == "jd-cs-senior-A"
    assert body["reply"]["author_kind"] == "admin"
    assert body["reply"]["user_id"] == "jd-demo-user"
    # no live subscriber in this test
    assert body["subscribers_notified"] == 0

    # row persisted
    with SessionLocal() as s:
        from sqlalchemy import select
        replies = s.execute(select(ComplaintReply)).scalars().all()
    assert len(replies) == 1
    assert replies[0].content == raw_reply

    # audit row uses hash, NOT raw content
    with SessionLocal() as s:
        from sqlalchemy import select
        rows = s.execute(
            select(AuditLog).where(AuditLog.event_type == "complaint_replied")
        ).scalars().all()
    assert len(rows) == 1
    extra = rows[0].extra
    assert extra["complaint_id"] == c["id"]
    assert extra["author_label"] == "jd-cs-senior-A"
    assert len(extra["content_hash"]) == 16
    assert raw_reply not in str(extra)  # paranoia: raw text never in audit
    assert extra["user_id_hash"] is not None
    assert "jd-demo-user" not in str(extra)  # raw user_id also never in audit


def test_admin_reply_404(client):
    resp = client.post("/admin/complaints/99999/reply",
                       json={"author_label": "anyone", "content": "hi"})
    assert resp.status_code == 404


def test_admin_reply_409_on_closed_complaint(client):
    """Once user closes a complaint, admin replies are rejected with 409.
    UI hides the reply box, but the API gate is the actual integrity check."""
    c = _make_complaint()
    # Close the complaint (user side does this via /agent/actions/cancel-complaint)
    with SessionLocal() as s:
        from rag.db.models import Complaint as CModel
        row = s.get(CModel, c["id"])
        row.status = "closed"
        row.escalated = False
        s.commit()
    r = client.post(f"/admin/complaints/{c['id']}/reply",
                    json={"author_label": "admin-X", "content": "let me sneak in"})
    assert r.status_code == 409
    assert "closed" in r.json()["detail"]


def test_admin_reply_empty_content_422(client):
    c = _make_complaint()
    # pydantic min_length=1 rejects at validation time
    resp = client.post(f"/admin/complaints/{c['id']}/reply",
                       json={"author_label": "x", "content": ""})
    assert resp.status_code == 422


def test_admin_reply_reports_subscribers_notified(client, monkeypatch):
    c = _make_complaint(user_id="jd-demo-user")

    # Inject a fake subscriber by directly manipulating the bus so we can
    # observe the count without spinning up an actual SSE client here.
    import asyncio
    from rag_api import user_events
    q: asyncio.Queue = asyncio.Queue(maxsize=4)
    user_events._subscribers["jd-demo-user"].append(q)
    assert active_subscriber_count("jd-demo-user") == 1
    try:
        resp = client.post(f"/admin/complaints/{c['id']}/reply",
                           json={"author_label": "admin-A", "content": "hi"})
        assert resp.status_code == 200
        assert resp.json()["subscribers_notified"] == 1
        # the fake subscriber received the event
        item = q.get_nowait()
        assert item["event"] == "complaint_reply"
        assert item["data"]["content"] == "hi"
        assert item["data"]["complaint_id"] == c["id"]
    finally:
        user_events._subscribers["jd-demo-user"].remove(q)
        if not user_events._subscribers["jd-demo-user"]:
            user_events._subscribers.pop("jd-demo-user", None)


def test_publish_user_event_returns_delivered_count_zero_when_no_subscribers():
    # No subscriber for this user -> 0 delivered.
    n = publish_user_event("nobody", "complaint_reply", {"x": 1})
    assert n == 0


def test_inbox_rehydrates_replies_after_page_reload(client):
    """User opens AgentChat after admin replied while no SSE subscriber
    was watching. The /users/{id}/inbox endpoint must surface those
    replies so the chat can show them as system bubbles."""
    c = _make_complaint(user_id="rehydrate-user")
    r1 = client.post(f"/admin/complaints/{c['id']}/reply",
                     json={"author_label": "admin-X", "content": "first reply"})
    assert r1.status_code == 200
    r2 = client.post(f"/admin/complaints/{c['id']}/reply",
                     json={"author_label": "admin-Y", "content": "second reply"})
    assert r2.status_code == 200

    # User opens chat → frontend hits inbox.
    inbox = client.get("/users/rehydrate-user/inbox").json()
    assert len(inbox["items"]) == 1
    item = inbox["items"][0]
    assert item["complaint_id"] == c["id"]
    assert item["severity"] == c["severity"]
    contents = [r["content"] for r in item["replies"]]
    assert contents == ["first reply", "second reply"]
    # Replies are time-ordered oldest -> newest within a complaint.
    ts = [r["created_at"] for r in item["replies"]]
    assert ts == sorted(ts)


def test_inbox_filters_by_since_timestamp(client):
    """?since=<earlier> returns everything; ?since=<future> returns nothing.

    SQLite's CURRENT_TIMESTAMP has 1-second granularity, so we don't try
    to slice between two replies in the same second — we test the bracket
    boundaries instead, which is what production callers rely on anyway."""
    from datetime import datetime, timedelta, timezone

    c = _make_complaint(user_id="since-user")
    client.post(f"/admin/complaints/{c['id']}/reply",
                json={"author_label": "a", "content": "first"})
    client.post(f"/admin/complaints/{c['id']}/reply",
                json={"author_label": "b", "content": "second"})

    earlier = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

    # Use params=... so TestClient URL-encodes the + in the timezone offset.
    inbox_all = client.get("/users/since-user/inbox", params={"since": earlier}).json()
    assert len(inbox_all["items"]) == 1
    assert [r["content"] for r in inbox_all["items"][0]["replies"]] == ["first", "second"]

    inbox_none = client.get("/users/since-user/inbox", params={"since": future}).json()
    assert inbox_none["items"] == []


def test_inbox_isolates_per_user(client):
    """One user's complaint replies must not leak to another user's inbox."""
    c_a = _make_complaint(user_id="alice")
    c_b = _make_complaint(user_id="bob")
    client.post(f"/admin/complaints/{c_a['id']}/reply",
                json={"author_label": "x", "content": "to alice"})
    client.post(f"/admin/complaints/{c_b['id']}/reply",
                json={"author_label": "y", "content": "to bob"})

    alice = client.get("/users/alice/inbox").json()
    bob = client.get("/users/bob/inbox").json()
    alice_msgs = [r["content"] for it in alice["items"] for r in it["replies"]]
    bob_msgs = [r["content"] for it in bob["items"] for r in it["replies"]]
    assert "to alice" in alice_msgs and "to bob" not in alice_msgs
    assert "to bob" in bob_msgs and "to alice" not in bob_msgs


def test_user_reply_persists_with_user_kind(client):
    """User-side replies on a complaint thread persist with author_kind='user'."""
    c = _make_complaint(user_id="reply-user")
    r = client.post(f"/complaints/{c['id']}/user-reply",
                    json={"user_id": "reply-user", "content": "我刚才提交的工单怎么样了"})
    assert r.status_code == 200, r.text
    rep = r.json()["reply"]
    assert rep["author_kind"] == "user"
    assert rep["author_label"] == "reply-user"
    assert rep["content"] == "我刚才提交的工单怎么样了"

    # Admin viewing the thread sees the user reply.
    detail = client.get(f"/admin/complaints/{c['id']}").json()
    kinds = [r["author_kind"] for r in detail["replies"]]
    assert "user" in kinds


def test_user_reply_rejects_cross_user_spoofing(client):
    """User X can't write a reply into user Y's complaint thread."""
    c = _make_complaint(user_id="alice")
    r = client.post(f"/complaints/{c['id']}/user-reply",
                    json={"user_id": "mallory", "content": "假冒 alice"})
    assert r.status_code == 404


def test_user_reply_404_on_missing_complaint(client):
    r = client.post("/complaints/99999/user-reply",
                    json={"user_id": "x", "content": "hello"})
    assert r.status_code == 404


def test_user_reply_audits_with_content_hash_only(client):
    from rag.db.models import AuditLog
    c = _make_complaint(user_id="auditme")
    raw = "敏感原文不该入审计 12345"
    client.post(f"/complaints/{c['id']}/user-reply",
                json={"user_id": "auditme", "content": raw})

    with SessionLocal() as s:
        from sqlalchemy import select
        rows = s.execute(
            select(AuditLog).where(AuditLog.event_type == "complaint_user_replied")
        ).scalars().all()
    assert len(rows) == 1
    extra = rows[0].extra
    assert extra["complaint_id"] == c["id"]
    assert len(extra["content_hash"]) == 16
    # Raw user content NEVER in the audit row.
    assert raw not in str(extra)
    assert "12345" not in str(extra)


def test_user_and_admin_replies_interleave_in_history(client):
    """Same thread, multiple replies from both sides — chronological order
    preserved, both visible via /admin/complaints/{id}."""
    c = _make_complaint(user_id="alice")
    client.post(f"/admin/complaints/{c['id']}/reply",
                json={"author_label": "admin-A", "content": "admin says 1"})
    client.post(f"/complaints/{c['id']}/user-reply",
                json={"user_id": "alice", "content": "user says 1"})
    client.post(f"/admin/complaints/{c['id']}/reply",
                json={"author_label": "admin-A", "content": "admin says 2"})

    detail = client.get(f"/admin/complaints/{c['id']}").json()
    replies = detail["replies"]
    assert len(replies) == 3
    # Server orders oldest -> newest by created_at, so insertion order wins.
    assert [r["content"] for r in replies] == [
        "admin says 1", "user says 1", "admin says 2",
    ]
    assert [r["author_kind"] for r in replies] == ["admin", "user", "admin"]


def test_inbox_400_on_bad_since():
    from fastapi.testclient import TestClient
    from rag_api.main import app
    c = TestClient(app)
    r = c.get("/users/whoever/inbox?since=not-a-date")
    assert r.status_code == 400
