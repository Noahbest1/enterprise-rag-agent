"""Tier-2 Complaint specialist tests.

Covers:
- emotion.detect_severity: high signals (12315 / 曝光 / 起诉) vs medium (不满 /
  退不了) vs low (plain inquiry)
- emotion.detect_topic: delivery / quality / refund / service / price / other
- emotion.classify: bundles severity + topic + matched markers
- create_complaint: high severity auto-escalates; medium/low stay open;
  SLA due time roughly matches severity; content_hash is sha256[:16] of
  the raw content (never the raw string)
- complaint_node: writes a row and expose entities; uses the RAW user
  utterance for severity (not the rewritten planner sub-query)
- complaint_node: high severity writes a `complaint_escalated` audit row
  with no raw user content, just the hash
- complaint_node: resolves order_id from entities.last_order_id
- planner: accepts "complaint" agent; agent_routes specialist_names includes it
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from rag.db.base import Base, SessionLocal, engine
from rag.db.models import AuditLog, Complaint, Order, User
from rag.nlp.emotion import classify, detect_severity, detect_topic
from agent.specialists.complaint import complaint_node
from agent.tools.complaints import create_complaint


@pytest.fixture(autouse=True)
def _fresh_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


def _state(query: str, *, tenant: str = "jd", entities: dict | None = None,
           user_utter: str | None = None):
    msgs = [{"role": "user", "content": user_utter if user_utter is not None else query}]
    return {
        "tenant": tenant,
        "user_id": "u1",
        "messages": msgs,
        "entities": entities or {},
        "step_results": {},
        "trace": [],
        "plan": [{"step_id": 1, "agent": "complaint", "query": query, "depends_on": [], "status": "pending"}],
        "current_step": 0,
    }


def _seed_order():
    with SessionLocal() as s:
        s.add(User(id="u1", tenant="jd", display_name="x"))
        s.add(Order(id="O1", user_id="u1", tenant="jd", status="delivered",
                    total_cents=100, placed_at=datetime.now(timezone.utc)))
        s.commit()


# ---------- emotion module ----------

def test_severity_high_regulatory():
    sev, high, med = detect_severity("你们再不处理我就打 12315")
    assert sev == "high"
    assert any("12315" in h for h in high)


def test_severity_high_media():
    sev, high, _ = detect_severity("我要去微博曝光你们的行为")
    assert sev == "high"
    assert any("曝光" in h for h in high)


def test_severity_high_ultimatum():
    sev, _, _ = detect_severity("必须今天给我退款,否则我起诉")
    assert sev == "high"


def test_severity_medium_unhappy():
    sev, high, med = detect_severity("我对这次购物非常不满意,想退货")
    assert sev == "medium"
    assert high == []
    assert med  # at least one medium signal


def test_severity_low_plain_inquiry():
    sev, _, _ = detect_severity("请问什么时候发货")
    assert sev == "low"


def test_severity_low_empty():
    assert detect_severity("")[0] == "low"
    assert detect_severity("   ")[0] == "low"


def test_topic_delivery():
    t, _ = detect_topic("快递一直没到,已经 5 天了")
    assert t == "delivery"


def test_topic_quality():
    t, _ = detect_topic("买来就坏了,明显是次品")
    assert t == "quality"


def test_topic_refund():
    t, _ = detect_topic("申请退款一周了一直没退")
    assert t == "refund"


def test_topic_other_when_no_match():
    t, m = detect_topic("这是什么情况啊")
    assert t == "other" and m is None


def test_classify_bundles_fields():
    v = classify("差评!产品质量太差,必须今天退钱否则 12315")
    assert v.severity == "high"
    assert v.topic == "quality"
    assert v.matched_high  # non-empty


# ---------- create_complaint ----------

def test_create_complaint_high_auto_escalates():
    c = create_complaint(
        user_id="u1", tenant="jd", order_id=None, topic="service",
        severity="high", content="你们客服态度真的太差了,必须 12315",
    )
    assert c["escalated"] is True
    assert c["status"] == "escalated"
    assert c["assigned_to"]  # non-empty
    # SLA is ~1h for high. SQLite stores naive UTC; re-attach tz for comparison.
    due = datetime.fromisoformat(c["sla_due_at"])
    if due.tzinfo is None:
        due = due.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta = due - now
    assert timedelta(minutes=55) <= delta <= timedelta(minutes=65)


def test_create_complaint_medium_stays_open():
    c = create_complaint(
        user_id="u1", tenant="jd", order_id=None, topic="refund",
        severity="medium", content="退款有点慢",
    )
    assert c["status"] == "open"
    assert c["escalated"] is False
    assert c["assigned_to"] is None
    due = datetime.fromisoformat(c["sla_due_at"])
    if due.tzinfo is None:
        due = due.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    assert timedelta(hours=3) < (due - now) < timedelta(hours=5)


def test_create_complaint_hashes_content_not_stores_raw():
    content = "超级个性化 PII 内容,不应该入 complaints 表"
    c = create_complaint(
        user_id="u1", tenant="jd", order_id=None, topic="other",
        severity="low", content=content,
    )
    expected = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    assert c["content_hash"] == expected
    with SessionLocal() as s:
        row = s.get(Complaint, c["id"])
        # Verify no attribute on the model carries the raw content.
        for col in ("content", "raw", "text"):
            assert not hasattr(row, col) or getattr(row, col) is None


# ---------- complaint_node ----------

def test_complaint_node_dry_run_writes_no_db_row():
    """Dry-run: classify + return preview, NEVER touch the DB."""
    out = complaint_node(_state("不满意,商品有点质量问题"))
    assert "entities" in out
    ent = out["entities"]
    # Preview-flavoured entity keys (no last_complaint_id since nothing's
    # been written yet)
    assert ent["last_complaint_preview_severity"] in ("medium", "low")
    assert ent["last_complaint_preview_would_escalate"] is False
    assert "last_complaint_id" not in ent

    step_res = out["step_results"][1]
    assert step_res["abstain"] is False
    assert step_res["complaint"] is None  # no row created
    assert step_res["preview"]["topic"] == "quality"
    assert step_res["preview"]["would_escalate"] is False
    assert "提交工单" in step_res["answer"] or "确认提交" in step_res["answer"]
    assert step_res["topic"] == "quality"

    # Crucially: zero rows in complaints table.
    with SessionLocal() as s:
        assert s.query(Complaint).count() == 0


def test_complaint_node_high_severity_still_dry_run_no_audit():
    """Even high-severity utterances stay dry-run — no DB row, no audit
    row. The button at /agent/actions/submit-complaint is the sole path."""
    out = complaint_node(_state(
        "这快递根本就没送,客服不理人,我直接 12315 投诉你们,否则微博曝光",
    ))
    ent = out["entities"]
    assert ent["last_complaint_preview_would_escalate"] is True
    assert ent["last_complaint_preview_severity"] == "high"

    step_res = out["step_results"][1]
    assert step_res["complaint"] is None
    preview = step_res["preview"]
    assert preview["would_escalate"] is True
    assert preview["suggested_assignee"]  # mock human CS agent
    # Preview includes the raw user utterance so the submit endpoint
    # can hash it server-side; don't leak it to other callers though.
    assert preview["content_for_submit"]

    with SessionLocal() as s:
        assert s.query(Complaint).count() == 0
        audit_rows = s.query(AuditLog).filter(AuditLog.event_type == "complaint_escalated").all()
        assert len(audit_rows) == 0  # audit only fires on real submit, not preview


def test_complaint_node_uses_raw_user_utterance_not_rewritten_query():
    # Planner-rewritten query is neutral; the USER utterance is furious.
    # The specialist must classify against the user utterance.
    out = complaint_node(_state(
        query="处理用户的投诉",  # calm planner rewrite
        user_utter="你们再不退钱我直接起诉,走法律途径",
    ))
    assert out["step_results"][1]["severity"] == "high"


def test_complaint_node_resolves_order_from_entities():
    _seed_order()
    out = complaint_node(_state(
        "这订单的东西有质量问题",
        entities={"last_order_id": "O1"},
    ))
    # Order id surfaces in the preview (not in a complaint row, since
    # we're dry-run now).
    assert out["step_results"][1]["preview"]["order_id"] == "O1"


def test_complaint_node_resolves_order_from_query():
    _seed_order()
    with SessionLocal() as s:
        s.add(Order(id="TB99999999", user_id="u1", tenant="jd", status="delivered",
                    total_cents=100, placed_at=datetime.now(timezone.utc)))
        s.commit()
    out = complaint_node(_state("订单 TB99999999 有问题不满意"))
    assert out["step_results"][1]["preview"]["order_id"] == "TB99999999"


def test_submit_complaint_endpoint_writes_row_and_audit():
    """The /agent/actions/submit-complaint endpoint is the SOLE DB-write
    path. Hits ``create_complaint`` + writes the audit row for high-severity.
    """
    from fastapi.testclient import TestClient
    from rag_api.main import app

    client = TestClient(app)
    body = {
        "severity": "high",
        "topic": "delivery",
        "user_id": "u1",
        "tenant": "jd",
        "order_id": None,
        "content": "我要去 12315 投诉,你们物流太差",
    }
    r = client.post("/agent/actions/submit-complaint", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["ok"] is True
    c = data["complaint"]
    assert c["escalated"] is True
    assert c["assigned_to"]  # mock human CS agent
    assert c["status"] == "escalated"

    # DB now has exactly one complaint row + one audit row.
    with SessionLocal() as s:
        assert s.query(Complaint).count() == 1
        audit_rows = s.query(AuditLog).filter(AuditLog.event_type == "complaint_escalated").all()
        assert len(audit_rows) == 1
        extra = audit_rows[0].extra
        # Raw content NEVER persisted — hash only.
        assert extra["content_hash"]
        assert "我要去" not in str(extra)
        assert extra["via"] == "user_button"
        assert extra["severity"] == "high"


def test_submit_complaint_endpoint_low_severity_does_not_audit():
    """Low/medium submissions don't trip the escalation audit breadcrumb."""
    from fastapi.testclient import TestClient
    from rag_api.main import app

    client = TestClient(app)
    body = {
        "severity": "low",
        "topic": "other",
        "user_id": "u1",
        "tenant": "jd",
        "order_id": None,
        "content": "随便提个建议",
    }
    r = client.post("/agent/actions/submit-complaint", json=body)
    assert r.status_code == 200
    c = r.json()["complaint"]
    assert c["escalated"] is False
    assert c["status"] == "open"

    with SessionLocal() as s:
        audit_rows = s.query(AuditLog).filter(AuditLog.event_type == "complaint_escalated").all()
        assert len(audit_rows) == 0


def test_submit_complaint_endpoint_rejects_bad_severity():
    from fastapi.testclient import TestClient
    from rag_api.main import app

    client = TestClient(app)
    r = client.post("/agent/actions/submit-complaint", json={
        "severity": "EXTREME", "topic": "other",
        "user_id": "u1", "tenant": "jd", "order_id": None, "content": "x",
    })
    assert r.status_code == 400


def test_reopen_complaint_high_severity_re_escalates():
    """High severity reopens go straight back to escalated state with a
    fresh assignee + 1h SLA — same logic as create_complaint, since
    severity didn't change."""
    from agent.tools.complaints import create_complaint, reopen_complaint
    from rag.db.base import SessionLocal
    from rag.db.models import Complaint as CModel

    c = create_complaint(user_id="u1", tenant="jd", order_id=None,
                        topic="delivery", severity="high",
                        content="12315 投诉")
    # User closes:
    with SessionLocal() as s:
        row = s.get(CModel, c["id"])
        row.status = "closed"
        row.escalated = False
        s.commit()

    reopened = reopen_complaint(c["id"])
    assert reopened["status"] == "escalated"
    assert reopened["escalated"] is True
    assert reopened["assigned_to"]  # freshly picked

    # SLA roughly 1h.
    due = datetime.fromisoformat(reopened["sla_due_at"])
    if due.tzinfo is None:
        due = due.replace(tzinfo=timezone.utc)
    delta = due - datetime.now(timezone.utc)
    assert timedelta(minutes=55) <= delta <= timedelta(minutes=65)


def test_reopen_complaint_low_severity_goes_to_open():
    """Low severity reopens go to status=open with the standard 24h SLA;
    escalated stays false."""
    from agent.tools.complaints import create_complaint, reopen_complaint
    from rag.db.base import SessionLocal
    from rag.db.models import Complaint as CModel

    c = create_complaint(user_id="u1", tenant="jd", order_id=None,
                        topic="other", severity="low", content="meh")
    with SessionLocal() as s:
        s.get(CModel, c["id"]).status = "closed"
        s.commit()

    reopened = reopen_complaint(c["id"])
    assert reopened["status"] == "open"
    assert reopened["escalated"] is False
    assert reopened["assigned_to"] is None


def test_reopen_complaint_rejects_non_closed():
    """Don't let callers reopen a complaint that's already active —
    that would silently re-shuffle the SLA mid-conversation."""
    from agent.tools.complaints import create_complaint, reopen_complaint
    c = create_complaint(user_id="u1", tenant="jd", order_id=None,
                        topic="other", severity="medium", content="x")
    # status is "open" — reopen should refuse.
    try:
        reopen_complaint(c["id"])
    except ValueError as e:
        assert "not closed" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_reopen_complaint_endpoint_full_round_trip():
    """End-to-end via the API endpoint — covers the audit row + 409 on
    non-closed + 404 on missing."""
    from fastapi.testclient import TestClient
    from rag_api.main import app

    client = TestClient(app)
    c = create_complaint(user_id="u1", tenant="jd", order_id=None,
                         topic="quality", severity="high", content="x")
    # Reopen-before-close should 409.
    r = client.post("/agent/actions/reopen-complaint", json={"complaint_id": c["id"]})
    assert r.status_code == 409

    # Close, then reopen — should succeed and write audit.
    with SessionLocal() as s:
        s.get(Complaint, c["id"]).status = "closed"
        s.commit()
    r = client.post("/agent/actions/reopen-complaint", json={"complaint_id": c["id"]})
    assert r.status_code == 200
    body = r.json()
    assert body["complaint"]["status"] == "escalated"
    assert body["complaint"]["escalated"] is True

    # Audit row.
    with SessionLocal() as s:
        from sqlalchemy import select
        rows = s.execute(
            select(AuditLog).where(AuditLog.event_type == "complaint_reopened")
        ).scalars().all()
    assert len(rows) == 1
    assert rows[0].extra["complaint_id"] == c["id"]
    assert rows[0].extra["new_status"] == "escalated"

    # 404 on missing.
    r = client.post("/agent/actions/reopen-complaint", json={"complaint_id": 99999})
    assert r.status_code == 404


def test_planner_routes_greeting_to_policy_qa_not_complaint():
    """Regression: '你好 你是什么客服' was being mis-classified as a complaint
    (the word '客服' aliased to the complaint specialist), creating ghost
    tickets. Planner prompt now has explicit greeting examples + a rule
    pinning small talk to policy_qa."""
    from agent.planner import SYSTEM
    # The system prompt MUST instruct that greetings/identity-questions
    # default to policy_qa, not complaint.
    assert "你好 你是什么客服" in SYSTEM
    assert 'policy_qa' in SYSTEM
    # And explicitly tell the LLM not to treat the word "客服" as complaint.
    assert "不要" in SYSTEM or "Do NOT" in SYSTEM or "do not" in SYSTEM.lower()


# ---------- integration with planner + agent_routes ----------

def test_planner_accepts_complaint_agent():
    from agent.planner import _parse_plan
    reply = '[{"step_id":1,"agent":"complaint","query":"受理投诉","depends_on":[]}]'
    plan = _parse_plan(reply)
    assert plan is not None and plan[0]["agent"] == "complaint"


def test_agent_routes_treats_complaint_as_specialist():
    import src.rag_api.agent_routes as ar
    import inspect
    src = inspect.getsource(ar._stream_agent_turn)
    assert '"complaint"' in src
