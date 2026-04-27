"""Extension-stage GDPR + retention tests.

Covers:
- purge_user: deletes user + feedback + audit rows for that user; cascades
  orders + order_items + return_requests via FK
- purge_user: non-existent user is a no-op (counts all zero)
- purge_user: dry_run reports counts without deleting
- purge_user: records a `gdpr_delete` audit row with user_id_hash (NOT raw id)
- cleanup: deletes audit_logs older than audit_days
- cleanup: deletes feedback older than feedback_days
- cleanup: deletes only TERMINAL return_requests older than the window
  (pending / approved ones are kept)
- cleanup: dry_run reports counts without deleting
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rag.db.base import Base, SessionLocal, engine
from rag.db.models import (
    AuditLog,
    Feedback,
    Order,
    OrderItem,
    ReturnRequest,
    User,
)
from scripts.gdpr_delete import purge_user
from scripts.retention_cleanup import cleanup


@pytest.fixture(autouse=True)
def _fresh_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


def _seed_user_with_everything(user_id: str = "u1", tenant: str = "jd"):
    with SessionLocal() as s:
        u = User(id=user_id, tenant=tenant, display_name="T", phone=None)
        s.add(u)
        o = Order(
            id=f"O-{user_id}",
            user_id=user_id,
            tenant=tenant,
            status="delivered",
            total_cents=10000,
            placed_at=datetime.now(timezone.utc),
        )
        s.add(o)
        s.flush()
        s.add(OrderItem(order_id=o.id, sku="SKU1", title="iPhone", qty=1, unit_price_cents=10000))
        s.add(ReturnRequest(
            order_id=o.id, tenant=tenant, kind="return", reason="test",
            status="pending", refund_cents=10000,
        ))
        s.add(Feedback(
            trace_id="t1", kb_id="jd_demo", query="q", answer="a",
            verdict="down", reason=None, user_id=user_id,
        ))
        s.add(AuditLog(event_type="unit", user_id=user_id, tenant_id=tenant))
        s.add(AuditLog(event_type="unit", user_id=None, tenant_id=tenant))  # unrelated
        s.commit()


# ---------- purge_user ----------

def test_purge_user_deletes_all_user_data():
    _seed_user_with_everything()
    counts = purge_user("u1")
    assert counts["users"] == 1
    assert counts["orders"] == 1
    assert counts["feedback"] == 1
    assert counts["audit_logs"] >= 1  # at least the unit row we seeded

    with SessionLocal() as s:
        assert s.get(User, "u1") is None
        assert s.query(Order).count() == 0
        assert s.query(OrderItem).count() == 0
        assert s.query(ReturnRequest).count() == 0
        assert s.query(Feedback).filter(Feedback.user_id == "u1").count() == 0
        assert s.query(AuditLog).filter(AuditLog.user_id == "u1").count() == 0
        # non-user-scoped audit rows untouched (plus the new gdpr_delete row)
        assert s.query(AuditLog).filter(AuditLog.user_id.is_(None)).count() >= 1


def test_purge_user_nonexistent_is_noop():
    counts = purge_user("nobody")
    assert counts == {"users": 0, "orders": 0, "feedback": 0, "audit_logs": 0}


def test_purge_user_dry_run_reports_but_deletes_nothing():
    _seed_user_with_everything()
    counts = purge_user("u1", dry_run=True)
    assert counts["users"] == 1
    with SessionLocal() as s:
        assert s.get(User, "u1") is not None
        assert s.query(Feedback).count() == 1
        assert s.query(AuditLog).filter(AuditLog.user_id == "u1").count() == 1


def test_purge_user_writes_gdpr_delete_audit_row():
    _seed_user_with_everything("alice")
    purge_user("alice")
    with SessionLocal() as s:
        rows = s.query(AuditLog).filter(AuditLog.event_type == "gdpr_delete").all()
        assert len(rows) == 1
        # Raw user_id must NOT appear anywhere in the row.
        assert rows[0].user_id is None
        hashed = rows[0].extra.get("user_id_hash")
        assert hashed and "alice" not in hashed and len(hashed) == 16


# ---------- retention cleanup ----------

def test_cleanup_deletes_old_audit():
    now = datetime.now(timezone.utc)
    with SessionLocal() as s:
        s.add(AuditLog(event_type="old", created_at=now - timedelta(days=200)))
        s.add(AuditLog(event_type="recent", created_at=now - timedelta(days=10)))
        s.commit()

    counts = cleanup(audit_days=180, feedback_days=9999, return_request_days=9999)
    assert counts["audit_logs"] == 1
    with SessionLocal() as s:
        kept = s.query(AuditLog).filter(AuditLog.event_type.in_(["old", "recent"])).all()
        assert len(kept) == 1
        assert kept[0].event_type == "recent"


def test_cleanup_deletes_old_feedback():
    now = datetime.now(timezone.utc)
    with SessionLocal() as s:
        s.add(Feedback(trace_id="t", kb_id="k", query="q", answer="a", verdict="up",
                       created_at=now - timedelta(days=400)))
        s.add(Feedback(trace_id="t", kb_id="k", query="q", answer="a", verdict="up",
                       created_at=now - timedelta(days=10)))
        s.commit()

    counts = cleanup(audit_days=9999, feedback_days=365, return_request_days=9999)
    assert counts["feedback"] == 1
    with SessionLocal() as s:
        assert s.query(Feedback).count() == 1


def test_cleanup_only_deletes_terminal_return_requests():
    now = datetime.now(timezone.utc)
    with SessionLocal() as s:
        # Seed supporting order first (FK cascade)
        s.add(User(id="u", tenant="jd", display_name="x"))
        s.add(Order(id="O1", user_id="u", tenant="jd", status="delivered",
                    total_cents=100, placed_at=now - timedelta(days=400)))
        s.flush()
        s.add(ReturnRequest(order_id="O1", tenant="jd", kind="return", reason="r",
                            status="completed",
                            created_at=now - timedelta(days=400)))
        s.add(ReturnRequest(order_id="O1", tenant="jd", kind="return", reason="r",
                            status="pending",  # non-terminal
                            created_at=now - timedelta(days=400)))
        s.add(ReturnRequest(order_id="O1", tenant="jd", kind="return", reason="r",
                            status="completed",
                            created_at=now - timedelta(days=10)))  # recent
        s.commit()

    counts = cleanup(audit_days=9999, feedback_days=9999, return_request_days=365)
    assert counts["return_requests"] == 1  # only old + terminal
    with SessionLocal() as s:
        remaining = s.query(ReturnRequest).all()
        assert len(remaining) == 2
        # pending old one kept + completed recent one kept
        assert {r.status for r in remaining} == {"pending", "completed"}


def test_cleanup_dry_run_leaves_rows():
    now = datetime.now(timezone.utc)
    with SessionLocal() as s:
        s.add(AuditLog(event_type="old", created_at=now - timedelta(days=400)))
        s.commit()
    counts = cleanup(audit_days=180, dry_run=True)
    assert counts["audit_logs"] == 1
    with SessionLocal() as s:
        assert s.query(AuditLog).filter(AuditLog.event_type == "old").count() == 1
