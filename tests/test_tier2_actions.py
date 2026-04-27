"""Tier-2 interactive action endpoint tests.

Covers:
- GET /invoice/{id}.pdf returns a real PDF (magic bytes %PDF)
- GET /invoice/{id}.pdf 404 when id is missing
- POST /agent/actions/confirm-return writes a ReturnRequest row
- POST /agent/actions/set-default-address flips the flag in DB
- POST /agent/actions/cancel-complaint closes the complaint
- POST /agent/actions/verify-phone-change:
    bad code -> 400, user.phone unchanged
    good code -> 200, user.phone updated, audit row hashed (no raw phone)
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from rag.db.base import Base, SessionLocal, engine
from rag.db.models import (
    AuditLog,
    Complaint,
    Invoice,
    Order,
    OrderItem,
    ReturnRequest,
    User,
    UserAddress,
)


@pytest.fixture(autouse=True)
def _fresh_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture
def client():
    from rag_api.main import app
    return TestClient(app)


def _seed_full():
    """User + delivered order + addresses + one requested invoice."""
    with SessionLocal() as s:
        s.add(User(id="u1", tenant="jd", display_name="demo", phone="13800001111"))
        o = Order(id="O1", user_id="u1", tenant="jd", status="delivered",
                  total_cents=10000, placed_at=datetime.now(timezone.utc))
        s.add(o); s.flush()
        s.add(OrderItem(order_id="O1", sku="SKU", title="iPhone", qty=1, unit_price_cents=10000))
        s.add(UserAddress(
            user_id="u1", label="家", recipient="张三", phone="13800001111",
            line1="北京 朝阳 1 号", is_default=True,
        ))
        s.add(UserAddress(
            user_id="u1", label="公司", recipient="张三", phone="13800001111",
            line1="北京 海淀 27 号", is_default=False,
        ))
        inv = Invoice(
            order_id="O1", tenant="jd", title="个人", tax_id=None,
            invoice_type="electronic", amount_cents=10000, status="issued",
            download_url=None,
        )
        s.add(inv)
        s.commit()
        return {"invoice_id": inv.id}


# ---------- /invoice/{id}.pdf ----------

def test_invoice_pdf_returns_real_pdf(client: TestClient):
    ids = _seed_full()
    r = client.get(f"/invoice/{ids['invoice_id']}.pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    # PDFs start with "%PDF-"
    assert r.content[:5] == b"%PDF-"
    # Non-trivial body
    assert len(r.content) > 1000


def test_invoice_pdf_missing_returns_404(client: TestClient):
    r = client.get("/invoice/99999.pdf")
    assert r.status_code == 404


# ---------- /agent/actions/confirm-return ----------

def test_confirm_return_writes_row(client: TestClient):
    _seed_full()
    r = client.post("/agent/actions/confirm-return", json={"order_id": "O1"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    # returns.py shape: request_id (not id)
    assert body["request"]["request_id"]

    with SessionLocal() as s:
        rows = s.query(ReturnRequest).filter(ReturnRequest.order_id == "O1").all()
        assert len(rows) == 1
        assert rows[0].status in ("pending", "approved")


def test_confirm_return_ineligible_400(client: TestClient):
    # Refunded order -> check_eligibility returns ok=False, reason="already_refunded".
    with SessionLocal() as s:
        s.add(User(id="u1", tenant="jd", display_name="x"))
        s.add(Order(id="OR", user_id="u1", tenant="jd", status="refunded",
                    total_cents=100, placed_at=datetime.now(timezone.utc)))
        s.commit()
    r = client.post("/agent/actions/confirm-return", json={"order_id": "OR"})
    assert r.status_code == 400


# ---------- /agent/actions/set-default-address ----------

def test_set_default_address_flips(client: TestClient):
    _seed_full()
    with SessionLocal() as s:
        company = s.query(UserAddress).filter(UserAddress.label == "公司").one()
    r = client.post("/agent/actions/set-default-address",
                    json={"user_id": "u1", "address_id": company.id})
    assert r.status_code == 200
    assert r.json()["address"]["is_default"] is True
    with SessionLocal() as s:
        rows = s.query(UserAddress).filter(UserAddress.user_id == "u1").all()
        default_count = sum(1 for a in rows if a.is_default)
        assert default_count == 1
        assert next(a for a in rows if a.label == "公司").is_default is True


def test_set_default_address_foreign_user_404(client: TestClient):
    _seed_full()
    with SessionLocal() as s:
        home = s.query(UserAddress).filter(UserAddress.label == "家").one()
    r = client.post("/agent/actions/set-default-address",
                    json={"user_id": "someoneelse", "address_id": home.id})
    assert r.status_code == 404


# ---------- /agent/actions/cancel-complaint ----------

def test_cancel_complaint_flips_status_and_audits(client: TestClient):
    with SessionLocal() as s:
        s.add(User(id="u1", tenant="jd", display_name="x"))
        s.flush()
        c = Complaint(user_id="u1", tenant="jd", order_id=None, topic="other",
                      severity="low", content_hash="abcd", status="open", escalated=False)
        s.add(c); s.commit(); s.refresh(c)
        cid = c.id

    r = client.post("/agent/actions/cancel-complaint", json={"complaint_id": cid})
    assert r.status_code == 200
    assert r.json()["status"] == "closed"
    with SessionLocal() as s:
        assert s.get(Complaint, cid).status == "closed"
        audit = s.query(AuditLog).filter(AuditLog.event_type == "complaint_cancelled").one()
        assert audit.extra["complaint_id"] == cid


def test_cancel_complaint_missing_404(client: TestClient):
    r = client.post("/agent/actions/cancel-complaint", json={"complaint_id": 99999})
    assert r.status_code == 404


# ---------- /agent/actions/verify-phone-change ----------

def test_verify_phone_bad_code_leaves_db_alone(client: TestClient):
    _seed_full()
    r = client.post("/agent/actions/verify-phone-change",
                    json={"user_id": "u1", "new_phone": "13999999999", "code": "000000"})
    assert r.status_code == 400
    with SessionLocal() as s:
        u = s.get(User, "u1")
        assert u.phone == "13800001111"  # unchanged


def test_verify_phone_good_code_updates_and_audits(client: TestClient):
    _seed_full()
    r = client.post("/agent/actions/verify-phone-change",
                    json={"user_id": "u1", "new_phone": "13999999999", "code": "123456"})
    assert r.status_code == 200
    body = r.json()
    assert body["phone_masked"] == "139****9999"
    with SessionLocal() as s:
        assert s.get(User, "u1").phone == "13999999999"
        audit = s.query(AuditLog).filter(AuditLog.event_type == "phone_changed").one()
        # raw phone NEVER stored
        assert "13800001111" not in str(audit.extra)
        assert "13999999999" not in str(audit.extra)
        assert audit.extra["old_phone_hash"]
        assert audit.extra["new_phone_hash"]


def test_verify_phone_unknown_user_404(client: TestClient):
    r = client.post("/agent/actions/verify-phone-change",
                    json={"user_id": "ghost", "new_phone": "13999999999", "code": "123456"})
    assert r.status_code == 404
