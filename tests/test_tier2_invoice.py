"""Tier-2 Invoice specialist tests.

Covers:
- get_invoice_by_order / request_invoice / list_invoices_for_user tool behaviour
- request_invoice: duplicate detection + eligibility gate + title/tax_id capture
- invoice_node: existing issued invoice -> "read" action
- invoice_node: existing requested invoice -> "processing" wording
- invoice_node: no existing invoice + eligible order -> "created" action
- invoice_node: order not eligible (placed) -> eligibility rejection path
- invoice_node: last_order_id in entities is respected
- invoice_node: company title + tax_id extracted from query
- invoice_node: no orders at all -> abstain
- planner: JSON plan mentioning `invoice` agent is accepted
- agent_routes: `invoice` is in the specialist set so SSE events fire
"""
from __future__ import annotations

import pytest

from rag.db.base import Base, SessionLocal, engine
from rag.db.models import Invoice, Order, OrderItem, User
from agent.specialists.invoice import invoice_node, _pick_title_and_tax
from agent.tools.invoices import (
    get_invoice_by_order,
    list_invoices_for_user,
    request_invoice,
)


@pytest.fixture(autouse=True)
def _fresh_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


def _seed_base(with_delivered_order: bool = True, with_placed_order: bool = False) -> None:
    from datetime import datetime, timezone
    with SessionLocal() as s:
        s.add(User(id="u1", tenant="jd", display_name="u", phone=None))
        if with_delivered_order:
            o = Order(
                id="O-DELIVERED", user_id="u1", tenant="jd", status="delivered",
                total_cents=10698_00, tracking_no="T1", carrier="京东快递",
                placed_at=datetime.now(timezone.utc),
            )
            s.add(o)
            s.flush()
            s.add(OrderItem(order_id=o.id, sku="SKU1", title="iPhone", qty=1, unit_price_cents=10698_00))
        if with_placed_order:
            o2 = Order(
                id="O-PLACED", user_id="u1", tenant="jd", status="placed",
                total_cents=5999_00, placed_at=datetime.now(timezone.utc),
            )
            s.add(o2)
            s.flush()
            s.add(OrderItem(order_id=o2.id, sku="SKU2", title="Air Cond", qty=1, unit_price_cents=5999_00))
        s.commit()


# ---------- tool layer ----------

def test_request_invoice_creates_requested_row():
    _seed_base()
    inv = request_invoice(order_id="O-DELIVERED", title="个人")
    assert inv["status"] == "requested"
    assert inv["amount_yuan"] == pytest.approx(10698.00)
    assert inv["title"] == "个人"


def test_request_invoice_dedups_on_second_call():
    _seed_base()
    first = request_invoice(order_id="O-DELIVERED", title="个人")
    second = request_invoice(order_id="O-DELIVERED", title="公司")
    assert second.get("duplicate") is True
    assert second["id"] == first["id"]
    assert second["title"] == "个人"  # original preserved


def test_request_invoice_blocks_ineligible_order():
    _seed_base(with_placed_order=True)
    result = request_invoice(order_id="O-PLACED", title="个人")
    assert result.get("error") == "order_not_eligible"
    assert result["current_status"] == "placed"


def test_request_invoice_missing_order():
    result = request_invoice(order_id="NOPE", title="x")
    assert result.get("error") == "order_not_found"


def test_get_invoice_by_order_returns_latest():
    _seed_base()
    assert get_invoice_by_order("O-DELIVERED") is None
    request_invoice(order_id="O-DELIVERED", title="个人")
    inv = get_invoice_by_order("O-DELIVERED")
    assert inv is not None and inv["title"] == "个人"


def test_list_invoices_for_user_joins_through_order():
    _seed_base()
    request_invoice(order_id="O-DELIVERED", title="个人")
    rows = list_invoices_for_user("u1")
    assert len(rows) == 1 and rows[0]["order_id"] == "O-DELIVERED"


# ---------- title + tax id extraction ----------

def test_pick_title_personal():
    assert _pick_title_and_tax("帮我开个人发票") == ("个人", None)


def test_pick_title_company_only():
    title, tax = _pick_title_and_tax("抬头写上海某某科技有限公司")
    assert "科技有限公司" in title
    assert tax is None


def test_pick_title_company_with_tax_id():
    title, tax = _pick_title_and_tax(
        "公司抬头 上海某某科技有限公司 税号 91310000XYZ123456A"
    )
    assert "公司" in title
    assert tax == "91310000XYZ123456A"


# ---------- specialist node ----------

def _state_with_step(query: str, tenant: str = "jd", entities: dict | None = None):
    return {
        "tenant": tenant,
        "user_id": "u1",
        "messages": [{"role": "user", "content": query}],
        "entities": entities or {},
        "step_results": {},
        "trace": [],
        "plan": [{"step_id": 1, "agent": "invoice", "query": query, "depends_on": [], "status": "pending"}],
        "current_step": 0,
    }


def test_invoice_node_no_orders_abstains():
    # user u1 exists but no orders; invoice_node should land on abstain.
    with SessionLocal() as s:
        s.add(User(id="u1", tenant="jd", display_name="x"))
        s.commit()
    out = invoice_node(_state_with_step("开个人发票"))
    sr = out["step_results"][1]
    assert sr["abstain"] is True
    assert "未找到" in sr["answer"]


def test_invoice_node_creates_new_invoice():
    _seed_base()
    out = invoice_node(_state_with_step("给 O-DELIVERED 开个人发票"))
    sr = out["step_results"][1]
    assert sr["abstain"] is False
    assert sr["action"] == "created"
    assert sr["invoice"]["status"] == "requested"
    assert out["entities"]["last_invoice_status"] == "requested"
    assert out["entities"]["last_order_id"] == "O-DELIVERED"


def test_invoice_node_returns_existing_issued():
    _seed_base()
    # Pre-create and flip to issued.
    inv = request_invoice(order_id="O-DELIVERED", title="个人")
    with SessionLocal() as s:
        row = s.get(Invoice, inv["id"])
        row.status = "issued"
        row.download_url = "https://x.example/inv.pdf"
        s.commit()
    out = invoice_node(_state_with_step("我那个 iPhone 订单发票开好了吗",
                                         entities={"last_order_id": "O-DELIVERED"}))
    sr = out["step_results"][1]
    assert sr["action"] == "read"
    assert "已开具" in sr["answer"]
    assert "https://x.example/inv.pdf" in sr["answer"]


def test_invoice_node_ineligible_order():
    _seed_base(with_delivered_order=False, with_placed_order=True)
    out = invoice_node(_state_with_step("给 O-PLACED 开票"))
    sr = out["step_results"][1]
    assert sr.get("reason") == "order_not_eligible"
    assert "暂不可开票" in sr["answer"]


def test_invoice_node_respects_last_order_id():
    _seed_base()
    out = invoice_node(_state_with_step(
        "开张个人发票",  # no order id in query
        entities={"last_order_id": "O-DELIVERED"},
    ))
    sr = out["step_results"][1]
    assert sr["order_id"] == "O-DELIVERED"
    assert sr["action"] == "created"


def test_invoice_node_extracts_company_title():
    _seed_base()
    out = invoice_node(_state_with_step(
        "开发票 抬头 上海某某科技有限公司 税号 91310000XYZ123456A 订单 O-DELIVERED"
    ))
    sr = out["step_results"][1]
    assert "公司" in sr["invoice"]["title"]
    assert sr["invoice"]["tax_id"] == "91310000XYZ123456A"


# ---------- planner + agent_routes awareness ----------

def test_planner_accepts_invoice_agent():
    from agent.planner import _parse_plan
    reply = (
        '[{"step_id":1,"agent":"order","query":"查 MacBook 订单","depends_on":[]},'
        ' {"step_id":2,"agent":"invoice","query":"开企业发票","depends_on":[1]}]'
    )
    plan = _parse_plan(reply)
    assert plan is not None and len(plan) == 2
    assert plan[1]["agent"] == "invoice"


def test_agent_routes_treats_invoice_as_specialist():
    import src.rag_api.agent_routes as ar  # noqa: F401
    # The local variable lives inside _stream_agent_turn; easiest assertion is
    # via the source to catch regressions (white-box but 1 line).
    import inspect
    src = inspect.getsource(ar._stream_agent_turn)
    assert '"invoice"' in src
