"""Tier-2 Account specialist tests.

Covers:
- mask_phone: 138****1111 pattern + edge cases
- list_addresses / get_default_address / add_address / set_default_address
- request_phone_change: DOES NOT mutate DB, returns pending_verification
- account_node intent routing: profile / list / set_default / add / change_phone
- change_phone: with + without a new phone number in the utterance
- set_default_address: picks by label ("设为默认 公司")
- phone masking bleeds into the answer text (never raw phone exposure)
- planner + agent_routes awareness of "account" agent
"""
from __future__ import annotations

import pytest

from rag.db.base import Base, SessionLocal, engine
from rag.db.models import User, UserAddress
from agent.specialists.account import account_node
from agent.tools.accounts import (
    add_address,
    get_default_address,
    get_user_profile,
    list_addresses,
    mask_phone,
    request_phone_change,
    set_default_address,
)


@pytest.fixture(autouse=True)
def _fresh_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


def _seed():
    with SessionLocal() as s:
        s.add(User(id="u1", tenant="jd", display_name="demo", phone="13800001111"))
        s.commit()
    add_address(
        user_id="u1", label="家", recipient="张三", phone="13800001111",
        line1="朝阳门外大街 1 号 5 栋 302", province="北京市",
        make_default=True,
    )
    add_address(
        user_id="u1", label="公司", recipient="张三", phone="13800001111",
        line1="中关村大街 27 号 12 层", province="北京市",
        make_default=False,
    )


def _state(query: str, *, user_utter: str | None = None,
           entities: dict | None = None, tenant: str = "jd"):
    msgs = [{"role": "user", "content": user_utter if user_utter is not None else query}]
    return {
        "tenant": tenant, "user_id": "u1",
        "messages": msgs,
        "entities": entities or {},
        "step_results": {},
        "trace": [],
        "plan": [{"step_id": 1, "agent": "account", "query": query, "depends_on": [], "status": "pending"}],
        "current_step": 0,
    }


# ---------- mask + helpers ----------

def test_mask_phone_basic():
    assert mask_phone("13800001111") == "138****1111"


def test_mask_phone_short():
    assert mask_phone("123") == "****"


def test_mask_phone_none():
    assert mask_phone(None) is None
    assert mask_phone("") == ""


# ---------- tool CRUD ----------

def test_add_and_list_addresses():
    _seed()
    rows = list_addresses("u1")
    assert len(rows) == 2
    # default-first ordering
    assert rows[0]["is_default"] is True
    # phone_masked never carries raw phone
    assert rows[0]["phone_masked"] == "138****1111"


def test_add_new_address_respects_make_default_flag():
    _seed()
    new = add_address(
        user_id="u1", label="临时",
        recipient="李四", phone="13900000000",
        line1="机场路 1 号", make_default=True,
    )
    assert new["is_default"] is True
    rows = list_addresses("u1")
    default_count = sum(1 for a in rows if a["is_default"])
    assert default_count == 1
    assert rows[0]["id"] == new["id"]


def test_set_default_address_flips_flags():
    _seed()
    addrs = list_addresses("u1")
    company = next(a for a in addrs if a["label"] == "公司")
    updated = set_default_address("u1", company["id"])
    assert updated["is_default"] is True
    # home address no longer default
    rows = list_addresses("u1")
    home = next(a for a in rows if a["label"] == "家")
    assert home["is_default"] is False


def test_set_default_foreign_user_rejected():
    _seed()
    with SessionLocal() as s:
        s.add(User(id="u2", tenant="jd", display_name="other"))
        s.commit()
    addrs = list_addresses("u1")
    # u2 can't flip u1's default
    out = set_default_address("u2", addrs[0]["id"])
    assert out is None


def test_get_default_address():
    _seed()
    a = get_default_address("u1")
    assert a is not None
    assert a["is_default"] is True


def test_request_phone_change_never_mutates():
    _seed()
    before = get_user_profile("u1")["phone"]
    out = request_phone_change("u1", "13999999999")
    assert out["status"] == "pending_verification"
    assert out["current_phone_masked"] == "138****1111"
    assert out["new_phone_masked"] == "139****9999"
    after = get_user_profile("u1")["phone"]
    assert after == before  # raw phone unchanged in DB


def test_request_phone_change_unknown_user():
    assert request_phone_change("ghost", "13800000000") == {"error": "user_not_found"}


# ---------- account_node ----------

def test_account_node_profile_default():
    _seed()
    out = account_node(_state("告诉我我的账户信息"))
    sr = out["step_results"][1]
    assert sr["intent"] == "profile"
    assert "138****1111" in sr["answer"]
    assert "13800001111" not in sr["answer"]  # raw phone never exposed


def test_account_node_lists_addresses_on_address_query():
    _seed()
    out = account_node(_state("查一下我保存的收货地址"))
    sr = out["step_results"][1]
    assert sr["intent"] == "list_addresses"
    assert "朝阳门外大街" in sr["answer"]
    assert "中关村大街" in sr["answer"]
    # Phone in answer is masked
    assert "138****1111" in sr["answer"]


def test_account_node_set_default_by_label():
    _seed()
    out = account_node(_state("把「公司」设为默认地址"))
    sr = out["step_results"][1]
    assert sr["intent"] == "set_default_address"
    assert "已将" in sr["answer"] and "设为默认" in sr["answer"]
    # verify DB reflects the change
    rows = list_addresses("u1")
    company = next(a for a in rows if a["label"] == "公司")
    assert company["is_default"] is True


def test_account_node_change_phone_without_number_prompts_for_one():
    _seed()
    out = account_node(_state("我想换绑手机"))
    sr = out["step_results"][1]
    assert sr["intent"] == "change_phone"
    assert "请提供新手机号" in sr["answer"] or "请提供新" in sr["answer"]
    assert "138****1111" in sr["answer"]  # current phone masked


def test_account_node_change_phone_with_number_triggers_verification():
    _seed()
    out = account_node(_state("换绑手机,新号码是 13999999999"))
    sr = out["step_results"][1]
    assert sr["intent"] == "change_phone"
    assert sr["status"] == "pending_verification"
    assert "139****9999" in sr["answer"]
    # raw new number NOT shown
    assert "13999999999" not in sr["answer"]


def test_account_node_add_address_prompts_for_fields():
    _seed()
    out = account_node(_state("帮我加一个新收货地址"))
    sr = out["step_results"][1]
    assert sr["intent"] == "add_address"
    assert "收件人" in sr["answer"]
    assert "详细地址" in sr["answer"]


def test_account_node_no_user_returns_abstain():
    out = account_node(_state("查地址"))
    sr = out["step_results"][1]
    assert sr["abstain"] is True
    assert "未找到" in sr["answer"]


# ---------- planner + agent_routes ----------

def test_planner_accepts_account_agent():
    from agent.planner import _parse_plan
    reply = '[{"step_id":1,"agent":"account","query":"查用户默认地址","depends_on":[]}]'
    plan = _parse_plan(reply)
    assert plan is not None and plan[0]["agent"] == "account"


def test_agent_routes_treats_account_as_specialist():
    import src.rag_api.agent_routes as ar
    import inspect
    src = inspect.getsource(ar._stream_agent_turn)
    assert '"account"' in src
