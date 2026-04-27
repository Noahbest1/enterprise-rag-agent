"""Account specialist — 查地址 / 改默认地址 / 改手机 (mock verification).

Intent routing:
    - "查地址" / "我的地址" / "list addresses" -> list_addresses
    - "改默认" / "设置为默认" (+ implicit or explicit label / id) -> set_default_address
    - "改手机" / "绑定新手机" (+ 新手机号 regex) -> request_phone_change (mock verification)
    - otherwise -> return full profile + masked phone + default address

Security:
    - Phone numbers are masked in the answer text (138****1111).
    - Phone change DOES NOT mutate the DB; it returns a pending verification
      stub so the UI can collect the SMS code. A real rollout would pair this
      with an SMS provider + Redis TTL.
"""
from __future__ import annotations

import re

from ..state import AgentState
from ..tools.accounts import (
    add_address,
    get_default_address,
    get_user_profile,
    list_addresses,
    mask_phone,
    request_phone_change,
    set_default_address,
)


_PHONE_RE = re.compile(r"\b1[3-9]\d{9}\b")
# Intent keywords -- tuned for ZH customer-service vernacular.
_INTENT_CHANGE_PHONE = (
    "改手机", "换手机", "换个手机", "换绑", "绑定新手机", "新手机号",
    "新号码", "更换手机", "更换号码",
)
_INTENT_ADDR_DEFAULT = (
    "改默认", "设为默认", "设置为默认", "默认地址", "设默认", "换默认", "改默认地址",
)
_INTENT_ADDR_ADD = (
    "加地址", "新增地址", "添加地址", "加个地址", "加收货地址", "新地址",
    "加一个", "新增一个", "新增收货地址", "添加收货地址", "新增一条", "加一条",
)
_INTENT_ADDR_LIST = (
    "查地址", "我的地址", "收货地址", "所有地址", "地址列表", "查我地址", "地址都有哪些",
)
_INTENT_PROFILE = (
    "我的账号", "账户信息", "个人信息", "我的个人", "profile", "账户资料",
)


def _default_user_for(tenant: str) -> str:
    return "jd-demo-user" if tenant == "jd" else "tb-demo-user"


def _step_from_plan(state: AgentState) -> tuple[dict | None, int]:
    plan = state.get("plan") or []
    idx = state.get("current_step", -1)
    if idx is not None and 0 <= idx < len(plan):
        return plan[idx], idx
    return None, idx


def _latest_user_content(state: AgentState, fallback: str) -> str:
    msgs = state.get("messages") or []
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
            return m["content"]
    return fallback


def _detect_intent(text: str) -> str:
    t = text.lower()
    if any(k in text for k in _INTENT_CHANGE_PHONE):
        return "change_phone"
    if any(k in text for k in _INTENT_ADDR_DEFAULT):
        return "set_default_address"
    if any(k in text for k in _INTENT_ADDR_ADD):
        return "add_address"
    if any(k in text for k in _INTENT_ADDR_LIST):
        return "list_addresses"
    if any(k in t for k in _INTENT_PROFILE) or any(k in text for k in _INTENT_PROFILE):
        return "profile"
    # Heuristic fallback: phrases with 地址 but no explicit add/default => list
    if "地址" in text:
        return "list_addresses"
    # Phone number mentioned without "change" verb is ambiguous -> profile.
    return "profile"


def _pick_default_label_from_text(text: str, addresses: list[dict]) -> int | None:
    """Look for an address label (家 / 公司 / 父母家 / etc.) in the user's text."""
    for a in addresses:
        if a["label"] and a["label"] in text:
            return a["id"]
    return None


def _format_address_line(a: dict) -> str:
    region = " ".join(x for x in (a.get("province"), a.get("city"), a.get("district")) if x)
    default_mark = " [默认]" if a.get("is_default") else ""
    return (
        f"[{a['label']}{default_mark}] {a['recipient']} · {a['phone_masked']}\n"
        f"  {region} {a['line1']}".rstrip()
    )


def _format_addresses(addresses: list[dict]) -> str:
    if not addresses:
        return "您当前没有保存的收货地址。"
    lines = ["您已保存的收货地址:"]
    for a in addresses:
        lines.append(_format_address_line(a))
    return "\n".join(lines)


def account_node(state: AgentState) -> dict:
    step, _ = _step_from_plan(state)
    if step is None:
        return {}
    step_id = step["step_id"]
    query = step.get("query") or ""
    user_utterance = _latest_user_content(state, query)

    tenant = state.get("tenant") or "jd"
    user_id = state.get("user_id") or _default_user_for(tenant)
    profile = get_user_profile(user_id)
    if profile is None:
        return {
            "step_results": {
                **(state.get("step_results") or {}),
                step_id: {
                    "agent": "account", "query": query,
                    "answer": "未找到账户资料。",
                    "citations": [], "abstain": True,
                },
            },
            "trace": [{"node": "account", "step_id": step_id, "status": "no_user"}],
        }

    intent = _detect_intent(user_utterance)
    addresses = list_addresses(user_id)

    if intent == "change_phone":
        m = _PHONE_RE.search(user_utterance)
        if not m:
            answer = (
                f"您当前绑定手机号 {mask_phone(profile['phone'])}。"
                "请提供新手机号,我们会向当前手机发送验证码以完成换绑。"
            )
            return _result(state, step_id, query, answer, intent,
                           {"profile": {**profile, "phone": mask_phone(profile['phone'])}})
        new_phone = m.group(0)
        pending = request_phone_change(user_id, new_phone)
        answer = (
            f"已发起手机号变更申请:{pending['current_phone_masked']} → "
            f"{pending['new_phone_masked']}。\n{pending['message']}"
        )
        return _result(state, step_id, query, answer, intent, pending)

    if intent == "set_default_address":
        # Prefer explicit label match (家 / 公司 / ...); else default to first address.
        target_id = _pick_default_label_from_text(user_utterance, addresses)
        if target_id is None and addresses:
            target_id = addresses[0]["id"]
        if target_id is None:
            return _result(state, step_id, query,
                           "您还没有任何收货地址,无法设置默认地址。", intent, {})
        updated = set_default_address(user_id, target_id)
        if updated is None:
            return _result(state, step_id, query, "未找到该地址。", intent, {})
        answer = f"已将「{updated['label']}」设为默认收货地址:\n{_format_address_line(updated)}"
        new_list = list_addresses(user_id)
        return _result(state, step_id, query, answer, intent, {"addresses": new_list})

    if intent == "add_address":
        answer = (
            "新增收货地址需要以下信息,请在下一轮一并提供:\n"
            "  收件人 / 手机号 / 省市区 / 详细地址 / 标签(家 or 公司)\n"
            "示例:李四 13900000000 北京 朝阳 三里屯路 1 号 (label=公司)"
        )
        return _result(state, step_id, query, answer, intent, {"addresses": addresses})

    if intent == "list_addresses":
        return _result(state, step_id, query, _format_addresses(addresses),
                       intent, {"addresses": addresses})

    # profile (default)
    default_addr = get_default_address(user_id)
    profile_masked = {**profile, "phone": mask_phone(profile["phone"])}
    lines = [
        f"您的账户资料:",
        f"  ID:{profile['id']}",
        f"  昵称:{profile['display_name']}",
        f"  绑定手机:{profile_masked['phone']}",
    ]
    if default_addr:
        lines.append(f"  默认收货地址:{default_addr['label']} — {default_addr['line1']}")
    lines.append(f"  收货地址数量:{len(addresses)}")
    return _result(state, step_id, query, "\n".join(lines), intent,
                   {"profile": profile_masked, "default_address": default_addr,
                    "address_count": len(addresses)})


def _result(state: AgentState, step_id: int, query: str, answer: str,
            intent: str, extras: dict) -> dict:
    return {
        "entities": {
            **(state.get("entities") or {}),
            "last_account_intent": intent,
        },
        "step_results": {
            **(state.get("step_results") or {}),
            step_id: {
                "agent": "account", "query": query, "answer": answer,
                "intent": intent, "citations": [], "abstain": False,
                **extras,
            },
        },
        "trace": [{"node": "account", "step_id": step_id, "intent": intent}],
    }
