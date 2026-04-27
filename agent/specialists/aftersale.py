"""AfterSale specialist -- dry-run eligibility check.

This specialist answers the QUESTION "can I return this order?" and never
writes to the return_requests table. The UI button at
`/agent/actions/confirm-return` is the sole creation path.

Steps executed inside this single node:
  1. Resolve target order (from state.entities if the Order specialist
     already ran, else from the query, else pick the most recent order).
  2. Check eligibility (7-day window + status gates).
  3. Surface any existing pending request (so a returning user sees the
     already-open ticket instead of the confirm button) -- but do NOT
     create a new one.
  4. Expose last_order_id / last_aftersale_kind in state.entities so
     follow-up turns can say "那就退吧".
"""
from __future__ import annotations

import re

from ..state import AgentState
from ..tools.orders import find_recent_order_with_keyword, list_user_orders
from ..tools.returns import check_eligibility, list_requests_for_order


_ORDER_ID_RE = re.compile(r"\b(?:JD|TB)\d{8,16}\b", re.IGNORECASE)


def _pick_kind(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ("换货", "exchange", "换一件")):
        return "exchange"
    if any(w in q for w in ("价保", "保价", "price protect", "差价")):
        return "price_protect"
    if any(w in q for w in ("退货", "return", "退回", "退掉")):
        return "return"
    if any(w in q for w in ("退款", "refund", "退钱")):
        return "refund"
    return "return"  # safest default


def _default_user_for(tenant: str) -> str:
    return "jd-demo-user" if tenant == "jd" else "tb-demo-user"


def _find_target_order(state: AgentState, query: str) -> dict | None:
    entities = state.get("entities") or {}
    if entities.get("last_order_id"):
        # trust the Order specialist's resolution
        from ..tools.orders import get_order
        hit = get_order(entities["last_order_id"])
        if hit:
            return hit

    m = _ORDER_ID_RE.search(query)
    if m:
        from ..tools.orders import get_order
        hit = get_order(m.group(0).upper())
        if hit:
            return hit

    tenant = state.get("tenant", "jd")
    user_id = state.get("user_id") or _default_user_for(tenant)
    for kw in ("MacBook", "iPhone", "AirPods", "兰蔻", "UNIQLO", "坚果", "松鼠", "羽绒服", "空调"):
        if kw.lower() in query.lower():
            hit = find_recent_order_with_keyword(user_id, tenant, kw)
            if hit:
                return hit
    recent = list_user_orders(user_id, tenant, limit=1)
    return recent[0] if recent else None


def _current_step(state: AgentState):
    plan = state.get("plan") or []
    idx = state.get("current_step") or 0
    if 0 <= idx < len(plan):
        return plan[idx]
    return None


def _format_answer(order: dict, elig: dict, kind: str, request: dict | None) -> str:
    items = " + ".join(it["title"] for it in (order.get("items") or []))
    lines: list[str] = [
        f"订单 {order['id']}({items}),金额 ¥{order['total_yuan']:.2f}"
    ]
    if elig.get("ok"):
        days_left = elig["days_left_in_window"]
        lines.append(f"7 天无理由窗口还剩 {days_left} 天,可申请{kind}。")
        if request and request.get("status") == "pending":
            rid = request.get("request_id")
            refund = request.get("refund_cents") or 0
            lines.append(
                f"当前已存在待处理 {kind} 申请(单号 R-{rid}),状态:pending"
                + (f",预计退款 ¥{refund/100:.2f}" if refund else "")
                + ";如需取消请在卡片上操作。"
            )
        else:
            lines.append("您确认后我们即可正式提交申请。")
    else:
        reason = elig.get("reason")
        if reason == "out_of_no_reason_window":
            lines.append("已超过 7 天无理由窗口,无法直接办理无理由退款;若商品存在质量问题可走质量问题售后流程。")
        elif reason == "already_refunded":
            lines.append("该订单已完成退款,无需重复申请。")
        elif reason == "order_cancelled":
            lines.append("该订单已取消,无法再次发起退款。")
        else:
            lines.append(f"当前订单不符合退款条件(原因:{reason})。")
    return "\n".join(lines)


def aftersale_node(state: AgentState) -> dict:
    step = _current_step(state)
    if step is None:
        return {}

    query = step["query"]
    kind = _pick_kind(query)
    order = _find_target_order(state, query)

    if order is None:
        return {
            "step_results": {
                **(state.get("step_results") or {}),
                step["step_id"]: {
                    "agent": "aftersale",
                    "query": query,
                    "answer": "(没有找到可办理售后的订单,请提供订单号或商品名称)",
                    "citations": [],
                    "abstain": True,
                },
            },
            "trace": [{"node": "aftersale", "step_id": step["step_id"], "status": "no_order"}],
        }

    elig = check_eligibility(order["id"])

    # Dry-run only: surface an existing pending request if any, but never
    # create one. The UI button at /agent/actions/confirm-return is the
    # sole create path.
    request = None
    if elig.get("ok") and kind in ("refund", "return", "exchange"):
        existing = list_requests_for_order(order["id"])
        pending = next((r for r in existing if r["status"] == "pending"), None)
        if pending is not None:
            request = pending

    answer = _format_answer(order, elig, kind, request)

    entity_patch = {
        "last_order_id": order["id"],
        "last_aftersale_kind": kind,
    }
    if request and request.get("request_id"):
        entity_patch["last_return_request_id"] = request["request_id"]

    return {
        "step_results": {
            **(state.get("step_results") or {}),
            step["step_id"]: {
                "agent": "aftersale",
                "query": query,
                "answer": answer,
                "kind": kind,
                "order_id": order["id"],
                "eligibility": elig,
                "request": request,
                "citations": [],
                "abstain": not elig.get("ok", False),
            },
        },
        "entities": {**(state.get("entities") or {}), **entity_patch},
        "trace": [
            {
                "node": "aftersale",
                "step_id": step["step_id"],
                "kind": kind,
                "order_id": order["id"],
                "eligible": elig.get("ok", False),
                "existing_request_id": request.get("request_id") if request else None,
            }
        ],
    }
