"""Invoice specialist — "给我开发票" / "发票下载" / "开个企业抬头".

Resolution order for the target order:
    1. Order id parsed from the current step's query
    2. state.entities["last_order_id"] (Order specialist already fired)
    3. Most recent delivered order for the tenant's demo user

Emits ``last_invoice_id`` + ``last_invoice_status`` into state.entities so
the follow-up turn ("那发了没?") can reference it without re-resolving.
"""
from __future__ import annotations

import re

from ..state import AgentState
from ..tools.invoices import get_invoice_by_order, request_invoice
from ..tools.orders import get_order, list_user_orders


_ORDER_ID_RE = re.compile(r"\b(?:JD|TB)\d{8,16}\b", re.IGNORECASE)
_TAX_ID_RE = re.compile(r"\b[A-Z0-9]{15,18}\b")


def _default_user_for(tenant: str) -> str:
    return "jd-demo-user" if tenant == "jd" else "tb-demo-user"


def _pick_title_and_tax(query: str) -> tuple[str, str | None]:
    """Cheap heuristics. Production: let the planner extract and pass structured."""
    q = query.strip()
    # Tax id = 15/18-char alnum block (统一社会信用代码)
    tax_m = _TAX_ID_RE.search(q)
    tax_id = tax_m.group(0) if tax_m else None
    # Company title heuristic: a Chinese-string before "公司" / "集团" / "中心"
    title_m = re.search(r"([一-龥A-Za-z0-9（）() ]{2,40}(?:公司|集团|研究院|中心))", q)
    if title_m:
        return title_m.group(1).strip(), tax_id
    if any(kw in q for kw in ("个人", "personal", "个人抬头")):
        return "个人", None
    # Default: personal (most consumer-side flows)
    return "个人", None


def _find_target_order(state: AgentState, query: str) -> dict | None:
    m = _ORDER_ID_RE.search(query)
    if m:
        hit = get_order(m.group(0))
        if hit:
            return hit
    entities = state.get("entities") or {}
    if entities.get("last_order_id"):
        hit = get_order(entities["last_order_id"])
        if hit:
            return hit
    tenant = state.get("tenant") or "jd"
    user_id = state.get("user_id") or _default_user_for(tenant)
    orders = list_user_orders(user_id, tenant, limit=5)
    # Prefer a delivered or shipped order for invoicing.
    for o in orders:
        if o.get("status") in ("delivered", "shipped"):
            return o
    return orders[0] if orders else None


def _step_from_plan(state: AgentState) -> tuple[dict | None, int]:
    plan = state.get("plan") or []
    idx = state.get("current_step", -1)
    if idx is not None and 0 <= idx < len(plan):
        return plan[idx], idx
    return None, idx


def invoice_node(state: AgentState) -> dict:
    step, _ = _step_from_plan(state)
    if step is None:
        return {}
    query = step.get("query") or ""
    step_id = step["step_id"]

    order = _find_target_order(state, query)
    if order is None:
        return {
            "step_results": {
                **(state.get("step_results") or {}),
                step_id: {
                    "agent": "invoice",
                    "query": query,
                    "answer": "未找到可开票的订单。",
                    "citations": [],
                    "abstain": True,
                },
            },
            "trace": [{"node": "invoice", "step_id": step_id, "status": "no_order"}],
        }

    # Existing invoice?
    existing = get_invoice_by_order(order["id"])
    if existing is not None:
        if existing["status"] == "issued":
            text = (
                f"订单 {order['id']} 的发票已开具。"
                f"金额 ¥{existing['amount_yuan']:.2f},抬头「{existing['title']}」,"
                f"类型:{'电子发票' if existing['invoice_type'] == 'electronic' else '纸质'}。"
                f"下载链接:{existing['download_url'] or '(生成中)'}"
            )
        elif existing["status"] == "requested":
            text = (
                f"订单 {order['id']} 的发票已在处理中(抬头「{existing['title']}」,"
                f"金额 ¥{existing['amount_yuan']:.2f}),一般在 1-2 个工作日内开具。"
            )
        else:
            text = f"订单 {order['id']} 的发票状态:{existing['status']}。"

        return {
            "entities": {
                **(state.get("entities") or {}),
                "last_order_id": order["id"],
                "last_invoice_id": existing["id"],
                "last_invoice_status": existing["status"],
            },
            "step_results": {
                **(state.get("step_results") or {}),
                step_id: {
                    "agent": "invoice",
                    "query": query,
                    "answer": text,
                    "order_id": order["id"],
                    "invoice": existing,
                    "action": "read",
                    "citations": [],
                    "abstain": False,
                },
            },
            "trace": [{"node": "invoice", "step_id": step_id, "action": "read", "invoice_id": existing["id"]}],
        }

    # No invoice yet -> try to create one.
    title, tax_id = _pick_title_and_tax(query)
    result = request_invoice(order_id=order["id"], title=title, tax_id=tax_id)
    if result.get("error") == "order_not_eligible":
        text = (
            f"订单 {order['id']} 当前状态为 {result.get('current_status')},"
            f"暂不可开票(一般需要已发货或已签收)。"
        )
        return {
            "step_results": {
                **(state.get("step_results") or {}),
                step_id: {
                    "agent": "invoice", "query": query, "answer": text,
                    "order_id": order["id"], "reason": "order_not_eligible",
                    "citations": [], "abstain": False,
                },
            },
            "trace": [{"node": "invoice", "step_id": step_id, "action": "reject_eligibility"}],
        }

    text = (
        f"已为订单 {order['id']} 提交开票申请(抬头「{title}」"
        f"{',税号 ' + tax_id if tax_id else ''},金额 ¥{result['amount_yuan']:.2f})。"
        f"电子发票通常 1-2 个工作日内出具,届时可在「我的发票」下载。"
    )
    return {
        "entities": {
            **(state.get("entities") or {}),
            "last_order_id": order["id"],
            "last_invoice_id": result["id"],
            "last_invoice_status": result["status"],
        },
        "step_results": {
            **(state.get("step_results") or {}),
            step_id: {
                "agent": "invoice", "query": query, "answer": text,
                "order_id": order["id"], "invoice": result, "action": "created",
                "citations": [], "abstain": False,
            },
        },
        "trace": [{"node": "invoice", "step_id": step_id, "action": "created", "invoice_id": result["id"]}],
    }
