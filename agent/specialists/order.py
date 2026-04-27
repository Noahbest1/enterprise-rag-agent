"""Order specialist -- DB-backed, replaces the Phase-A stub."""
from __future__ import annotations

from ..state import AgentState
from ..tools.orders import find_recent_order_with_keyword, get_order, list_user_orders


JD_DEFAULT_USER = "jd-demo-user"
TB_DEFAULT_USER = "tb-demo-user"


def _default_user_for(tenant: str) -> str:
    return JD_DEFAULT_USER if tenant == "jd" else TB_DEFAULT_USER


def _current_step(state: AgentState):
    plan = state.get("plan") or []
    idx = state.get("current_step") or 0
    if 0 <= idx < len(plan):
        return plan[idx]
    return None


def _format_order_card(o: dict) -> str:
    items = " + ".join(f"{it['title']} ×{it['qty']}" for it in o["items"])
    tracking = f"{o['carrier']} {o['tracking_no']}" if o["tracking_no"] else "(未发货)"
    return (
        f"订单 {o['id']} ({o['status']},¥{o['total_yuan']:.2f}) — {items};"
        f" 物流: {tracking}"
    )


def order_node(state: AgentState) -> dict:
    step = _current_step(state)
    if step is None:
        return {}

    tenant = state.get("tenant", "jd")
    user_id = state.get("user_id") or _default_user_for(tenant)
    query = step["query"]

    # Best-effort dispatch: if the query mentions a product keyword, do the
    # keyword search; if it mentions an explicit order id, fetch by id;
    # otherwise list recent orders.
    found_orders: list[dict] = []
    mode = "recent"

    for tok in query.split():
        if tok.upper().startswith(("JD", "TB")) and len(tok) >= 8:
            hit = get_order(tok)
            if hit:
                found_orders = [hit]
                mode = "by_id"
                break

    if not found_orders:
        for kw in ("MacBook", "iPhone", "AirPods", "空调", "兰蔻", "UNIQLO", "坚果", "松鼠", "羽绒服"):
            if kw.lower() in query.lower():
                hit = find_recent_order_with_keyword(user_id, tenant, kw)
                if hit:
                    found_orders = [hit]
                    mode = "by_keyword"
                    break

    if not found_orders:
        found_orders = list_user_orders(user_id, tenant, limit=3)
        mode = "recent"

    summary = (
        "\n".join("• " + _format_order_card(o) for o in found_orders)
        if found_orders
        else "(没有找到相关订单)"
    )

    # Bubble up entity info so downstream steps (logistics, aftersale)
    # can re-use the order_id / tracking_no without re-asking.
    entity_patch: dict = {}
    if found_orders:
        top = found_orders[0]
        entity_patch = {
            "last_order_id": top["id"],
            "last_tracking_no": top["tracking_no"],
            "last_carrier": top["carrier"],
            "last_item_title": top["items"][0]["title"] if top["items"] else None,
        }

    return {
        "step_results": {
            **(state.get("step_results") or {}),
            step["step_id"]: {
                "agent": "order",
                "query": query,
                "mode": mode,
                "orders": found_orders,
                "answer": summary,
                "citations": [],
                "abstain": not bool(found_orders),
            },
        },
        "entities": {**(state.get("entities") or {}), **entity_patch},
        "trace": [
            {
                "node": "order",
                "step_id": step["step_id"],
                "mode": mode,
                "count": len(found_orders),
            }
        ],
    }
