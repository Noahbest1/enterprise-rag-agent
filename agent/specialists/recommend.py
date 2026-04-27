"""Recommend specialist -- "给我推荐类似的商品"."""
from __future__ import annotations

from ..state import AgentState
from ..tools.orders import get_order
from ..tools.recommend import similar_products


def _current_step(state: AgentState):
    plan = state.get("plan") or []
    idx = state.get("current_step") or 0
    if 0 <= idx < len(plan):
        return plan[idx]
    return None


def _anchor_from_state(state: AgentState, query: str) -> tuple[str, str | None]:
    """Pick the text + optional sku-to-exclude from state or query.

    If a previous step (Order / AfterSale) resolved an order, use the
    first item's title as the anchor (so "推荐类似的" after "查我 MacBook 订单"
    means "other products like MacBook").
    """
    entities = state.get("entities") or {}
    last_item_title = entities.get("last_item_title")
    if last_item_title:
        return last_item_title, None

    last_order_id = entities.get("last_order_id")
    if last_order_id:
        order = get_order(last_order_id)
        if order and order.get("items"):
            top = order["items"][0]
            return top["title"], top.get("sku")
    return query, None


def recommend_node(state: AgentState) -> dict:
    step = _current_step(state)
    if step is None:
        return {}
    query = step["query"]
    anchor_text, exclude_sku = _anchor_from_state(state, query)

    results = similar_products(anchor_text, top_k=5, exclude_sku=exclude_sku)

    if not results:
        return {
            "step_results": {
                **(state.get("step_results") or {}),
                step["step_id"]: {
                    "agent": "recommend",
                    "query": query,
                    "answer": "(没有可推荐的相似商品)",
                    "citations": [],
                    "abstain": True,
                },
            },
            "trace": [{"node": "recommend", "step_id": step["step_id"], "count": 0}],
        }

    lines = [f"基于 '{anchor_text}' 的相似推荐:"]
    for r in results:
        lines.append(f"• {r['title']}(¥{r['price_yuan']:.2f},相似度 {r['similarity']:.2f})")
    answer = "\n".join(lines)

    return {
        "step_results": {
            **(state.get("step_results") or {}),
            step["step_id"]: {
                "agent": "recommend",
                "query": query,
                "answer": answer,
                "anchor": anchor_text,
                "items": results,
                "citations": [],
                "abstain": False,
            },
        },
        "trace": [
            {"node": "recommend", "step_id": step["step_id"], "anchor": anchor_text, "count": len(results)}
        ],
    }
