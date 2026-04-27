"""OrderAgent v2 -- uses native function calling.

Kept as a sibling to ``order.py`` so both styles are demo-able:
  * ``order_node``        : imperative; Python decides what to call.
  * ``order_agent_v2``    : LLM decides. Same inputs/outputs.

Flip between them by editing graph.py. Useful when interviewers want to
see OpenAI-compatible ``tool_calls`` wiring.
"""
from __future__ import annotations

from ..persona import get_persona
from ..state import AgentState
from ..tool_calling import Tool, run_tool_calling_loop
from ..tools.orders import find_recent_order_with_keyword, get_order, list_user_orders, track_package


def _order_tools(user_id: str, tenant: str) -> list[Tool]:
    """Create tool bindings that close over the current user/tenant.

    We inject user_id + tenant rather than trusting the LLM to repeat them,
    which prevents cross-tenant data leaks by construction.
    """
    return [
        Tool(
            name="get_order",
            description="Fetch a specific order by its ID. Returns null if not found.",
            parameters={
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "Order ID like JD20260418123"}},
                "required": ["order_id"],
            },
            fn=lambda order_id: get_order(order_id),
        ),
        Tool(
            name="list_recent_orders",
            description="List the current user's most recent orders. Use when the user asks about 'my orders' or 'recent orders' without naming a specific order.",
            parameters={
                "type": "object",
                "properties": {"limit": {"type": "integer", "default": 3, "minimum": 1, "maximum": 20}},
                "required": [],
            },
            fn=lambda limit=3: list_user_orders(user_id, tenant, limit=limit),
        ),
        Tool(
            name="find_order_by_product",
            description="Find the user's most recent order that contains a product whose title matches a keyword (e.g. 'MacBook', 'iPhone').",
            parameters={
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
            },
            fn=lambda keyword: find_recent_order_with_keyword(user_id, tenant, keyword),
        ),
        Tool(
            name="track_package",
            description="Look up the shipping timeline for a tracking number.",
            parameters={
                "type": "object",
                "properties": {"tracking_no": {"type": "string"}},
                "required": ["tracking_no"],
            },
            fn=lambda tracking_no: track_package(tracking_no),
        ),
    ]


def _current_step(state: AgentState):
    plan = state.get("plan") or []
    idx = state.get("current_step") or 0
    if 0 <= idx < len(plan):
        return plan[idx]
    return None


SYSTEM = """你是订单查询助手。根据用户问题选择合适的工具并给出简洁回答。

规则:
- 优先使用工具获取信息,不要凭空推断订单号 / 快递单号。
- 找到订单后,直接把结果摘要给用户(订单号、商品、金额、状态)。
- 如果需要物流,先拿到订单的 tracking_no,再调 track_package。
- 用与用户相同的语言(中文 / 英文)回答。
- 3-5 句为宜,不啰嗦。
"""


def order_agent_v2(state: AgentState) -> dict:
    step = _current_step(state)
    if step is None:
        return {}

    tenant = state.get("tenant", "jd")
    user_id = state.get("user_id") or ("jd-demo-user" if tenant == "jd" else "tb-demo-user")
    persona = get_persona(tenant)

    tools = _order_tools(user_id, tenant)

    try:
        result = run_tool_calling_loop(
            system_prompt=SYSTEM + "\n\n" + persona["system_prompt"],
            user_prompt=step["query"],
            tools=tools,
            max_steps=4,
            task="intent",
        )
    except Exception as e:  # noqa: BLE001
        return {
            "step_results": {
                **(state.get("step_results") or {}),
                step["step_id"]: {
                    "agent": "order_v2",
                    "query": step["query"],
                    "answer": f"(工具调用失败:{e})",
                    "citations": [],
                    "abstain": True,
                },
            },
            "trace": [{"node": "order_v2", "step_id": step["step_id"], "error": str(e)}],
        }

    # Pull first resolved order out of the tool trace for entity carry-forward.
    resolved_order = None
    for tc in result.tool_calls:
        if tc.name in ("get_order", "find_order_by_product") and isinstance(tc.result, dict):
            resolved_order = tc.result
            break

    entity_patch: dict = {}
    if resolved_order:
        entity_patch["last_order_id"] = resolved_order["id"]
        entity_patch["last_tracking_no"] = resolved_order.get("tracking_no")
        entity_patch["last_carrier"] = resolved_order.get("carrier")
        items = resolved_order.get("items") or []
        if items:
            entity_patch["last_item_title"] = items[0]["title"]

    return {
        "step_results": {
            **(state.get("step_results") or {}),
            step["step_id"]: {
                "agent": "order_v2",
                "query": step["query"],
                "answer": result.final_text,
                "tool_calls": [
                    {"name": t.name, "args": t.args, "error": t.error}
                    for t in result.tool_calls
                ],
                "citations": [],
                "abstain": not bool(resolved_order),
            },
        },
        "entities": {**(state.get("entities") or {}), **entity_patch},
        "trace": [
            {
                "node": "order_v2",
                "step_id": step["step_id"],
                "tool_call_count": len(result.tool_calls),
                "final_text_preview": result.final_text[:100],
            }
        ],
    }
