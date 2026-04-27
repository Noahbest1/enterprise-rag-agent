"""Plan-and-Execute planner.

Given the user's latest message + conversation history + known entities,
the LLM emits a JSON execution plan. Each step names one specialist, a
sub-query for that specialist, and dependencies on earlier steps.

Fallback: if JSON parsing fails we default to a single product_qa step --
this keeps the agent alive even when the LLM misbehaves.
"""
from __future__ import annotations

import json
import re

from rag.answer.llm_client import LLMError, chat_once

from .state import AgentState, PlanStep


SYSTEM = """You are the Planner for an e-commerce customer-service agent.

Given the user's latest question and conversation history, decompose it into an execution plan of 1-4 steps. Each step is handled by ONE specialist.

Specialists you can call:
- "product_qa":   商品信息(价格/规格/库存/型号) -- queries the product knowledge base
- "policy_qa":    政策/规则(退换货/运费/会员/售后) -- queries the policy knowledge base
- "order":        订单查询(my orders, order status by id) -- mock DB
- "logistics":    物流查询(tracking number, delivery status) -- mock API
- "aftersale":    售后办理(申请退货/换货/价保) -- multi-step workflow
- "recommend":    商品推荐(similar/alternatives) -- vector similarity
- "invoice":      开发票 / 发票状态 / 发票下载 / 企业抬头发票 -- DB-backed invoice lookup & request
- "complaint":    投诉 / 抱怨 / 不满意 / 要曝光 / 12315 -- creates a complaint ticket, auto-escalates on strong anger signals
- "account":      账户 / 收货地址 / 改默认地址 / 换绑手机 -- profile + address management (phone change goes through mock SMS verification)

Output ONLY a JSON array of step objects, nothing else. Each step has:
- step_id:    1-based integer
- agent:      one of the specialist names above
- query:      the concrete sub-question this step should answer
- depends_on: list of step_ids whose results this step needs

Examples:

User: "iPhone 16 Pro 多少钱"
Output:
[{"step_id":1,"agent":"product_qa","query":"iPhone 16 Pro 价格和配置","depends_on":[]}]

User: "我上周买的 MacBook 还没到,怎么查物流?"
Output:
[
 {"step_id":1,"agent":"order","query":"查找最近的 MacBook 订单","depends_on":[]},
 {"step_id":2,"agent":"logistics","query":"根据订单的快递单号查询物流进度","depends_on":[1]}
]

User: "买了一支口红想退"
Output:
[
 {"step_id":1,"agent":"policy_qa","query":"口红/化妆品 7天无理由退货是否适用,需要满足哪些条件","depends_on":[]}
]

User: "你好 你是什么客服"
Output:
[{"step_id":1,"agent":"policy_qa","query":"介绍我们的客服身份、服务范围、能做什么","depends_on":[]}]

User: "你好"
Output:
[{"step_id":1,"agent":"policy_qa","query":"问候语,简介可帮助的事情","depends_on":[]}]

User: "你能帮我做什么"
Output:
[{"step_id":1,"agent":"policy_qa","query":"列出本客服能办理的事项","depends_on":[]}]

Rules:
- Be minimal. Don't add speculative steps.
- If the user's question is pure fact (no order needed), use ONE step (product_qa or policy_qa).
- **Greetings, identity questions, capability questions, or generic small talk
  default to ONE policy_qa step.** Do NOT classify as "complaint" just because
  the user mentions the word "客服" / "service" / "agent" — only classify as
  "complaint" when there is concrete dissatisfaction or threat language
  (差评 / 投诉 / 12315 / 曝光 / 起诉 / 不满 / 退不了 / 等了X天 etc.).
- depends_on is used by Coordinator to sequence steps; parallelisable steps have empty depends_on.
- Output ONLY the JSON array. No explanation, no markdown.
"""


_JSON_ARRAY_RE = re.compile(r"\[\s*\{.*\}\s*\]", re.DOTALL)


def _parse_plan(reply: str) -> list[PlanStep] | None:
    m = _JSON_ARRAY_RE.search(reply or "")
    if not m:
        return None
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, list) or not raw:
        return None
    valid_agents = {"product_qa", "policy_qa", "order", "logistics", "aftersale", "recommend", "invoice", "complaint", "account"}
    plan: list[PlanStep] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        agent = item.get("agent")
        if agent not in valid_agents:
            continue
        plan.append(
            PlanStep(
                step_id=int(item.get("step_id") or len(plan) + 1),
                agent=agent,
                query=str(item.get("query") or ""),
                depends_on=[int(x) for x in (item.get("depends_on") or [])],
                status="pending",
            )
        )
    return plan or None


def _fallback_plan(query: str) -> list[PlanStep]:
    """One-step product_qa fallback if LLM output can't be parsed."""
    return [PlanStep(step_id=1, agent="product_qa", query=query, depends_on=[], status="pending")]


def _format_history(messages: list[dict], max_turns: int = 6) -> str:
    tail = messages[-max_turns:]
    lines = []
    for m in tail:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        tag = "User" if role == "user" else "Assistant" if role == "assistant" else "System"
        if len(content) > 300:
            content = content[:300].rstrip() + " ..."
        lines.append(f"{tag}: {content}")
    return "\n".join(lines)


def planner_node(state: AgentState) -> dict:
    """LangGraph node: read user's latest message and produce state['plan']."""
    messages = state.get("messages") or []
    if not messages:
        return {"plan": []}

    latest = messages[-1].get("content", "") if messages[-1].get("role") == "user" else ""
    history = _format_history(messages[:-1]) if len(messages) > 1 else ""
    entities = state.get("entities") or {}

    # 4th-layer Agent memory: cross-session user preferences. Injected as a
    # compact summary into the planner's user message so every plan starts
    # already knowing "this user prefers JD courier", "speaks English only",
    # "is allergic to lactose" etc. Failures are swallowed — preferences
    # are a nice-to-have, not a planning blocker.
    pref_summary = ""
    user_id = state.get("user_id")
    if user_id:
        try:
            from .tools.preferences import render_for_planner
            pref_summary = render_for_planner(user_id)
        except Exception:
            pref_summary = ""

    user_msg_parts = []
    if pref_summary:
        user_msg_parts.append(pref_summary)
    if history:
        user_msg_parts.append(f"Conversation so far:\n{history}")
    if entities:
        user_msg_parts.append(f"Known entities: {json.dumps(entities, ensure_ascii=False)}")
    user_msg_parts.append(f"Latest user question: {latest}")
    user_msg_parts.append("Produce the JSON plan.")
    user_msg = "\n\n".join(user_msg_parts)

    try:
        reply = chat_once(
            system=SYSTEM,
            user=user_msg,
            max_output_tokens=400,
            temperature=0.1,
        )
    except LLMError:
        return {
            "plan": _fallback_plan(latest),
            "current_step": 0,
            "trace": [{"node": "planner", "status": "llm_error"}],
        }

    plan = _parse_plan(reply) or _fallback_plan(latest)
    return {
        "plan": plan,
        "current_step": 0,
        "step_results": {},
        "trace": [{"node": "planner", "plan": plan, "raw_reply": reply[:500] if reply else ""}],
    }
