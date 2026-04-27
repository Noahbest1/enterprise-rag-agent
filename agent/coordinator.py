"""Coordinator + Summary nodes.

``advance_node`` picks the next step to execute given the plan and already
completed steps. ``route_next`` returns which specialist to dispatch to
(or ``"summary"`` when all steps are done).

``summary_node`` folds all per-step results into one assistant message
referencing sources. Keeps citation numbers stable within one turn by
renumbering across specialists.
"""
from __future__ import annotations

from rag.answer.llm_client import LLMError, chat_once

from .persona import get_persona
from .state import AgentState


def advance_node(state: AgentState) -> dict:
    """Pick the next pending step whose depends_on are all done; advance cursor."""
    plan = state.get("plan") or []
    results = state.get("step_results") or {}
    done_ids = set(results.keys())

    for i, step in enumerate(plan):
        if step["status"] != "pending":
            continue
        if all(dep in done_ids for dep in step["depends_on"]):
            return {"current_step": i, "trace": [{"node": "advance", "picked_step_id": step["step_id"]}]}

    # Nothing left to run
    return {"current_step": -1, "trace": [{"node": "advance", "picked_step_id": None}]}


def route_next(state: AgentState) -> str:
    """Conditional edge target from advance_node."""
    idx = state.get("current_step", -1)
    plan = state.get("plan") or []
    if idx < 0 or idx >= len(plan):
        return "summary"
    return plan[idx]["agent"]


def mark_done_node(state: AgentState) -> dict:
    """After a specialist runs, mark its step as done."""
    idx = state.get("current_step", -1)
    plan = state.get("plan") or []
    if idx < 0 or idx >= len(plan):
        return {}
    new_plan = [dict(p) for p in plan]
    new_plan[idx]["status"] = "done"
    return {"plan": new_plan}


SUMMARY_SYSTEM_TEMPLATE = """{persona}

以下是若干专家代理的执行结果,请整合成一个面向用户的统一回复:

规则:
- 只使用下方专家结果和资料片段里的内容。不要编造价格、政策、订单号、快递单号或商品信息。
- 引用知识库的事实后挂 [n] 编号,编号就是下方 "资料片段" 的编号。
- 专家的订单/物流查询结果(如订单号、快递单号、配送状态)可以直接陈述,不用加 [n]。
- 如果订单专家找到了订单,必须把**订单号、商品、状态、快递单号**清晰告诉用户。
- 如果物流专家查到了物流信息,必须告知**当前状态**和**预计送达时间**(若有)。
- 所有专家都没有实质结果时,说明需要更多信息(例如订单号)。
- 语气遵循前述人设。回答 3-6 句为宜,简洁不啰嗦。
- 用与用户提问相同的语言回答(中文/英文)。
"""


def _build_sources_block(step_results: dict) -> tuple[str, list[dict]]:
    """Merge citations across steps into one numbered block."""
    n = 0
    lines: list[str] = []
    flat: list[dict] = []
    for step_id in sorted(step_results):
        res = step_results[step_id]
        if not isinstance(res, dict):
            continue
        for cit in res.get("citations", []):
            n += 1
            title = cit.get("title") or ""
            section = " / ".join(cit.get("section_path") or []) if isinstance(cit.get("section_path"), list) else (cit.get("section_title") or "")
            snippet = cit.get("snippet") or ""
            header = f"[{n}] {title}"
            if section:
                header += f" — {section}"
            lines.append(f"{header}\n{snippet}")
            flat.append(
                {
                    "n": n,
                    "title": title,
                    "section_path": cit.get("section_path") or [],
                    "source_id": cit.get("source_id"),
                    "snippet": snippet,
                    "from_step": step_id,
                }
            )
    return "\n\n".join(lines), flat


def summary_node(state: AgentState) -> dict:
    persona = get_persona(state["tenant"])
    user_msg = ""
    msgs = state.get("messages") or []
    if msgs and msgs[-1].get("role") == "user":
        user_msg = msgs[-1].get("content", "")

    step_results = state.get("step_results") or {}
    sources_block, flat_citations = _build_sources_block(step_results)

    # If every step abstained or errored out, answer honestly.
    any_answer = any(
        isinstance(r, dict) and r.get("answer") and not r.get("abstain") and not r.get("error")
        for r in step_results.values()
    )

    # Also surface specialists that don't produce citations (order, logistics).
    specialist_lines: list[str] = []
    for step_id in sorted(step_results):
        res = step_results[step_id]
        if not isinstance(res, dict):
            continue
        agent = res.get("agent") or "?"
        if agent in ("product_qa", "policy_qa"):
            continue  # these are represented via citations
        if res.get("answer"):
            specialist_lines.append(f"[{agent}]\n{res['answer']}")

    specialist_block = "\n\n".join(specialist_lines) if specialist_lines else ""

    system = SUMMARY_SYSTEM_TEMPLATE.format(persona=persona["system_prompt"])
    user_parts = [f"用户问题: {user_msg}"]
    if specialist_block:
        user_parts.append(f"专家结果:\n{specialist_block}")
    user_parts.append(f"资料片段:\n{sources_block or '(无)'}")
    user_parts.append("请输出面向用户的回复。")
    user = "\n\n".join(user_parts)

    if not any_answer:
        fallback = "非常抱歉,当前没有找到相关资料。如果您能提供更多信息(例如订单号、商品名称),我可以帮您进一步查询。"
        return {
            "final_answer": fallback,
            "messages": [{"role": "assistant", "content": fallback}],
            "trace": [{"node": "summary", "mode": "abstain"}],
        }

    try:
        answer = chat_once(
            system=system,
            user=user,
            max_output_tokens=600,
            temperature=0.2,
        )
    except LLMError as e:
        answer = f"非常抱歉,服务暂时不可用({e})。"

    return {
        "final_answer": answer,
        "messages": [{"role": "assistant", "content": answer, "citations": flat_citations}],
        "trace": [{"node": "summary", "citation_count": len(flat_citations)}],
    }
