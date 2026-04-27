"""Complaint specialist — dry-run preview only.

This specialist used to write a row in `complaints` and (for high severity)
an audit row, the moment the planner routed a step to it. That coupling
created the same double-write trap we already removed for AfterSale: a
mis-classified greeting like "你好 你是什么客服" silently wrote a real
ticket. Fixed by making the specialist DRY-RUN: it classifies severity +
topic + would-escalate signals, returns those as a `preview` payload,
and the UI button at `/agent/actions/submit-complaint` is the SOLE path
that actually inserts a row + writes the audit breadcrumb.

What this node does now:
  1. Classify emotion (severity + matched signals) + topic from the
     user's RAW utterance (not the planner-rewritten neutral query).
  2. Resolve order_id from the query, state.entities, or leave null.
  3. Return a preview dict the UI can render with a "提交工单" button.
  4. Do NOT touch the database. Do NOT write audit. Both happen only
     when the user explicitly clicks submit.
"""
from __future__ import annotations

import re

from rag.nlp.emotion import classify

from ..state import AgentState
from ..tools.complaints import _pick_human_agent


_ORDER_ID_RE = re.compile(r"\b(?:JD|TB)\d{8,16}\b", re.IGNORECASE)


def _default_user_for(tenant: str) -> str:
    return "jd-demo-user" if tenant == "jd" else "tb-demo-user"


def _pick_order_id(state: AgentState, query: str) -> str | None:
    m = _ORDER_ID_RE.search(query)
    if m:
        return m.group(0)
    entities = state.get("entities") or {}
    if entities.get("last_order_id"):
        return entities["last_order_id"]
    return None


def _latest_user_content(state: AgentState, fallback: str) -> str:
    """The raw user utterance is richer than the planner-rewritten step query.
    We keep the actual user text so severity detection catches their real tone."""
    msgs = state.get("messages") or []
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
            return m["content"]
    return fallback


def _step_from_plan(state: AgentState) -> tuple[dict | None, int]:
    plan = state.get("plan") or []
    idx = state.get("current_step", -1)
    if idx is not None and 0 <= idx < len(plan):
        return plan[idx], idx
    return None, idx


def _topic_label(topic: str) -> str:
    return {
        "delivery": "物流延误",
        "quality": "商品质量",
        "service": "客服服务",
        "refund": "退款问题",
        "price": "价格问题",
        "other": "其他",
    }.get(topic, topic)


def _severity_label(severity: str) -> str:
    return {"high": "紧急", "medium": "普通", "low": "一般"}.get(severity, severity)


def _compose_preview_answer(verdict, would_escalate: bool) -> str:
    """User-facing narration for the dry-run preview. We deliberately do NOT
    pretend a ticket exists; we tell the user we IDENTIFIED their issue and
    invite them to confirm submission via the button."""
    lines: list[str] = [
        f"我识别到您可能想反馈一个问题(类型:{_topic_label(verdict.topic)},"
        f"严重等级:{_severity_label(verdict.severity)})。"
    ]
    if would_escalate:
        lines.append(
            "由于触发了升级信号,如确认提交,我会同时升级到资深客服,"
            "1 小时内主动联系您。"
        )
    else:
        sla_label = {"medium": "4 小时", "low": "24 小时"}.get(verdict.severity, "24 小时")
        lines.append(f"如确认提交,我们会在 {sla_label} 内回复您。")
    if verdict.matched_high:
        lines.append(f"(识别到的升级信号:{'、'.join(verdict.matched_high[:3])})")
    lines.append("点击下方「提交工单」即可正式立案,或继续描述问题以便我更准确判断。")
    return "\n".join(lines)


def complaint_node(state: AgentState) -> dict:
    step, _ = _step_from_plan(state)
    if step is None:
        return {}
    step_id = step["step_id"]
    query = step.get("query") or ""

    # Severity is about the USER'S tone, so classify the raw utterance —
    # not the planner's rewritten sub-question, which reads much more neutral.
    user_utterance = _latest_user_content(state, query)
    verdict = classify(user_utterance)

    tenant = state.get("tenant") or "jd"
    user_id = state.get("user_id") or _default_user_for(tenant)
    order_id = _pick_order_id(state, query)

    would_escalate = verdict.severity == "high"
    suggested_assignee = _pick_human_agent(tenant) if would_escalate else None

    preview = {
        "severity": verdict.severity,
        "topic": verdict.topic,
        "would_escalate": would_escalate,
        "suggested_assignee": suggested_assignee,
        "user_id": user_id,
        "tenant": tenant,
        "order_id": order_id,
        # The frontend echoes this back when the user clicks submit, so the
        # endpoint can re-hash it for the audit row. The raw text never
        # leaves the user's session and never enters the audit table.
        "content_for_submit": user_utterance,
    }

    answer = _compose_preview_answer(verdict, would_escalate)

    return {
        "entities": {
            **(state.get("entities") or {}),
            # NOTE: no last_complaint_id yet — the row doesn't exist until the
            # button hits /agent/actions/submit-complaint. We do expose the
            # severity preview so a follow-up turn ("那就提交吧") can read it.
            "last_complaint_preview_severity": verdict.severity,
            "last_complaint_preview_topic": verdict.topic,
            "last_complaint_preview_would_escalate": would_escalate,
            **({"last_order_id": order_id} if order_id else {}),
        },
        "step_results": {
            **(state.get("step_results") or {}),
            step_id: {
                "agent": "complaint",
                "query": query,
                "answer": answer,
                # complaint stays None until the user explicitly submits.
                "complaint": None,
                "preview": preview,
                "severity": verdict.severity,
                "topic": verdict.topic,
                "matched_high": verdict.matched_high,
                "matched_medium": verdict.matched_medium,
                "citations": [],
                "abstain": False,
            },
        },
        "trace": [{
            "node": "complaint", "step_id": step_id,
            "severity": verdict.severity, "topic": verdict.topic,
            "would_escalate": would_escalate, "dry_run": True,
        }],
    }
