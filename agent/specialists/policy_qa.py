"""PolicyQA specialist.

Same underlying RAG call as ProductQA, but with a metadata filter that
prefers policy-like chunks (return policy, shipping, membership, etc.).
We express that as a source_path_contains filter. When no filter matches
(e.g. the KB doesn't happen to have a policy-named file) we fall back to
an unfiltered search so the user still gets an answer.
"""
from __future__ import annotations

from ..persona import get_persona
from ..state import AgentState
from ..tools import RAGError, rag_search


POLICY_PATH_HINTS = ("policy", "return", "aftersale", "membership", "vip", "plus", "coupon", "dispute", "freight", "promotion", "shipping")


def _policy_filter(query: str) -> dict | None:
    # Basic keyword-based filter selection. Falls back to unfiltered if
    # nothing obvious matches. Kept lightweight -- the RAG itself has
    # source_prior that already nudges relevant chunks up.
    q = query.lower()
    triggers = {
        "return": ["退货", "退款", "无理由", "return", "refund"],
        "shipping": ["运费", "邮费", "shipping", "freight", "运费险"],
        "membership": ["plus", "88vip", "会员", "member"],
        "aftersale": ["售后", "保价", "维修", "投诉", "维权", "介入"],
    }
    for family, words in triggers.items():
        if any(w in q for w in words):
            return {"source_path_contains": family}
    return None


def _find_current_step(state: AgentState):
    plan = state.get("plan") or []
    idx = state.get("current_step") or 0
    if 0 <= idx < len(plan):
        return plan[idx], idx
    return None, idx


def policy_qa_node(state: AgentState) -> dict:
    step, idx = _find_current_step(state)
    if step is None:
        return {}

    persona = get_persona(state["tenant"])
    kb_id = persona["kb_id"]
    query = step["query"]
    flt = _policy_filter(query)

    conv = [
        {"role": m["role"], "content": m["content"]}
        for m in (state.get("messages") or [])[-6:]
        if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()
    ]

    try:
        result = rag_search(query, kb_id, conversation=conv, filter=flt, use_multi_query=False)
        # If filter produced nothing useful, retry without filter.
        if not result["citations"] and flt:
            result = rag_search(query, kb_id, conversation=conv, use_multi_query=False)
    except RAGError as e:
        return {
            "step_results": {
                **(state.get("step_results") or {}),
                step["step_id"]: {"error": str(e), "agent": "policy_qa"},
            },
            "trace": [{"node": "policy_qa", "error": str(e), "step_id": step["step_id"]}],
        }

    return {
        "step_results": {
            **(state.get("step_results") or {}),
            step["step_id"]: {
                "agent": "policy_qa",
                "query": query,
                "filter": flt,
                "answer": result["answer"],
                "citations": result["citations"],
                "abstain": result["abstain"],
            },
        },
        "trace": [
            {
                "node": "policy_qa",
                "step_id": step["step_id"],
                "filter": flt,
                "citations_count": len(result["citations"]),
                "abstain": result["abstain"],
            }
        ],
    }
