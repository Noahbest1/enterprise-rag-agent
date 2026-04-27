"""ProductQA specialist.

Wraps the RAG service. The product and policy KB are the same for now
(``jd_demo`` or ``taobao_demo``); in a production multi-tenant setup
they'd be separate collections.
"""
from __future__ import annotations

from ..persona import get_persona
from ..state import AgentState
from ..tools import RAGError, rag_search


def _to_rag_conversation(messages: list[dict]) -> list[dict]:
    """Keep only user/assistant turns for RAG's coreference resolution."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in messages[-6:]
        if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()
    ]


def _find_current_step(state: AgentState):
    plan = state.get("plan") or []
    idx = state.get("current_step") or 0
    if 0 <= idx < len(plan):
        return plan[idx], idx
    return None, idx


def product_qa_node(state: AgentState) -> dict:
    step, idx = _find_current_step(state)
    if step is None:
        return {}

    persona = get_persona(state["tenant"])
    kb_id = persona["kb_id"]
    query = step["query"]

    conv = _to_rag_conversation(state.get("messages") or [])
    try:
        result = rag_search(query, kb_id, conversation=conv, use_multi_query=False)
    except RAGError as e:
        return {
            "step_results": {
                **(state.get("step_results") or {}),
                step["step_id"]: {"error": str(e), "agent": "product_qa"},
            },
            "trace": [{"node": "product_qa", "error": str(e), "step_id": step["step_id"]}],
        }

    return {
        "step_results": {
            **(state.get("step_results") or {}),
            step["step_id"]: {
                "agent": "product_qa",
                "query": query,
                "answer": result["answer"],
                "citations": result["citations"],
                "abstain": result["abstain"],
            },
        },
        "trace": [
            {
                "node": "product_qa",
                "step_id": step["step_id"],
                "rewritten": result.get("rewritten_query"),
                "citations_count": len(result["citations"]),
                "abstain": result["abstain"],
            }
        ],
    }
