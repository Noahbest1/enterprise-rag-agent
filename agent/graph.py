"""Compose the LangGraph StateGraph.

Flow:

    planner ─► advance ─┬─► product_qa ──► mark_done ─► advance ...
                        ├─► policy_qa ───► mark_done ─► advance ...
                        └─► summary ──► END

Checkpointer:
    A checkpointer persists state between graph invocations for the same
    ``thread_id``, which is how multi-turn conversations survive a process
    restart. We default to SQLite (file at ``data/langgraph.sqlite``) so
    no infra is required. Pass ``checkpointer=None`` for stateless tests.
"""
from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

from .coordinator import advance_node, mark_done_node, route_next, summary_node
from .planner import planner_node
from .specialists import (
    account_node,
    aftersale_node,
    complaint_node,
    invoice_node,
    logistics_node,
    order_node,
    policy_qa_node,
    product_qa_node,
    recommend_node,
)
from .state import AgentState


_DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "data" / "langgraph.sqlite"


def get_default_checkpointer():
    """File-backed SQLite checkpointer (sync). Safe to call many times.

    For async graphs (the FastAPI streaming endpoint) use
    :func:`get_default_async_checkpointer` instead — sync ``SqliteSaver``
    will raise ``NotImplementedError`` from inside ``graph.astream``.
    """
    _DEFAULT_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    import sqlite3
    conn = sqlite3.connect(str(_DEFAULT_CHECKPOINT_PATH), check_same_thread=False)
    return SqliteSaver(conn)


def get_default_async_checkpointer() -> AsyncSqliteSaver:
    """File-backed AsyncSqliteSaver (uses aiosqlite under the hood).

    Required for ``graph.astream`` / ``graph.aget_state`` — the sync
    SqliteSaver throws ``NotImplementedError`` when those are called.
    Same on-disk file as the sync version, so a process can use either.
    """
    _DEFAULT_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    import aiosqlite
    conn = aiosqlite.connect(str(_DEFAULT_CHECKPOINT_PATH))
    return AsyncSqliteSaver(conn)


def build_graph(checkpointer=None, *, with_checkpointer: bool = False, async_mode: bool = False):
    """Build and compile the agent graph.

    - ``with_checkpointer=True`` enables persistent state (SQLite).
    - ``async_mode=True`` picks the async-flavoured SqliteSaver (required
      for ``graph.astream`` callers); the sync variant is fine for tests
      that use ``graph.invoke``.
    - Pass a specific ``checkpointer`` (e.g. MemorySaver) for tests.
    - Omit both for an ephemeral graph -- convenient for one-shot CLI calls.
    """
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("advance", advance_node)
    g.add_node("product_qa", product_qa_node)
    g.add_node("policy_qa", policy_qa_node)
    g.add_node("order", order_node)
    g.add_node("logistics", logistics_node)
    g.add_node("aftersale", aftersale_node)
    g.add_node("recommend", recommend_node)
    g.add_node("invoice", invoice_node)
    g.add_node("complaint", complaint_node)
    g.add_node("account", account_node)
    g.add_node("mark_done", mark_done_node)
    g.add_node("summary", summary_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "advance")
    g.add_conditional_edges(
        "advance",
        route_next,
        {
            "product_qa": "product_qa",
            "policy_qa": "policy_qa",
            "order": "order",
            "logistics": "logistics",
            "aftersale": "aftersale",
            "recommend": "recommend",
            "invoice": "invoice",
            "complaint": "complaint",
            "account": "account",
            "summary": "summary",
        },
    )
    for specialist in ("product_qa", "policy_qa", "order", "logistics", "aftersale", "recommend", "invoice", "complaint", "account"):
        g.add_edge(specialist, "mark_done")
    g.add_edge("mark_done", "advance")
    g.add_edge("summary", END)

    cp = checkpointer
    if cp is None and with_checkpointer:
        cp = get_default_async_checkpointer() if async_mode else get_default_checkpointer()
    return g.compile(checkpointer=cp) if cp else g.compile()


def _stub_specialist(name: str):
    """Phase-A placeholder that records a stub result and lets the graph advance."""
    def _node(state: AgentState) -> dict:
        plan = state.get("plan") or []
        idx = state.get("current_step", -1)
        if idx < 0 or idx >= len(plan):
            return {}
        step = plan[idx]
        return {
            "step_results": {
                **(state.get("step_results") or {}),
                step["step_id"]: {
                    "agent": name,
                    "query": step["query"],
                    "answer": f"({name} 专家尚未实现,Phase B/C 会补上)",
                    "citations": [],
                    "abstain": True,
                },
            },
            "trace": [{"node": name, "status": "stub", "step_id": step["step_id"]}],
        }
    return _node
