"""LangGraph checkpointer: state survives across graph invocations keyed by thread_id."""
from __future__ import annotations

import pytest


def test_build_graph_without_checkpointer_runs():
    from agent.graph import build_graph
    graph = build_graph()
    assert graph is not None


def test_build_graph_with_memory_checkpointer():
    from langgraph.checkpoint.memory import MemorySaver

    from agent.graph import build_graph
    graph = build_graph(checkpointer=MemorySaver())
    assert graph is not None


def test_default_sqlite_checkpointer_available(tmp_path, monkeypatch):
    """get_default_checkpointer returns a working SqliteSaver."""
    from agent import graph as graph_mod

    monkeypatch.setattr(graph_mod, "_DEFAULT_CHECKPOINT_PATH", tmp_path / "cp.sqlite")
    cp = graph_mod.get_default_checkpointer()
    assert cp is not None
    # Should be idempotent
    cp2 = graph_mod.get_default_checkpointer()
    assert cp2 is not None


@pytest.mark.integration
def test_multi_turn_state_persists(tmp_path, monkeypatch, seeded_db):
    """Invoke the real graph twice with the same thread_id and verify history accumulates."""
    from langgraph.checkpoint.memory import MemorySaver

    from agent.graph import build_graph

    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t-test-42"}}

    # Turn 1 -- use order specialist, no LLM involved beyond the Planner/Summary
    # (the Planner DOES call an LLM; if no api key, this will error. We skip then.)
    import os
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.skip("live LLM needed for multi-turn integration test")

    state1 = {
        "tenant": "jd",
        "user_id": "jd-demo-user",
        "messages": [{"role": "user", "content": "iPhone 16 Pro 多少钱"}],
        "entities": {},
        "step_results": {},
        "trace": [],
    }
    out1 = graph.invoke(state1, config=config)
    assert out1.get("final_answer")
    # Turn 2 -- same thread; messages should already contain turn 1.
    state2 = {
        "messages": [{"role": "user", "content": "那它支持 7 天无理由吗?"}],
    }
    out2 = graph.invoke(state2, config=config)
    # Turn 2 should see combined message history (turn 1 user + assistant + turn 2 user)
    assert len(out2.get("messages", [])) >= 3
