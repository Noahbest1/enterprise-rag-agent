"""LangGraph State for the e-commerce agent.

Kept flat and JSON-serialisable so it can checkpoint to Redis/SQLite later.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict
from operator import add


class PlanStep(TypedDict):
    step_id: int
    agent: Literal["product_qa", "policy_qa", "order", "logistics", "aftersale", "recommend"]
    query: str              # what this step should ask
    depends_on: list[int]   # previous step_ids whose output feeds this step
    status: Literal["pending", "done", "skipped", "error"]


class AgentState(TypedDict, total=False):
    # Identity
    tenant: Literal["jd", "taobao"]      # which persona + KB to use
    user_id: str

    # Conversation (persists across turns via LangGraph checkpointer)
    messages: Annotated[list[dict], add]  # each {"role":"user"|"assistant"|"tool", "content": str}
    # Planner output
    plan: list[PlanStep]
    current_step: int

    # Per-step results keyed by step_id
    step_results: dict[int, Any]

    # Entities mentioned in conversation (product_name, order_id, etc.) so
    # follow-up turns can reuse them without re-asking.
    entities: dict[str, Any]

    # Final assistant answer for this turn
    final_answer: str

    # Trace for debugging / UI streaming
    trace: Annotated[list[dict], add]
