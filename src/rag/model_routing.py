"""Pick the right Qwen model tier per task.

Tasks we classify:
- "rewrite"      : cross-lingual query rewrite. Short, deterministic.
- "intent"       : intent classification / planner. Short.
- "contextual"   : contextual summary per chunk at ingestion. Short, many.
- "generate"     : final grounded answer. Long, quality-critical.
- "judge"        : LLM-as-Judge in eval. Short, deterministic.

For "rewrite" / "intent" / "contextual" / "judge" we route to ``qwen_turbo_model``
(cheaper, faster). For "generate" we stay on ``qwen_model`` (quality).
Disabled by default so behaviour is predictable; enable via
``MODEL_ROUTING=true`` in env.
"""
from __future__ import annotations

from typing import Literal

from .config import settings


TaskKind = Literal[
    "rewrite", "intent", "contextual", "generate", "judge",
    "meta_query", "chitchat",
    "default",
]


_ROUTING: dict[TaskKind, str] = {
    "rewrite": "turbo",
    "intent": "turbo",
    "contextual": "turbo",
    "judge": "turbo",
    # meta_query / chitchat are short, latency-sensitive, no retrieval to
    # ground against — turbo is the right tier.
    "meta_query": "turbo",
    "chitchat": "turbo",
    "generate": "plus",
    "default": "plus",
}


def resolve_model(task: TaskKind = "default") -> str:
    if not getattr(settings, "enable_model_routing", False):
        return settings.qwen_model
    tier = _ROUTING.get(task, "plus")
    if tier == "turbo":
        return getattr(settings, "qwen_turbo_model", settings.qwen_model)
    return settings.qwen_model
