"""Generate a grounded answer from retrieved hits."""
from __future__ import annotations

from ..query.normalize import detect_language
from ..types import Hit
from .compose import build_answer_prompt
from .llm_client import LLMError, chat_once


def generate_answer(
    query: str,
    hits: list[Hit],
    conversation: list[dict] | None = None,
) -> tuple[str, str]:
    """Return (answer_text, language)."""
    language = detect_language(query)
    system, user = build_answer_prompt(query, hits, language, conversation=conversation)
    try:
        reply = chat_once(system=system, user=user)
    except LLMError as e:
        fallback = (
            "当前知识库内信息不足,无法回答这个问题。"
            if language == "zh"
            else "I don't have enough information in the knowledge base to answer this."
        )
        return f"{fallback} (LLM error: {e})", language
    return reply.strip(), language
