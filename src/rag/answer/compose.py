"""Build the LLM context and prompt.

Context layout is numeric-cited so the LLM must use [1], [2]... form.
Truncation is character-budget based; each chunk carries a header with
title and section path so the model can cite precisely.

The optional ``conversation`` parameter feeds the last 2 turns into the
prompt as **supporting context for understanding the user's intent
only**. The system prompt explicitly forbids using history as a source
of facts — every claim still has to be backed by a numbered chunk and
cited with [n]. This split exists because:

  - The rewriter already resolves pronouns ("那个怎么退") against
    history, but its output is a single-line standalone query and can
    miss conversational nuance ("PRO 那条呢" → "PRO" is a model name
    inferred from earlier turns; sometimes the rewrite drops it).
  - Letting the answerer SEE the recent turns (without trusting their
    facts) catches those misses without re-introducing the staleness
    drift that motivated the intent-router work.
"""
from __future__ import annotations

from ..config import settings
from ..types import Hit


SYSTEM_EN = """You are a grounded question-answering assistant.

Answer using ONLY the numbered sources below. Cite each fact with its source
number in square brackets, like [1] or [2][3]. Do not mention a fact that is
not supported by the sources.

If the sources do not contain enough information, reply exactly:
  I don't have enough information in the knowledge base to answer this.
(or in the user's language equivalent).

Answer in the same language as the user's question. Be concise and specific.
"""

SYSTEM_ZH = """你是一个基于知识库的问答助手。

只能使用下面编号的资料回答,每个事实必须在句末用方括号标注来源编号,
例如 [1] 或 [2][3]。不要引入资料里没有的事实。

如果资料不足以回答,请严格回复:
  当前知识库内信息不足,无法回答这个问题。

用与问题相同的语言回答,简洁、精确。
"""

# Strict directive appended when conversation history is included. Without
# this rule the LLM tends to repeat answers from prior turns even when
# fresh facts have moved (e.g. price changes between turns), which is the
# exact failure mode RAG is supposed to prevent.
HISTORY_RULE_EN = """\

CONVERSATION HISTORY USAGE — STRICT RULE:
- Use the history below ONLY to understand what the user is referring to
  (pronouns, follow-up topics, "this", "that").
- Do NOT treat any statement in the history as a verified fact, including
  statements made by the assistant in earlier turns.
- Every factual claim in your answer MUST be supported by the numbered
  Sources, and cited with [n]. If a fact appeared in the history but is
  not in the Sources, treat it as unknown.
"""

HISTORY_RULE_ZH = """\

历史对话使用规则(严格):
- 历史对话仅用于理解用户当前问题的指代关系(代词、后续话题、"那个"、"这个")。
- 不得将历史中出现的任何陈述当作已验证的事实,包括 Assistant 自己在前几轮的回答。
- 答案中的每一条事实都必须由下方"资料"支持,并用 [n] 标注。历史中提过但资料里
  没有的内容,一律视为未知。
"""


def _hit_block(n: int, hit: Hit, max_chars: int) -> str:
    breadcrumb = " / ".join(hit.section_path) if hit.section_path else ""
    header = f"[{n}] {hit.title}"
    if breadcrumb:
        header += f" — {breadcrumb}"
    body = hit.text.strip()
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + " ..."
    return f"{header}\n{body}"


def build_context(hits: list[Hit], max_chars: int | None = None) -> str:
    budget = max_chars or settings.max_context_chars
    per_block = max(400, budget // max(1, len(hits)))
    parts = [_hit_block(i + 1, h, per_block) for i, h in enumerate(hits)]
    return "\n\n".join(parts)


def _format_history_block(
    conversation: list[dict] | None,
    *,
    max_turns: int,
    max_chars_per_turn: int,
) -> str:
    """Format the tail of a conversation for the answerer prompt.

    ``max_turns`` counts user+assistant **pairs**, so the slice is
    ``-2 * max_turns`` messages. Long assistant answers are clipped per
    turn to keep prompt tokens predictable across long sessions.
    """
    if not conversation:
        return ""
    tail = conversation[-(max_turns * 2):]
    lines: list[str] = []
    for turn in tail:
        role = (turn.get("role") or "user").lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        tag = "User" if role == "user" else "Assistant"
        if len(content) > max_chars_per_turn:
            content = content[:max_chars_per_turn].rstrip() + " ..."
        lines.append(f"{tag}: {content}")
    return "\n".join(lines)


def build_answer_prompt(
    query: str,
    hits: list[Hit],
    language: str,
    conversation: list[dict] | None = None,
) -> tuple[str, str]:
    """Build (system, user) prompt for grounded answering.

    When ``conversation`` is provided AND ``settings.enable_answer_with_history``
    is True, the last 2 turns are included verbatim and the system prompt
    gets a strict "history is for intent only, not facts" addendum.
    """
    base_system = SYSTEM_ZH if language == "zh" else SYSTEM_EN
    context = build_context(hits)

    include_history = (
        getattr(settings, "enable_answer_with_history", True)
        and conversation
    )
    history_block = ""
    if include_history:
        history_block = _format_history_block(
            conversation,
            max_turns=getattr(settings, "answer_history_turns", 2),
            max_chars_per_turn=getattr(settings, "answer_history_chars_per_turn", 300),
        )

    if history_block:
        system = base_system + (HISTORY_RULE_ZH if language == "zh" else HISTORY_RULE_EN)
        user = (
            f"Recent conversation:\n{history_block}\n\n"
            f"Question: {query}\n\nSources:\n{context}\n\nAnswer:"
        )
    else:
        system = base_system
        user = f"Question: {query}\n\nSources:\n{context}\n\nAnswer:"
    return system, user
