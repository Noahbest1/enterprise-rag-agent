"""LLM-based query rewrite for cross-lingual + multi-turn retrieval.

Two jobs in one LLM call:

1. **Cross-lingual expansion** -- ZH query gets EN keyword synonyms
   appended so a bilingual corpus can match either side. Vice versa.

2. **Coreference resolution** -- if a ``conversation`` history is passed,
   pronouns in the latest query ("this", "that", "it", "这个",
   "上面那个") are resolved against earlier turns. The subject entity
   (product, version, person) is carried forward so the query can be
   searched independently.

Failure is non-fatal: the original query is returned if the LLM errors.
"""
from __future__ import annotations

from ..answer.llm_client import chat_once, LLMError
from ..config import settings


SYSTEM = """You rewrite the user's latest search query into a complete, standalone query so a bilingual BM25 + vector index can retrieve relevant chunks.

Rules:
- Resolve pronouns and references ("this", "that", "it", "这个", "那个", "上面那个", "前面说的") in the latest query using the earlier conversation turns.
- **Recency bias** -- when the latest query refers to OR operates on prior content (e.g. "translate that", "summarize the above", "翻译成英文", "具体讲讲"), the content source is the **MOST RECENT assistant message**, NOT the earliest user question. The latest assistant turn is what the user is actually looking at on screen.
- Carry forward the implicit subject (product name, version, person, topic) from the most recent relevant turn when the follow-up would otherwise be ambiguous.
- Add cross-lingual keyword synonyms: Chinese queries should be followed by English keyword hints, English queries should be followed by Chinese hints.
- Preserve proper nouns, numbers, versions, flags, commands, and quoted strings EXACTLY.
- Do NOT answer the question. Do NOT add unrelated terms.
- Output ONLY the standalone query on a single line. No prefix, no quotes, no explanation.

Examples (to clarify the recency rule):

Conversation:
  User: 课件讲了什么
  Assistant: 课件讲了软件设计原则的概念。
  User: 具体讲讲
  Assistant: 包括 Liskov 替换原则、接口隔离原则、组合优于继承等。
Latest user question: 翻译成英文
Correct rewrite: Liskov substitution principle, interface segregation principle, composition over inheritance translate into English 翻译

Conversation:
  User: PLUS 会员价格
  Assistant: 198 元/年。
  User: 那 PRO 呢
Correct rewrite: PRO 会员年费价格 PRO membership annual fee
"""


def _format_conversation(conversation: list[dict] | None) -> str:
    if not conversation:
        return ""
    lines = []
    # Keep the tail -- only last 6 turns matter for coreference.
    for turn in conversation[-6:]:
        role = turn.get("role", "user").lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        tag = "User" if role == "user" else "Assistant"
        # Clip very long assistant answers; the structure carries the meaning.
        if len(content) > 400:
            content = content[:400].rstrip() + " ..."
        lines.append(f"{tag}: {content}")
    return "\n".join(lines)


def rewrite_query(query: str, conversation: list[dict] | None = None) -> str:
    if not settings.enable_query_rewrite:
        return query
    if len(query.strip()) <= 2 and not conversation:
        return query

    history_block = _format_conversation(conversation)
    if history_block:
        user_msg = (
            f"Conversation so far:\n{history_block}\n\n"
            f"Latest user question: {query.strip()}\n\n"
            f"Rewrite the latest question into a standalone bilingual search query."
        )
    else:
        user_msg = query.strip()

    try:
        reply = chat_once(
            system=SYSTEM,
            user=user_msg,
            max_output_tokens=160,
            temperature=0.1,
            task="rewrite",
        )
    except LLMError:
        return query

    reply = reply.strip().splitlines()[0].strip() if reply else ""
    if not reply:
        return query

    # Always keep the original phrasing so BM25 exact matches still score.
    if conversation:
        # When rewriting across turns, LLM output already carries context;
        # don't re-prepend the ambiguous original ("这个怎么退货").
        return reply
    return f"{query} {reply}".strip()
