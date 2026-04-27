"""Multi-query expansion.

A single user query, especially a vague one ("这个怎么搞", "有啥功能"),
often fails to retrieve the right chunks. We ask the LLM to generate a
handful of alternative search queries covering different phrasings,
specificity levels, and languages. Each variant runs through the normal
retrieve() pipeline; results are fused so every good chunk has multiple
paths to make it into the final top-k.

Failure is non-fatal: on any error we return just [original_query].
"""
from __future__ import annotations

import json
import re

from ..answer.llm_client import LLMError, chat_once


SYSTEM = """You generate alternative search queries for a hybrid BM25+vector knowledge base.

Given the user's original query (and optional conversation), produce 3 to 4 alternative queries that would retrieve diverse, relevant chunks.

Rules:
- Output ONLY a JSON array of strings. No keys, no explanation, no markdown fences.
- Each variant must be a search query, NOT an answer.
- Cover different angles:
    * one more specific / noun-heavy
    * one more general / conceptual
    * one with synonyms or alternate phrasing
    * if the original is Chinese include at least one English variant, and vice versa
- Preserve proper nouns, numbers, versions, and quoted strings EXACTLY.
- Do NOT repeat the original query.
- Output example:
  ["how to set up a connection", "configure source destination sync", "connection setup tutorial", "新建连接教程"]
"""


_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)


def _parse_array(reply: str) -> list[str]:
    m = _ARRAY_RE.search(reply or "")
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
    return out


def expand_queries(
    query: str,
    conversation: list[dict] | None = None,
    max_variants: int = 4,
) -> list[str]:
    """Return [original, var1, var2, ...] with up to ``max_variants`` variants."""
    query = query.strip()
    if not query:
        return [query]

    # Build user message; include last turn of conversation for context.
    user_msg = f"Original query: {query}"
    if conversation:
        last = conversation[-2:]
        hist = "\n".join(
            f"{t.get('role','user').capitalize()}: {(t.get('content') or '').strip()[:300]}"
            for t in last
            if (t.get("content") or "").strip()
        )
        if hist:
            user_msg = f"Conversation tail:\n{hist}\n\n" + user_msg

    try:
        reply = chat_once(
            system=SYSTEM,
            user=user_msg,
            max_output_tokens=220,
            temperature=0.3,
        )
    except LLMError:
        return [query]

    variants = _parse_array(reply)
    # Dedupe (case-insensitive), drop ones that equal the original, cap count.
    seen = {query.lower()}
    out = [query]
    for v in variants:
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
        if len(out) - 1 >= max_variants:
            break
    return out
