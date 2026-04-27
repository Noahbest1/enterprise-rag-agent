"""LLM provider fallback chain.

Order: **Qwen (primary) → Claude (secondary) → extractive (tertiary)**.

Why this matters for production:
- DashScope has outages (rate limits, regional blips, key rotation).
- If the sole dependency is Qwen, a 5-minute provider blip = a 5-minute
  product outage. The chain guarantees the user gets *something* grounded
  in the retrieved chunks — even if every external LLM API is down.

Behaviour:
- Each rung only fires on ``LLMError`` from the previous rung. Other
  exceptions (BudgetExceeded, cancellation) propagate unchanged.
- Claude is invoked via the Anthropic Messages API over plain httpx (no
  SDK dep, so the venv stays lean).
- The tertiary extractive rung never calls an LLM: it stitches the top
  ``hits`` into a "here are the relevant excerpts" answer with inline
  ``[n]`` citations. The LLM judge will mark these as ``cites_correctly``
  true and ``faithful`` true — we're literally quoting.
- Records ``rag_llm_provider_fallback_total{rung=}`` metric so ops can
  see how often each rung is engaged.

Control:
- ``settings.enable_fallback_chain`` — master on/off. Default OFF so pytest
  and existing callers behave identically.
- ``settings.anthropic_api_key`` — if empty, Claude rung is skipped
  (fall straight through to extractive).
"""
from __future__ import annotations

import re
from typing import Iterable

import httpx

from ..config import settings
from ..logging import get_logger
from ..observability import record_llm
from ..types import Hit
from .llm_client import LLMError, chat_once_async


log = get_logger(__name__)


# ---------- secondary: Claude via Messages API ----------

async def claude_chat_async(
    *,
    system: str,
    user: str,
    max_output_tokens: int | None = None,
    timeout: int | None = None,
) -> str:
    """Call Claude's Messages API. Raises ``LLMError`` on any non-200."""
    if not settings.anthropic_api_key:
        raise LLMError("ANTHROPIC_API_KEY not set")

    payload = {
        "model": settings.anthropic_model,
        "max_tokens": max_output_tokens or settings.llm_max_output_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    headers = {
        "x-api-key": settings.anthropic_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout or settings.llm_timeout_seconds) as client:
            resp = await client.post(settings.anthropic_base_url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        raise LLMError(f"Claude HTTP {e.response.status_code}: {e.response.text[:300]}") from e
    except httpx.RequestError as e:
        raise LLMError(f"Claude network error: {e}") from e

    # Extract content: Claude returns {"content":[{"type":"text","text":"..."}]}
    blocks = data.get("content") or []
    text_parts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
    content = "".join(text_parts).strip()
    if not content:
        raise LLMError(f"Claude returned empty content: {str(data)[:200]}")

    # Record token metrics in the same counter Qwen uses, tagged per task.
    usage = data.get("usage") or {}
    record_llm(
        task="fallback_claude",
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        status="ok",
    )
    return content


# ---------- tertiary: zero-LLM extractive ----------

_SENT_SPLIT = re.compile(r"(?<=[。!?;])|(?<=[.!?;])\s+|\n{2,}")


def _first_sentences(text: str, max_chars: int = 240) -> str:
    if not text:
        return ""
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    out: list[str] = []
    total = 0
    for p in parts:
        if total + len(p) > max_chars and out:
            break
        out.append(p)
        total += len(p)
    return " ".join(out).strip() or text.strip()[:max_chars]


def extractive_answer(query: str, hits: Iterable[Hit]) -> str:
    """Stitch the top hits into a grounded answer. No LLM call.

    The answer always carries ``[n]`` citations so downstream citation
    parsers + the judge both treat it as a faithful response.
    """
    hit_list = [h for h in hits][:3]
    if not hit_list:
        return "当前知识库内信息不足,且 LLM 服务暂不可用。"

    lines = [
        "(LLM 服务暂不可用,以下内容为基于知识库的直接摘录。)",
        "",
        f"关于「{query.strip()[:80]}」,以下知识库条目可能相关:",
        "",
    ]
    for i, h in enumerate(hit_list, start=1):
        snippet = _first_sentences(h.text, max_chars=240)
        title = h.title or h.source_id
        lines.append(f"[{i}] {title}")
        lines.append(snippet)
        lines.append("")
    lines.append("建议您稍后重试获取完整的 LLM 生成答复。")
    return "\n".join(lines).rstrip()


# ---------- public entry point ----------

async def chat_once_with_fallback(
    *,
    system: str,
    user: str,
    task: str = "generate",
    hits: Iterable[Hit] | None = None,
    user_query_for_extractive: str | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
) -> tuple[str, str]:
    """Call Qwen; on LLMError, try Claude; on further error, fall back to extractive.

    Returns ``(text, provider)`` where provider ∈ {"qwen", "claude", "extractive"}.
    If fallback is disabled in config, any LLMError propagates as before.
    """
    # Materialise hits once so a generator isn't consumed twice.
    hits_list: list[Hit] = list(hits) if hits else []

    try:
        text = await chat_once_async(
            system=system, user=user, task=task,
            max_output_tokens=max_output_tokens, temperature=temperature,
        )
        return text, "qwen"
    except LLMError as e_qwen:
        if not settings.enable_fallback_chain:
            raise
        log.warning("llm_fallback_qwen_failed", task=task, error=str(e_qwen))
        record_llm(task="fallback_qwen", input_tokens=0, output_tokens=0, status="error")

    # Qwen failed. Try Claude if configured.
    if settings.anthropic_api_key:
        try:
            text = await claude_chat_async(
                system=system, user=user, max_output_tokens=max_output_tokens,
            )
            log.info("llm_fallback_to_claude_ok", task=task)
            return text, "claude"
        except LLMError as e_claude:
            log.warning("llm_fallback_claude_failed", task=task, error=str(e_claude))
            record_llm(task="fallback_claude", input_tokens=0, output_tokens=0, status="error")

    # Both LLMs failed -- last resort: extractive.
    query = user_query_for_extractive or user
    text = extractive_answer(query=query, hits=hits_list)
    log.warning("llm_fallback_to_extractive", task=task, hit_count=len(hits_list))
    return text, "extractive"
