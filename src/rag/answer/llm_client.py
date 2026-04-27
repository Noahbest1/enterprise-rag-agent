"""LLM client (sync + async) with optional semantic-level caching.

Keeps the original ``chat_once`` sync signature so everything we already
wrote (rewrite, contextual, planner, summary) continues to work. Adds
``chat_once_async`` on top using httpx.AsyncClient, plus a cache layer
keyed by (model, system+user hash) so repeat queries within a TTL are free.

Cache details:
- Only requests with temperature <= 0.2 are cached (deterministic-ish).
- Key excludes max_tokens so variations of cap length don't break cache hits.
- 10-minute TTL by default. Adjustable via ``llm_cache_ttl_seconds`` setting.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from ..budget import (
    BudgetExceeded,
    check_prompt_budget,
    estimate_tokens,
    tenant_tracker,
)
from ..cache import cache_get, cache_set, stable_key
from ..config import settings
from ..logging import current_tenant_id, get_logger


log = get_logger(__name__)


class LLMError(RuntimeError):
    pass


def _payload(
    system: str,
    user: str,
    max_output_tokens: int | None,
    temperature: float | None,
    task: str = "default",
) -> dict:
    from ..model_routing import resolve_model
    return {
        "model": resolve_model(task),  # type: ignore[arg-type]
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature if temperature is not None else settings.llm_temperature,
        "max_tokens": max_output_tokens or settings.llm_max_output_tokens,
    }


def _headers() -> dict:
    if not settings.qwen_api_key:
        raise LLMError("DASHSCOPE_API_KEY not set")
    return {
        "Authorization": f"Bearer {settings.qwen_api_key}",
        "Content-Type": "application/json",
    }


def _should_cache(temperature: float | None) -> bool:
    t = temperature if temperature is not None else settings.llm_temperature
    return t <= 0.2


def _make_key(system: str, user: str, task: str = "default") -> str:
    from ..model_routing import resolve_model
    return "llm:" + stable_key(resolve_model(task), system, user)  # type: ignore[arg-type]


def _extract_content(data: dict) -> str:
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise LLMError(f"Unexpected LLM response: {json.dumps(data)[:300]}")


def _extract_usage(data: dict) -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens). Defaults to 0 if absent."""
    u = data.get("usage") or {}
    return int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


def _record_llm_metrics(task: str, data: dict, status: str = "ok") -> None:
    from ..observability import record_llm
    in_toks, out_toks = _extract_usage(data) if status == "ok" else (0, 0)
    record_llm(task=task, input_tokens=in_toks, output_tokens=out_toks, status=status)


def _preflight_budget_check(system: str, user: str) -> int:
    """Run before every LLM call. Raises BudgetExceeded if either cap is hit.

    Returns the prompt-token estimate so callers can attribute cost.
    """
    prompt_tokens = check_prompt_budget(system + "\n" + user)
    tenant = current_tenant_id() or settings.anonymous_tenant_id
    tenant_tracker().check(
        tenant_id=tenant,
        would_add=prompt_tokens + settings.max_output_tokens_per_request,
        limit=settings.tenant_token_budget,
        window_s=settings.tenant_budget_window_seconds,
    )
    return prompt_tokens


def _record_tenant_consumption(data: dict) -> None:
    """Record actual tokens consumed (prompt + completion) for the tenant."""
    tenant = current_tenant_id() or settings.anonymous_tenant_id
    in_toks, out_toks = _extract_usage(data)
    # Fallback to our char-based estimate if the provider didn't return usage.
    tenant_tracker().record(tenant_id=tenant, tokens=in_toks + out_toks)


# ---------- sync (existing callers) ----------

def chat_once(
    *,
    system: str,
    user: str,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    task: str = "default",
) -> str:
    """Sync Qwen call. Kept for non-async callers (rewrite, contextual, planner).

    ``task`` selects the model tier via model_routing when
    ``settings.enable_model_routing`` is True.
    """
    if settings.llm_provider != "qwen":
        raise LLMError(f"Unsupported LLM provider: {settings.llm_provider}")

    if _should_cache(temperature):
        try:
            hit = asyncio.run(cache_get(_make_key(system, user, task)))
            if isinstance(hit, str):
                return hit
        except RuntimeError:
            hit = None

    _preflight_budget_check(system, user)

    try:
        with httpx.Client(timeout=timeout or settings.llm_timeout_seconds) as client:
            resp = client.post(
                settings.qwen_chat_url,
                json=_payload(system, user, max_output_tokens, temperature, task),
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            _record_llm_metrics(task, data, status="ok")
            _record_tenant_consumption(data)
            content = _extract_content(data)
    except httpx.HTTPStatusError as e:
        _record_llm_metrics(task, {}, status="error")
        raise LLMError(f"LLM HTTP {e.response.status_code}: {e.response.text[:300]}") from e
    except httpx.RequestError as e:
        _record_llm_metrics(task, {}, status="error")
        raise LLMError(f"LLM network error: {e}") from e

    if _should_cache(temperature):
        try:
            asyncio.run(cache_set(_make_key(system, user, task), content, ttl_seconds=600))
        except RuntimeError:
            pass
    return content


# ---------- async ----------

async def chat_once_async(
    *,
    system: str,
    user: str,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    client: httpx.AsyncClient | None = None,
    task: str = "default",
) -> str:
    if settings.llm_provider != "qwen":
        raise LLMError(f"Unsupported LLM provider: {settings.llm_provider}")

    cache_key = None
    if _should_cache(temperature):
        cache_key = _make_key(system, user, task)
        hit = await cache_get(cache_key)
        if isinstance(hit, str):
            log.info("llm_cache_hit", key=cache_key[:16], task=task)
            return hit

    _preflight_budget_check(system, user)

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=timeout or settings.llm_timeout_seconds)
    from ..observability import get_tracer
    tracer = get_tracer("rag.llm")
    try:
        with tracer.start_as_current_span("llm.chat_once_async") as span:
            span.set_attribute("llm.task", task)
            span.set_attribute("llm.provider", settings.llm_provider)
            resp = await client.post(
                settings.qwen_chat_url,
                json=_payload(system, user, max_output_tokens, temperature, task),
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            in_toks, out_toks = _extract_usage(data)
            span.set_attribute("llm.tokens.input", in_toks)
            span.set_attribute("llm.tokens.output", out_toks)
            _record_llm_metrics(task, data, status="ok")
            _record_tenant_consumption(data)
            content = _extract_content(data)
    except httpx.HTTPStatusError as e:
        _record_llm_metrics(task, {}, status="error")
        raise LLMError(f"LLM HTTP {e.response.status_code}: {e.response.text[:300]}") from e
    except httpx.RequestError as e:
        _record_llm_metrics(task, {}, status="error")
        raise LLMError(f"LLM network error: {e}") from e
    finally:
        if owns_client:
            await client.aclose()

    if cache_key is not None:
        await cache_set(cache_key, content, ttl_seconds=600)
        log.info("llm_cache_store", key=cache_key[:16], task=task)
    return content


# ---------- streaming (sprint A.4) ----------

async def chat_stream_async(
    *,
    system: str,
    user: str,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    task: str = "default",
):
    """Yield output text deltas from Qwen as they arrive.

    Usage:
        async for delta in chat_stream_async(system=..., user=...):
            await ws.send(delta)

    Bypasses the LLM cache (there's no point caching a generator). Budget
    checks still run before the HTTP request. Token metrics are recorded
    at the end once we have the provider's final `usage` payload; if the
    stream ends without one (some providers), we fall back to our
    char-based estimate so dashboards are never empty.
    """
    if settings.llm_provider != "qwen":
        raise LLMError(f"Unsupported LLM provider: {settings.llm_provider}")

    _preflight_budget_check(system, user)

    payload = _payload(system, user, max_output_tokens, temperature, task)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}

    from ..observability import get_tracer
    tracer = get_tracer("rag.llm")
    final_usage: dict | None = None
    output_text_parts: list[str] = []

    try:
        with tracer.start_as_current_span("llm.chat_stream_async") as span:
            span.set_attribute("llm.task", task)
            span.set_attribute("llm.provider", settings.llm_provider)
            async with httpx.AsyncClient(timeout=timeout or settings.llm_timeout_seconds) as client:
                async with client.stream(
                    "POST",
                    settings.qwen_chat_url,
                    json=payload,
                    headers=_headers(),
                ) as resp:
                    resp.raise_for_status()
                    async for raw_line in resp.aiter_lines():
                        if not raw_line or not raw_line.startswith("data:"):
                            continue
                        line = raw_line[5:].strip()
                        if line == "[DONE]":
                            break
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        # Some providers emit a trailing usage-only frame.
                        if "usage" in chunk and chunk["usage"]:
                            final_usage = chunk["usage"]
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        piece = delta.get("content")
                        if piece:
                            output_text_parts.append(piece)
                            yield piece
    except httpx.HTTPStatusError as e:
        _record_llm_metrics(task, {}, status="error")
        raise LLMError(f"LLM HTTP {e.response.status_code}: {e.response.text[:300]}") from e
    except httpx.RequestError as e:
        _record_llm_metrics(task, {}, status="error")
        raise LLMError(f"LLM network error: {e}") from e

    # Usage reporting. Prefer provider numbers; fall back to estimates.
    if final_usage:
        data_like = {"usage": final_usage}
    else:
        data_like = {"usage": {
            "prompt_tokens": estimate_tokens(system + "\n" + user),
            "completion_tokens": estimate_tokens("".join(output_text_parts)),
        }}
    _record_llm_metrics(task, data_like, status="ok")
    _record_tenant_consumption(data_like)
