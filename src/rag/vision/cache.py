"""VLM / OCR result cache keyed by pixel hash.

Same backend as the rest of the project (Redis with in-process fallback),
just a separate key prefix so we can flush vision results independently
if a new model version changes outputs.

``cache_vision_result`` is a decorator-style wrapper; call it with a
function that takes ``(processed_image, **kwargs)`` and returns a dict.
On cache hit, the dict is deserialised and returned. On miss, the
function runs, its result is cached, and returned.

TTL defaults to 24 hours. Images are usually immutable (hashing pixels),
so long TTL is safe; a new model rev triggers a prefix bump.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable

from ..cache import cache_get, cache_set
from ..logging import get_logger


log = get_logger(__name__)

_PREFIX = "vision:v1"
DEFAULT_TTL_S = 24 * 3600


def _key(pixel_hash: str, task: str) -> str:
    return f"{_PREFIX}:{task}:{pixel_hash}"


def cached_vision_call(
    pixel_hash: str,
    task: str,
    runner: Callable[[], Any],
    *,
    ttl: int = DEFAULT_TTL_S,
) -> Any:
    """Sync helper. Runs inside an asyncio.run so callers in scripts work."""
    key = _key(pixel_hash, task)
    try:
        hit = asyncio.run(cache_get(key))
        if hit is not None:
            log.info("vision_cache_hit", key=key[:32], task=task)
            return hit
    except RuntimeError:
        hit = None

    result = runner()
    try:
        asyncio.run(cache_set(key, result, ttl_seconds=ttl))
        log.info("vision_cache_store", key=key[:32], task=task)
    except RuntimeError:
        pass
    return result


async def cached_vision_call_async(
    pixel_hash: str,
    task: str,
    runner: Callable[[], Any],
    *,
    ttl: int = DEFAULT_TTL_S,
) -> Any:
    """Async variant for FastAPI request handlers."""
    key = _key(pixel_hash, task)
    hit = await cache_get(key)
    if hit is not None:
        log.info("vision_cache_hit", key=key[:32], task=task)
        return hit
    result = runner() if not asyncio.iscoroutinefunction(runner) else await runner()
    await cache_set(key, result, ttl_seconds=ttl)
    log.info("vision_cache_store", key=key[:32], task=task)
    return result
