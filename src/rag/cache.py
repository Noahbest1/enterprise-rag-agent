"""Async cache with Redis primary + in-process TTL fallback.

Decisions:
- Fall back to an in-process TTL dict when Redis is unreachable. Dev laptops
  without Docker can keep working.
- Expose only ``cache_get`` / ``cache_set`` / ``cache_clear``. Callers should
  not see which backend is in use.
- Keys are arbitrary strings; values are serialised JSON so the on-disk
  storage is human-readable and TTL-truncated values don't leak binaries.
- Thread/async-safe. Redis client is lazily created once per process.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from .config import settings
from .logging import get_logger


log = get_logger(__name__)


# ---------------- in-process fallback ----------------

@dataclass
class _LocalEntry:
    value: str
    expires_at: float


class _LocalCache:
    """Tiny LRU-free TTL store; good enough for dev + tests."""

    def __init__(self, max_size: int = 5_000):
        self._store: dict[str, _LocalEntry] = {}
        self._lock = asyncio.Lock()
        self._max = max_size

    async def get(self, key: str) -> str | None:
        async with self._lock:
            e = self._store.get(key)
            if not e:
                return None
            if e.expires_at < time.time():
                self._store.pop(key, None)
                return None
            return e.value

    async def set(self, key: str, value: str, ttl: int) -> None:
        async with self._lock:
            if len(self._store) >= self._max:
                # drop the earliest-expiring entry -- O(n) but n is small
                oldest = min(self._store, key=lambda k: self._store[k].expires_at)
                self._store.pop(oldest, None)
            self._store[key] = _LocalEntry(value=value, expires_at=time.time() + ttl)

    async def clear(self, prefix: str | None = None) -> int:
        async with self._lock:
            if prefix is None:
                n = len(self._store)
                self._store.clear()
                return n
            to_drop = [k for k in self._store if k.startswith(prefix)]
            for k in to_drop:
                self._store.pop(k, None)
            return len(to_drop)


# ---------------- Redis backend ----------------

class _RedisCache:
    def __init__(self, url: str):
        self._url = url
        self._client = None  # lazy

    async def _get_client(self):
        if self._client is None:
            try:
                import redis.asyncio as aioredis
            except ImportError as e:
                raise RuntimeError("redis package not installed") from e
            self._client = aioredis.from_url(self._url, decode_responses=True)
            # Ping to make sure we can actually reach it; raise if not.
            await self._client.ping()
        return self._client

    async def get(self, key: str) -> str | None:
        c = await self._get_client()
        return await c.get(key)

    async def set(self, key: str, value: str, ttl: int) -> None:
        c = await self._get_client()
        await c.set(key, value, ex=ttl)

    async def clear(self, prefix: str | None = None) -> int:
        c = await self._get_client()
        if prefix is None:
            return await c.flushdb()
        n = 0
        async for key in c.scan_iter(match=f"{prefix}*"):
            await c.delete(key)
            n += 1
        return n


# ---------------- facade ----------------

_backend: Any | None = None
_backend_name: str = "unknown"
_init_lock = asyncio.Lock()


async def _ensure_backend():
    global _backend, _backend_name
    if _backend is not None:
        return
    async with _init_lock:
        if _backend is not None:
            return
        if settings.redis_url and not settings.redis_url.startswith(("memory:", "fake:")):
            try:
                redis_backend = _RedisCache(settings.redis_url)
                # Prime the connection so we fail fast
                await redis_backend._get_client()
                _backend = redis_backend
                _backend_name = "redis"
                log.info("cache_backend_selected", backend="redis", url=settings.redis_url)
                return
            except Exception as e:
                log.warning("cache_redis_unavailable", url=settings.redis_url, error=str(e))
        _backend = _LocalCache()
        _backend_name = "in_process"
        log.info("cache_backend_selected", backend="in_process", reason="redis_unavailable_or_disabled")


def _layer_from_key(key: str) -> str:
    """Infer cache layer from the key prefix so metrics can break it down."""
    if key.startswith("emb:"):
        return "embedding"
    if key.startswith("llm:"):
        return "llm"
    if key.startswith("sem") or key.startswith("semcache"):
        return "semantic"
    if key.startswith("vision"):
        return "vision"
    return "other"


async def cache_get(key: str) -> Any | None:
    await _ensure_backend()
    raw = await _backend.get(key)
    # Record hit/miss for observability. Done at the facade layer so every
    # caller (llm_client, semantic_cache, vision, embedding) is covered
    # without each of them having to instrument separately.
    from .observability import record_cache
    record_cache(layer=_layer_from_key(key), hit=raw is not None)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


async def cache_set(key: str, value: Any, ttl_seconds: int = 600) -> None:
    await _ensure_backend()
    payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    await _backend.set(key, payload, ttl=ttl_seconds)


async def cache_clear(prefix: str | None = None) -> int:
    await _ensure_backend()
    return await _backend.clear(prefix)


def current_backend_name() -> str:
    return _backend_name


def reset_for_tests(backend: Any | None = None, name: str = "test") -> None:
    """Wire in a test double (e.g. fakeredis-backed client)."""
    global _backend, _backend_name
    _backend = backend or _LocalCache()
    _backend_name = name


def stable_key(*parts: Any) -> str:
    """Hash args into a short stable cache key."""
    material = "|".join(str(p) for p in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
