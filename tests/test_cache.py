"""Cache facade tests with in-process backend + fakeredis."""
from __future__ import annotations

import asyncio

import pytest

from rag import cache


@pytest.fixture(autouse=True)
def _reset_cache():
    cache.reset_for_tests()
    yield
    cache.reset_for_tests()


def test_roundtrip_string():
    async def go():
        await cache.cache_set("k1", "hello", ttl_seconds=60)
        assert await cache.cache_get("k1") == "hello"

    asyncio.run(go())


def test_roundtrip_json():
    async def go():
        await cache.cache_set("k2", {"a": 1, "b": [2, 3]}, ttl_seconds=60)
        assert await cache.cache_get("k2") == {"a": 1, "b": [2, 3]}

    asyncio.run(go())


def test_miss():
    async def go():
        assert await cache.cache_get("never-set") is None

    asyncio.run(go())


def test_ttl_expiry():
    async def go():
        await cache.cache_set("expires-fast", "boom", ttl_seconds=0)
        await asyncio.sleep(0.01)
        # local fallback stores exact expires_at; 0 TTL means immediately past.
        assert await cache.cache_get("expires-fast") is None

    asyncio.run(go())


def test_stable_key_deterministic():
    k1 = cache.stable_key("qwen-plus", "sys", "hello")
    k2 = cache.stable_key("qwen-plus", "sys", "hello")
    assert k1 == k2
    assert k1 != cache.stable_key("qwen-plus", "sys", "goodbye")


def test_clear_prefix():
    async def go():
        await cache.cache_set("llm:a", "x", 60)
        await cache.cache_set("llm:b", "y", 60)
        await cache.cache_set("other:c", "z", 60)
        dropped = await cache.cache_clear("llm:")
        assert dropped == 2
        assert await cache.cache_get("llm:a") is None
        assert await cache.cache_get("other:c") == "z"

    asyncio.run(go())


def test_fakeredis_backend_works():
    """If fakeredis is installed, we can plug it in as the backend."""
    try:
        import fakeredis.aioredis as fake_aio
    except ImportError:
        pytest.skip("fakeredis not installed")

    class _FakeBackend:
        def __init__(self):
            self.c = fake_aio.FakeRedis(decode_responses=True)

        async def get(self, key):
            return await self.c.get(key)

        async def set(self, key, value, ttl):
            await self.c.set(key, value, ex=ttl)

        async def clear(self, prefix=None):
            if prefix is None:
                await self.c.flushdb()
                return 0
            n = 0
            async for k in self.c.scan_iter(match=f"{prefix}*"):
                await self.c.delete(k)
                n += 1
            return n

    cache.reset_for_tests(backend=_FakeBackend(), name="fakeredis")
    assert cache.current_backend_name() == "fakeredis"

    async def go():
        await cache.cache_set("k", "v", 30)
        assert await cache.cache_get("k") == "v"

    asyncio.run(go())
