"""Async RAG pipeline smoke tests. Do not hit the LLM -- we mock the outbound
LLM call so the test is deterministic and free.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from rag import cache


@pytest.fixture(autouse=True)
def _reset_cache():
    cache.reset_for_tests()
    yield
    cache.reset_for_tests()


def test_chat_once_async_uses_cache():
    """Second call with identical prompt + low temperature hits the cache."""
    from rag.answer import llm_client

    calls = {"count": 0}

    class FakeResp:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            calls["count"] += 1
            return {"choices": [{"message": {"content": "cached-answer"}}]}

    class FakeClient:
        def __init__(self, *a, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            pass
        async def post(self, *a, **kw):
            return FakeResp()
        async def aclose(self):
            pass

    async def go():
        with patch.object(llm_client, "httpx", __import__("httpx")), \
             patch.object(llm_client.httpx, "AsyncClient", FakeClient):
            # set a dummy key so the client doesn't short-circuit on missing creds
            with patch.object(llm_client.settings, "qwen_api_key", "test-key"):
                r1 = await llm_client.chat_once_async(system="s", user="u", temperature=0.0)
                r2 = await llm_client.chat_once_async(system="s", user="u", temperature=0.0)
        return r1, r2, calls["count"]

    r1, r2, n = asyncio.run(go())
    assert r1 == "cached-answer"
    assert r2 == "cached-answer"
    assert n == 1, "second call should come from cache, not network"


def test_chat_once_async_high_temperature_not_cached():
    """Temperature > 0.2 disables caching (non-deterministic outputs)."""
    from rag.answer import llm_client

    calls = {"count": 0}

    class FakeResp:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            calls["count"] += 1
            return {"choices": [{"message": {"content": f"answer-{calls['count']}"}}]}

    class FakeClient:
        def __init__(self, *a, **kw):
            pass
        async def aclose(self):
            pass
        async def post(self, *a, **kw):
            return FakeResp()

    async def go():
        with patch.object(llm_client.httpx, "AsyncClient", FakeClient), \
             patch.object(llm_client.settings, "qwen_api_key", "test-key"):
            r1 = await llm_client.chat_once_async(system="s", user="u", temperature=0.9)
            r2 = await llm_client.chat_once_async(system="s", user="u", temperature=0.9)
        return r1, r2, calls["count"]

    r1, r2, n = asyncio.run(go())
    assert r1 != r2
    assert n == 2


def test_retrieve_async_parallelism(monkeypatch):
    """retrieve_async should kick off BM25 + vector concurrently, not serially."""
    import time

    from rag.retrieval import hybrid

    # Make both searches "take" 0.3s via asyncio.to_thread so parallel should be ~0.3s.
    class SlowStore:
        def search(self, query, limit):
            time.sleep(0.3)
            return []

    monkeypatch.setattr("rag.retrieval.hybrid.BM25Index", lambda path: SlowStore())
    monkeypatch.setattr("rag.retrieval.hybrid.get_vector_store", lambda *a, **kw: SlowStore())

    async def go():
        from pathlib import Path
        t0 = time.perf_counter()
        await hybrid.retrieve_async("q", Path("/tmp/irrelevant"), rerank=False, final_top_k=5)
        return time.perf_counter() - t0

    elapsed = asyncio.run(go())
    assert elapsed < 0.6, f"expected <0.6s for parallel (got {elapsed:.2f}s)"
