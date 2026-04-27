"""Semantic cache correctness + threshold behaviour."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from rag import cache
from rag.semantic_cache import SemanticCache


@pytest.fixture(autouse=True)
def _reset_cache():
    cache.reset_for_tests()
    yield
    cache.reset_for_tests()


def test_empty_cache_miss():
    sc = SemanticCache()
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    hit, sim, best_q = asyncio.run(sc.lookup("kb1", v))
    assert hit is None
    assert sim == 0.0
    assert best_q == ""


def test_exact_match_returns_cached_answer():
    sc = SemanticCache(similarity_threshold=0.9)
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    asyncio.run(sc.store("kb1", "q1", v, {"text": "cached answer"}))
    hit, sim, best_q = asyncio.run(sc.lookup("kb1", v))
    assert hit is not None
    assert hit["text"] == "cached answer"
    assert sim > 0.99
    assert best_q == "q1"


def test_similar_query_above_threshold_hits():
    sc = SemanticCache(similarity_threshold=0.95)
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    asyncio.run(sc.store("kb1", "q1", v1, {"text": "A"}))
    v2 = np.array([0.98, 0.2, 0.0], dtype=np.float32)  # cos ≈ 0.98
    hit, sim, _ = asyncio.run(sc.lookup("kb1", v2))
    assert hit is not None
    assert sim >= 0.95


def test_dissimilar_query_below_threshold_misses():
    sc = SemanticCache(similarity_threshold=0.95)
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    asyncio.run(sc.store("kb1", "q1", v1, {"text": "A"}))
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # orthogonal
    hit, sim, _ = asyncio.run(sc.lookup("kb1", v2))
    assert hit is None
    assert sim < 0.95


def test_per_kb_isolation():
    sc = SemanticCache(similarity_threshold=0.9)
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    asyncio.run(sc.store("kb_a", "q", v, {"text": "A-only"}))
    hit_a, _, _ = asyncio.run(sc.lookup("kb_a", v))
    hit_b, _, _ = asyncio.run(sc.lookup("kb_b", v))
    assert hit_a is not None
    assert hit_b is None


def test_eviction_keeps_recent():
    sc = SemanticCache(similarity_threshold=0.5, max_entries_per_kb=3)
    for i in range(5):
        v = np.array([1.0, float(i), 0.0], dtype=np.float32)
        asyncio.run(sc.store("kb1", f"q{i}", v, {"text": f"ans{i}"}))
    assert len(sc._per_kb["kb1"]) == 3
    # the oldest three (q0, q1) should have been evicted
    stored_queries = [e.query for e in sc._per_kb["kb1"]]
    assert "q4" in stored_queries
