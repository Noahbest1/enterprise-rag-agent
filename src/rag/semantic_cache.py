"""Semantic answer cache.

Standard string-hash caches miss when users rephrase the same question:
"PLUS 年卡多少钱" vs "PLUS 会员一年费用" → same answer, different hash.

The semantic cache stores past (normalised query, embedding, answer) tuples
per KB and, for each new query, does a linear cosine-similarity search. If
similarity ≥ ``similarity_threshold`` the cached answer is returned instead
of re-running retrieval + generation.

Implementation choices:
- Linear scan over an in-memory deque (per KB). Keeps code trivial and, at
  ~500 entries per KB, scan is sub-millisecond.
- Entries also roundtrip through Redis (if available) so a process restart
  doesn't cold-start the cache. Redis stores a JSON blob per KB.
- Only cache answers that are ``not abstained``. Abstain answers are
  common and caching them would punish future improvements.
- TTL is long (2 hours) because the underlying KB changes slowly. Bump or
  invalidate when rebuilding a KB.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cache import cache_get, cache_set
from .config import settings
from .logging import get_logger


log = get_logger(__name__)


@dataclass
class _Entry:
    query: str
    embedding: list[float]
    answer: dict[str, Any]  # serialised Answer
    created_at: float


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.95, max_entries_per_kb: int = 500):
        self.threshold = similarity_threshold
        self.max_entries = max_entries_per_kb
        self._per_kb: dict[str, deque[_Entry]] = {}
        self._lock = asyncio.Lock()
        self._loaded: set[str] = set()

    # ---------- persistence (Redis) ----------

    @staticmethod
    def _redis_key(kb_id: str) -> str:
        return f"semcache:{kb_id}"

    async def _maybe_load(self, kb_id: str) -> None:
        if kb_id in self._loaded:
            return
        try:
            raw = await cache_get(self._redis_key(kb_id))
            if isinstance(raw, list):
                self._per_kb[kb_id] = deque(
                    (_Entry(**e) for e in raw),
                    maxlen=self.max_entries,
                )
        except Exception as e:
            log.warning("semcache_load_failed", kb_id=kb_id, error=str(e))
        finally:
            self._loaded.add(kb_id)

    async def _persist(self, kb_id: str) -> None:
        entries = list(self._per_kb.get(kb_id, ()))
        try:
            await cache_set(
                self._redis_key(kb_id),
                [
                    {
                        "query": e.query,
                        "embedding": e.embedding,
                        "answer": e.answer,
                        "created_at": e.created_at,
                    }
                    for e in entries
                ],
                ttl_seconds=7200,
            )
        except Exception as e:
            log.warning("semcache_persist_failed", kb_id=kb_id, error=str(e))

    # ---------- API ----------

    async def lookup(self, kb_id: str, query_embedding: np.ndarray) -> tuple[dict | None, float, str]:
        """Return (cached_answer_or_None, best_similarity, best_query).

        Caller decides whether to use the result based on similarity.
        """
        await self._maybe_load(kb_id)
        entries = self._per_kb.get(kb_id)
        if not entries:
            return None, 0.0, ""

        qv = query_embedding.flatten()
        qn = np.linalg.norm(qv) or 1.0

        best_sim = -1.0
        best_entry: _Entry | None = None
        for e in entries:
            ev = np.asarray(e.embedding, dtype=np.float32)
            en = np.linalg.norm(ev) or 1.0
            sim = float(np.dot(qv, ev) / (qn * en))
            if sim > best_sim:
                best_sim = sim
                best_entry = e

        if best_entry is not None and best_sim >= self.threshold:
            return best_entry.answer, best_sim, best_entry.query
        return None, best_sim, best_entry.query if best_entry else ""

    async def store(self, kb_id: str, query: str, query_embedding: np.ndarray, answer: dict) -> None:
        await self._maybe_load(kb_id)
        dq = self._per_kb.setdefault(kb_id, deque(maxlen=self.max_entries))
        dq.append(
            _Entry(
                query=query,
                embedding=query_embedding.flatten().tolist(),
                answer=answer,
                created_at=time.time(),
            )
        )
        await self._persist(kb_id)

    async def invalidate(self, kb_id: str) -> None:
        self._per_kb.pop(kb_id, None)
        self._loaded.discard(kb_id)
        try:
            await cache_set(self._redis_key(kb_id), [], ttl_seconds=1)
        except Exception:
            pass

    def reset_for_tests(self) -> None:
        self._per_kb.clear()
        self._loaded.clear()


# Module-level singleton so all callers share one cache across a process.
_singleton: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    global _singleton
    if _singleton is None:
        _singleton = SemanticCache(similarity_threshold=0.95)
    return _singleton
