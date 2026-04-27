"""Cross-encoder rerank with BGE-reranker-base.

Score is normalized to [0, 1] via sigmoid so it's directly comparable to
the abstain threshold in config.
"""
from __future__ import annotations

import math
import threading
from functools import lru_cache

from ..config import settings
from ..types import Hit
from .source_prior import score_prior


_rerank_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(settings.reranker_model)


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def rerank_hits(query: str, hits: list[Hit]) -> list[Hit]:
    if not hits:
        return []
    pairs = [(query, f"{h.title}\n{h.text}") for h in hits]
    with _rerank_lock:
        scores = get_reranker().predict(pairs, show_progress_bar=False)

    reranked: list[Hit] = []
    for hit, raw in zip(hits, scores):
        normalized = _sigmoid(float(raw))
        prior = score_prior(query, hit.source_path)
        # Clamp so a prior can't push a very-low-confidence chunk past others.
        adjusted = max(0.0, min(1.0, normalized + prior))
        reranked.append(
            Hit(
                chunk_id=hit.chunk_id,
                score=adjusted,
                text=hit.text,
                title=hit.title,
                source_id=hit.source_id,
                source_path=hit.source_path,
                section_path=hit.section_path,
                retrieval_source=f"{hit.retrieval_source}+rerank",
                metadata={
                    **hit.metadata,
                    "rerank_raw": float(raw),
                    "rerank_sigmoid": normalized,
                    "source_prior": prior,
                },
            )
        )
    reranked.sort(key=lambda h: h.score, reverse=True)
    return reranked
