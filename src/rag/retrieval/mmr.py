"""Maximal Marginal Relevance (MMR) diversification.

After rerank, hits are ordered by relevance alone. With large KBs and
paraphrased sources (e.g. the same policy repeated across 3 FAQ pages),
the top-5 can end up near-duplicates -- the LLM sees the same fact 3 times
and misses wider context.

MMR rebalances:

    score(h) = lambda * relevance(h) - (1 - lambda) * max_sim(h, selected)

Where ``sim`` is the content similarity between ``h`` and any already-
selected hit. We use Jaccard over 5-gram character shingles -- same
shingle function as PH2's dedup, so the cost is just a hash-set union.
No extra embedding call, no extra LLM round trip.

``lambda`` is tunable via ``settings.mmr_lambda`` (default 0.7 -- still
leans toward relevance, just penalises duplicates).
"""
from __future__ import annotations

from ..config import settings
from ..types import Hit


def _shingles(text: str, n: int = 5) -> set[str]:
    """Character n-gram shingles. Matches PH2 dedup conventions."""
    s = text.strip()
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def apply_mmr(
    hits: list[Hit],
    *,
    top_k: int,
    lambda_: float | None = None,
) -> list[Hit]:
    """Re-order ``hits`` using MMR; return the first ``top_k``.

    Preserves hit order inside the returned list: position = MMR selection
    step. Relevance is taken from ``hit.score`` directly, so MMR only
    makes sense AFTER rerank (when ``score`` is calibrated).
    """
    if not hits:
        return []
    if top_k >= len(hits):
        # No room to diversify; short-circuit.
        return hits

    lam = lambda_ if lambda_ is not None else settings.mmr_lambda

    # Normalise relevance to [0,1] so it's on the same scale as Jaccard.
    scores = [h.score for h in hits]
    lo, hi = min(scores), max(scores)
    span = hi - lo if hi > lo else 1.0

    shingles = [_shingles(h.text) for h in hits]
    remaining = list(range(len(hits)))
    selected: list[int] = []

    # Greedy: pick the most-relevant first, then each subsequent pick
    # maximises (lambda * rel - (1-lambda) * max_sim_to_selected).
    first = max(remaining, key=lambda i: hits[i].score)
    selected.append(first)
    remaining.remove(first)

    while remaining and len(selected) < top_k:
        best_idx = None
        best_mmr = -float("inf")
        for i in remaining:
            rel = (hits[i].score - lo) / span
            max_sim = max(_jaccard(shingles[i], shingles[j]) for j in selected)
            mmr = lam * rel - (1 - lam) * max_sim
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i
        selected.append(best_idx)  # type: ignore[arg-type]
        remaining.remove(best_idx)

    return [hits[i] for i in selected]
