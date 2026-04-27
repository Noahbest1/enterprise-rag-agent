"""Sprint A.1 MMR diversification tests.

Covers:
- _jaccard / _shingles unit behaviour
- Top of list unchanged when all candidates are diverse
- Duplicates deprioritised in favour of a fresh candidate
- lambda=1 reduces to pure relevance; lambda=0 favours diversity only
- Returns hits unchanged when top_k >= len(hits)
"""
from __future__ import annotations

import pytest

from rag.retrieval.mmr import _jaccard, _shingles, apply_mmr
from rag.types import Hit


def _hit(i: int, text: str, score: float) -> Hit:
    return Hit(
        chunk_id=f"c{i}",
        score=score,
        text=text,
        title=f"t{i}",
        source_id=f"s{i}",
        source_path=f"/p{i}",
        section_path=[],
        retrieval_source="test",
    )


def test_shingles_short_text():
    assert _shingles("abcd", n=5) == {"abcd"}
    assert _shingles("", n=5) == set()


def test_shingles_basic():
    s = _shingles("hello world", n=5)
    assert "hello" in s
    assert " worl" in s
    assert "world" in s


def test_jaccard_identical_is_1():
    s = _shingles("hello world hello world", n=5)
    assert _jaccard(s, s) == 1.0


def test_jaccard_disjoint_is_0():
    a = _shingles("abcdefghij", n=5)
    b = _shingles("zyxwvutsrq", n=5)
    assert _jaccard(a, b) == 0.0


def test_mmr_short_circuits_when_top_k_covers_all():
    hits = [_hit(0, "foo", 0.9), _hit(1, "bar", 0.8)]
    result = apply_mmr(hits, top_k=5)
    assert result == hits  # top_k >= len, no re-ordering


def test_mmr_lambda_1_is_pure_relevance():
    hits = [
        _hit(0, "京东 PLUS 会员价格", 0.9),
        _hit(1, "京东 PLUS 会员价格", 0.85),  # near-dup
        _hit(2, "淘宝 88VIP 开通条件完全不同", 0.6),
    ]
    result = apply_mmr(hits, top_k=2, lambda_=1.0)
    # Pure relevance -> top 2 by score
    assert [h.chunk_id for h in result] == ["c0", "c1"]


def test_mmr_lambda_0_point_5_diversifies():
    # With lambda<1, a near-dup of the top hit should be deprioritised.
    hits = [
        _hit(0, "京东 PLUS 会员价格是 148 元一年", 0.9),
        _hit(1, "京东 PLUS 会员价格是 148 元一年", 0.85),  # near-dup of hit 0
        _hit(2, "淘宝 88VIP 开通条件完全不同另一个话题", 0.6),
    ]
    result = apply_mmr(hits, top_k=2, lambda_=0.5)
    ids = [h.chunk_id for h in result]
    # First pick is still the highest relevance (c0).
    assert ids[0] == "c0"
    # Second should be c2 (diverse) not c1 (near-dup), even though c1 has higher raw score.
    assert ids[1] == "c2"


def test_mmr_preserves_order_within_selections():
    # Diverse hits -> order should roughly follow score.
    hits = [
        _hit(0, "apple orange banana grape", 0.9),
        _hit(1, "京东 PLUS 会员 月卡 季卡 年卡", 0.8),
        _hit(2, "淘宝 88VIP 天猫 折扣 优惠券", 0.7),
    ]
    result = apply_mmr(hits, top_k=3, lambda_=0.7)
    assert result[0].chunk_id == "c0"  # top relevance picked first
    assert len(result) == 3
    # All three diverse -> MMR should include all.
    assert set(h.chunk_id for h in result) == {"c0", "c1", "c2"}


def test_mmr_empty_list():
    assert apply_mmr([], top_k=5) == []
