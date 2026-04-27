"""Sprint A — semantic chunking tests.

Mocks BGE-M3 via monkeypatch on ``embed_texts`` so the test suite stays fast
(real encoder is covered by the retrieval eval).

Covers:
- split_sentences: CJK + ASCII terminators, paragraph breaks, empty input
- _cosine_distances: basic math + short-input degenerate cases
- semantic_split: single-sentence passthrough, high-percentile cuts,
  min_tokens greedy merge, max_tokens hard-cut
"""
from __future__ import annotations

import numpy as np
import pytest

from rag.ingest.semantic_chunking import (
    _cosine_distances,
    semantic_split,
    split_sentences,
)


# ---------- split_sentences ----------

def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   \n ") == []


def test_split_sentences_single():
    assert split_sentences("hello world") == ["hello world"]


def test_split_sentences_ascii_terminators():
    out = split_sentences("First. Second! Third?")
    assert out == ["First.", "Second!", "Third?"]


def test_split_sentences_cjk_terminators():
    out = split_sentences("今天天气好。明天也好!后天如何?")
    assert out == ["今天天气好。", "明天也好!", "后天如何?"]


def test_split_sentences_paragraph_breaks():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird."
    out = split_sentences(text)
    assert len(out) == 3


# ---------- _cosine_distances ----------

def test_cosine_distances_two_identical():
    v = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    d = _cosine_distances(v)
    assert d.shape == (1,)
    assert d[0] == pytest.approx(0.0, abs=1e-6)


def test_cosine_distances_two_orthogonal():
    v = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    d = _cosine_distances(v)
    assert d[0] == pytest.approx(1.0, abs=1e-6)


def test_cosine_distances_short_input():
    v = np.array([[1.0, 0.0]], dtype=np.float32)
    assert _cosine_distances(v).size == 0


# ---------- semantic_split ----------

def _unit(i: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(i)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) or 1.0)


def test_semantic_split_single_sentence_passthrough(monkeypatch):
    # Only one sentence -> no need to embed; we never need to call the mock.
    def _unused(texts, **kw):  # noqa: ARG001
        raise AssertionError("embed_texts should not be called for a single sentence")
    monkeypatch.setattr("rag.ingest.semantic_chunking.embed_texts", _unused)
    out = semantic_split("Only one sentence here.", min_tokens=0, max_tokens=500)
    assert out == ["Only one sentence here."]


def test_semantic_split_empty(monkeypatch):
    out = semantic_split("", min_tokens=0, max_tokens=500)
    assert out == []


def test_semantic_split_cuts_at_topic_shift(monkeypatch):
    # Three sentences: first two about topic A, third about topic B.
    # Provide vectors where A and B are orthogonal so the second distance
    # is >>> the first and gets flagged by the 95th-percentile rule.
    def _fake_embed(texts, **kw):  # noqa: ARG001
        # texts[0], texts[1] -> same vector; texts[2] -> orthogonal
        return np.array([_unit(1), _unit(1), _unit(2)], dtype=np.float32)
    monkeypatch.setattr("rag.ingest.semantic_chunking.embed_texts", _fake_embed)

    text = "Topic A part one. Topic A part two. Totally new topic B."
    chunks = semantic_split(text, min_tokens=0, max_tokens=500, breakpoint_percentile=50)
    # Expect 2 chunks (split between sentence 2 and 3).
    assert len(chunks) == 2
    assert "Topic A part one." in chunks[0]
    assert "Topic A part two." in chunks[0]
    assert "Totally new topic B." in chunks[1]


def test_semantic_split_respects_min_tokens(monkeypatch):
    # All four sentences are topically distinct (would normally yield 4 chunks),
    # but min_tokens is high so they merge into 1-2 chunks.
    def _fake_embed(texts, **kw):  # noqa: ARG001
        return np.array([_unit(i) for i in range(len(texts))], dtype=np.float32)
    monkeypatch.setattr("rag.ingest.semantic_chunking.embed_texts", _fake_embed)

    text = "A a a a a. B b b b b. C c c c c. D d d d d."
    chunks = semantic_split(text, min_tokens=50, max_tokens=500, breakpoint_percentile=0)
    # With min_tokens>>sentence_tokens, the greedy pass should merge everything.
    assert len(chunks) <= 2


def test_semantic_split_hard_cut_on_oversized(monkeypatch):
    def _fake_embed(texts, **kw):  # noqa: ARG001
        return np.array([_unit(1)] * len(texts), dtype=np.float32)
    monkeypatch.setattr("rag.ingest.semantic_chunking.embed_texts", _fake_embed)

    # A single giant run that would exceed max_tokens must be broken up.
    long_text = ". ".join([f"Sentence number {i} with enough filler words" for i in range(40)])
    chunks = semantic_split(long_text, min_tokens=0, max_tokens=30, breakpoint_percentile=50)
    assert len(chunks) > 1
    # Every chunk should be at most 2x the cap (hard-cut is sentence-boundaried)
    from rag.ingest.chunking import approx_tokens
    for c in chunks:
        assert approx_tokens(c) <= 60  # generous upper bound
