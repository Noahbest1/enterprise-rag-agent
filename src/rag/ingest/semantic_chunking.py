"""Embedding-distance based semantic chunking.

Algorithm (Greg Kamradt, 2024):
    1. Split text into sentences.
    2. Embed each sentence (BGE-M3, the same encoder the retrieval index uses).
    3. Compute cosine distance between each adjacent pair.
    4. Cut wherever the distance exceeds a high percentile (default 95th) --
       that's where the topic shifts most.
    5. Merge neighbouring cuts so no chunk falls below ``min_tokens`` and none
       exceeds ``max_tokens`` (greedy pack).

Why this instead of LLM-assisted boundary detection:
    - Zero LLM calls. Embedding is already happening in the pipeline.
    - Deterministic (no temperature wobble).
    - Plays well with the existing BGE-M3 vectors -- same space, no drift.

What this DOES NOT replace:
    - Structural chunking (`chunk_document`) still owns the parent-child
      hierarchy. Semantic chunking is a sub-splitter for oversized leaf
      sections: hand it a blob, get back balanced sub-chunks.
    - Opt-in: ``settings.use_semantic_chunking`` (default False) because
      embedding every sentence slows down ingest 2-4x on text-heavy KBs.
"""
from __future__ import annotations

import re

import numpy as np

from ..index.faiss_store import embed_texts
from ..ingest.chunking import approx_tokens


# Sentence splitter. CJK terminators (。!?;) and paragraph breaks split with
# or without trailing whitespace; ASCII (.!?;) requires trailing whitespace so
# decimal numbers ("3.14") don't get mid-split.
_SENT_SPLIT = re.compile(r"(?<=[。!?;])|(?<=[.!?;])\s+|\n{2,}")


def split_sentences(text: str) -> list[str]:
    """Return non-empty sentences. Short paragraphs count as one sentence each."""
    if not text or not text.strip():
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def _cosine_distances(vecs: np.ndarray) -> np.ndarray:
    """Distance between vecs[i] and vecs[i+1]. Inputs assumed L2-normalised."""
    if len(vecs) < 2:
        return np.zeros(0, dtype=np.float32)
    sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
    sims = np.clip(sims, -1.0, 1.0)
    return (1.0 - sims).astype(np.float32)


def semantic_split(
    text: str,
    *,
    min_tokens: int = 150,
    max_tokens: int = 800,
    breakpoint_percentile: float = 95.0,
) -> list[str]:
    """Split ``text`` into semantically coherent chunks.

    Guarantees (soft):
        - each chunk has >= ``min_tokens`` except possibly the last one
        - no chunk exceeds ``max_tokens``; oversized survivors are hard-cut
          on sentence boundaries

    Args:
        breakpoint_percentile: higher = fewer cuts. 95 is a sensible default;
            drop to 85-90 on dense technical text if chunks feel too large.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    # One embed call for the whole doc -- BGE-M3 batches internally.
    vecs = embed_texts(sentences, batch_size=32)
    distances = _cosine_distances(vecs)
    if distances.size == 0:
        return [" ".join(sentences)]

    threshold = float(np.percentile(distances, breakpoint_percentile))

    # First pass: cut at every spike above the threshold.
    groups: list[list[str]] = []
    current: list[str] = [sentences[0]]
    for i, d in enumerate(distances):
        # d compares sentences[i] vs sentences[i+1]
        if d >= threshold:
            groups.append(current)
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])
    groups.append(current)

    # Second pass: enforce min/max tokens by greedy packing.
    packed: list[list[str]] = []
    buffer: list[str] = []
    buf_tokens = 0
    for g in groups:
        g_tokens = sum(approx_tokens(s) for s in g)
        # If the current buffer is below min AND adding g still fits in max, merge.
        if buf_tokens < min_tokens and buf_tokens + g_tokens <= max_tokens:
            buffer.extend(g)
            buf_tokens += g_tokens
            continue
        if buffer:
            packed.append(buffer)
        # g might itself exceed max_tokens -- hard-cut it.
        if g_tokens > max_tokens:
            for hard in _hard_cut(g, max_tokens):
                packed.append(hard)
            buffer = []
            buf_tokens = 0
        else:
            buffer = list(g)
            buf_tokens = g_tokens
    if buffer:
        packed.append(buffer)

    return [" ".join(block) for block in packed]


def _hard_cut(sentences: list[str], max_tokens: int) -> list[list[str]]:
    """Fall-back packer when a semantic group itself exceeds max_tokens."""
    out: list[list[str]] = []
    cur: list[str] = []
    cur_tok = 0
    for s in sentences:
        t = approx_tokens(s)
        if cur_tok + t > max_tokens and cur:
            out.append(cur)
            cur = [s]
            cur_tok = t
        else:
            cur.append(s)
            cur_tok += t
    if cur:
        out.append(cur)
    return out
