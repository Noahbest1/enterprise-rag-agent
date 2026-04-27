"""Hybrid retrieval: BM25 + vector fused by Reciprocal Rank Fusion,
then optional cross-encoder rerank.

Why RRF and not weighted score sum:
- BM25 and cosine scores live on different scales; calibrating weights per
  corpus is brittle. RRF only uses rank and is remarkably robust.
- BM25 wins on exact-term queries (version numbers, code), vector wins on
  paraphrases. RRF lets both contribute without one dominating by score
  magnitude.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from ..config import settings
from ..types import Hit
from ..index.bm25 import BM25Index
from ..index.vectorstore import get_vector_store
from .rerank import rerank_hits
from .source_prior import infer_source_type


def _hit_matches_filter(hit: Hit, filter: dict | None) -> bool:
    """Post-retrieval metadata filter.

    Supported keys (all optional, AND-combined):
        source_path_contains: substring match on hit.source_path
        title_contains:       substring match on hit.title
        source_type:          exact match, string or list of strings
                              (matched against infer_source_type(hit.source_path))
        section_contains:     substring match on " / ".join(hit.section_path)
        entities_contains:    list[str]; passes if ANY entity in hit.metadata["entities"]
                              case-insensitively equals or contains any of the given strings.
                              Requires metadata enrichment at ingest (sprint A.5).
        topics_contains:      list[str]; passes if ANY topic in hit.metadata["topics"]
                              case-insensitively matches (substring or exact).
        date_after / date_before: ISO-8601 YYYY-MM-DD; passes only when
                              hit.metadata["date"] falls in the range.
    """
    if not filter:
        return True
    path_contains = filter.get("source_path_contains")
    if path_contains and path_contains.lower() not in hit.source_path.lower():
        return False
    title_contains = filter.get("title_contains")
    if title_contains and title_contains.lower() not in hit.title.lower():
        return False
    sect_contains = filter.get("section_contains")
    if sect_contains:
        section = " / ".join(hit.section_path).lower()
        if sect_contains.lower() not in section:
            return False
    stype = filter.get("source_type")
    if stype:
        actual = infer_source_type(hit.source_path)
        if isinstance(stype, list):
            if actual not in stype:
                return False
        elif actual != stype:
            return False

    md = hit.metadata or {}

    ents_wanted = filter.get("entities_contains")
    if ents_wanted:
        have = [e.lower() for e in (md.get("entities") or []) if isinstance(e, str)]
        wanted = [w.lower() for w in (ents_wanted if isinstance(ents_wanted, list) else [ents_wanted])]
        # Passes if ANY wanted entity matches (substring match in either direction)
        if not any(any(w in h or h in w for h in have) for w in wanted):
            return False

    topics_wanted = filter.get("topics_contains")
    if topics_wanted:
        have = [t.lower() for t in (md.get("topics") or []) if isinstance(t, str)]
        wanted = [w.lower() for w in (topics_wanted if isinstance(topics_wanted, list) else [topics_wanted])]
        if not any(any(w in h or h in w for h in have) for w in wanted):
            return False

    hit_date = md.get("date") if isinstance(md.get("date"), str) else None
    date_after = filter.get("date_after")
    if date_after:
        if not hit_date or hit_date < date_after:
            return False
    date_before = filter.get("date_before")
    if date_before:
        if not hit_date or hit_date > date_before:
            return False

    return True


def rrf_fuse(rankings: list[list[Hit]], *, k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion. ``k`` dampens head dominance; 60 is the standard."""
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    sources: dict[str, list[str]] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            if hit.chunk_id not in best:
                best[hit.chunk_id] = hit
            sources.setdefault(hit.chunk_id, []).append(hit.retrieval_source)

    fused: list[Hit] = []
    for cid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        hit = best[cid]
        fused.append(
            Hit(
                chunk_id=hit.chunk_id,
                score=score,
                text=hit.text,
                title=hit.title,
                source_id=hit.source_id,
                source_path=hit.source_path,
                section_path=hit.section_path,
                retrieval_source="+".join(sorted(set(sources[cid]))),
                metadata=hit.metadata,
            )
        )
    return fused


def retrieve(
    query: str,
    kb_dir: Path,
    *,
    bm25_top_k: int | None = None,
    vector_top_k: int | None = None,
    rerank: bool = True,
    rerank_top_k: int | None = None,
    final_top_k: int | None = None,
    filter: dict | None = None,
) -> list[Hit]:
    bm25_top_k = bm25_top_k or settings.bm25_top_k
    vector_top_k = vector_top_k or settings.vector_top_k
    rerank_top_k = rerank_top_k or settings.rerank_top_k
    final_top_k = final_top_k or settings.final_top_k

    # Over-retrieve when filtering so we still have enough survivors.
    scale = 3 if filter else 1

    bm25 = BM25Index(kb_dir / "bm25.sqlite")
    # kb_id is the directory name, needed by Qdrant to pick the collection.
    vector = get_vector_store(kb_dir, kb_dir.name)

    bm25_hits = bm25.search(query, limit=bm25_top_k * scale)
    vector_hits = vector.search(query, limit=vector_top_k * scale)

    fused = rrf_fuse([bm25_hits, vector_hits], k=settings.rrf_k)
    if filter:
        fused = [h for h in fused if _hit_matches_filter(h, filter)]
    if not fused:
        return []

    if rerank:
        fused = rerank_hits(query, fused[:max(rerank_top_k, final_top_k)])

    if settings.use_mmr and len(fused) > final_top_k:
        from .mmr import apply_mmr
        fused = apply_mmr(fused, top_k=final_top_k)
        return fused

    return fused[:final_top_k]


async def retrieve_async(
    query: str,
    kb_dir: Path,
    *,
    bm25_top_k: int | None = None,
    vector_top_k: int | None = None,
    rerank: bool = True,
    rerank_top_k: int | None = None,
    final_top_k: int | None = None,
    filter: dict | None = None,
) -> list[Hit]:
    """Async variant: runs BM25 and vector search in parallel threads.

    BM25 hits SQLite (fast, blocking) and vector search runs BGE-M3 encoding
    + similarity (CPU-heavy, blocking). Both are offloaded via asyncio.to_thread
    so an async API handler can serve other requests while retrieval runs.
    """
    bm25_top_k = bm25_top_k or settings.bm25_top_k
    vector_top_k = vector_top_k or settings.vector_top_k
    rerank_top_k = rerank_top_k or settings.rerank_top_k
    final_top_k = final_top_k or settings.final_top_k

    scale = 3 if filter else 1

    bm25 = BM25Index(kb_dir / "bm25.sqlite")
    vector = get_vector_store(kb_dir, kb_dir.name)

    bm25_hits, vector_hits = await asyncio.gather(
        asyncio.to_thread(bm25.search, query, bm25_top_k * scale),
        asyncio.to_thread(vector.search, query, vector_top_k * scale),
    )

    fused = rrf_fuse([bm25_hits, vector_hits], k=settings.rrf_k)
    if filter:
        fused = [h for h in fused if _hit_matches_filter(h, filter)]
    if not fused:
        return []

    if rerank:
        fused = await asyncio.to_thread(
            rerank_hits, query, fused[:max(rerank_top_k, final_top_k)]
        )

    if settings.use_mmr and len(fused) > final_top_k:
        from .mmr import apply_mmr
        fused = await asyncio.to_thread(apply_mmr, fused, top_k=final_top_k)
        return fused

    return fused[:final_top_k]
