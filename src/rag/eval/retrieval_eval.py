"""Retrieval-layer metrics: Hit@k, MRR, Recall@k.

Input: list of eval rows, each with
    query:            the user query
    kb_id:            which KB to search
    relevant_source_ids: list of source_ids considered correct answers

A hit matches if any retrieved chunk's source_id appears in relevant_source_ids.
Source-level (not chunk-level) matching is more forgiving and matches how
end-users experience the system: "did it find the right document?"
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..config import settings
from ..knowledge_base import get_kb
from ..query.multi_query import expand_queries
from ..query.normalize import normalize_query
from ..query.rewrite import rewrite_query
from ..retrieval.hybrid import retrieve, rrf_fuse
from ..retrieval.rerank import rerank_hits


@dataclass
class RetrievalRowResult:
    query: str
    kb_id: str
    relevant_source_ids: list[str]
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    reciprocal_rank: float
    retrieved_source_ids: list[str]
    category: str = "direct"
    scored: bool = True  # False for out_of_domain / empty-relevant rows


@dataclass
class CategoryMetrics:
    n: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr_at_10: float


@dataclass
class RetrievalMetrics:
    n: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    mrr_at_10: float
    per_row: list[RetrievalRowResult] = field(default_factory=list)
    by_category: dict[str, CategoryMetrics] = field(default_factory=dict)
    n_skipped: int = 0  # out_of_domain rows not included in Hit@k/MRR

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _first_hit_rank(retrieved: list[str], relevant: set[str]) -> int | None:
    for i, sid in enumerate(retrieved, start=1):
        if sid in relevant:
            return i
    return None


def evaluate_retrieval(
    rows: list[dict],
    *,
    top_k: int = 10,
    use_rewrite: bool = True,
    use_rerank: bool = True,
    use_multi_query: bool = False,
) -> RetrievalMetrics:
    per_row: list[RetrievalRowResult] = []
    hit1 = hit3 = hit5 = 0
    rr_sum = 0.0

    for row in rows:
        kb = get_kb(row["kb_id"])
        q = normalize_query(row["query"])
        if use_rewrite:
            q = rewrite_query(q)
        if use_multi_query:
            variants = expand_queries(q)
            per_variant_hits: list = []
            for v in variants:
                hs = retrieve(v, kb.root, rerank=False, final_top_k=settings.bm25_top_k)
                if hs:
                    per_variant_hits.append(hs)
            if per_variant_hits:
                fused = rrf_fuse(per_variant_hits, k=settings.rrf_k)
                hits = rerank_hits(row["query"], fused[: max(settings.rerank_top_k, top_k)])[:top_k] if use_rerank else fused[:top_k]
            else:
                hits = []
        else:
            hits = retrieve(q, kb.root, rerank=use_rerank, final_top_k=top_k)
        retrieved = [h.source_id for h in hits]
        # Dedupe preserving order -- one relevant source hit is enough.
        seen: set[str] = set()
        ordered: list[str] = []
        for sid in retrieved:
            if sid not in seen:
                seen.add(sid)
                ordered.append(sid)

        relevant = set(row["relevant_source_ids"])
        category = row.get("category", "direct")
        # out_of_domain rows (or any row with no relevant ids) measure abstain,
        # not retrieval -- skip them from Hit@k/MRR so they don't mechanically
        # drag scores down. They still get scored by answer_eval's abstain_correct.
        scored = category != "out_of_domain" and bool(relevant)

        rank = _first_hit_rank(ordered, relevant) if scored else None
        rr = 1.0 / rank if rank is not None else 0.0
        h1 = rank is not None and rank <= 1
        h3 = rank is not None and rank <= 3
        h5 = rank is not None and rank <= 5

        per_row.append(
            RetrievalRowResult(
                query=row["query"],
                kb_id=row["kb_id"],
                relevant_source_ids=row["relevant_source_ids"],
                hit_at_1=h1,
                hit_at_3=h3,
                hit_at_5=h5,
                reciprocal_rank=rr,
                retrieved_source_ids=ordered[:top_k],
                category=category,
                scored=scored,
            )
        )
        if scored:
            hit1 += int(h1)
            hit3 += int(h3)
            hit5 += int(h5)
            rr_sum += rr

    scored_rows = [r for r in per_row if r.scored]
    n_scored = len(scored_rows) or 1

    by_category: dict[str, CategoryMetrics] = {}
    cats = {r.category for r in per_row}
    for cat in sorted(cats):
        cat_scored = [r for r in scored_rows if r.category == cat]
        if not cat_scored:
            # out_of_domain lands here -- report n only, metrics stay 0.0
            cat_total = sum(1 for r in per_row if r.category == cat)
            by_category[cat] = CategoryMetrics(n=cat_total, hit_at_1=0.0, hit_at_3=0.0, hit_at_5=0.0, mrr_at_10=0.0)
            continue
        denom = len(cat_scored)
        by_category[cat] = CategoryMetrics(
            n=denom,
            hit_at_1=sum(int(r.hit_at_1) for r in cat_scored) / denom,
            hit_at_3=sum(int(r.hit_at_3) for r in cat_scored) / denom,
            hit_at_5=sum(int(r.hit_at_5) for r in cat_scored) / denom,
            mrr_at_10=sum(r.reciprocal_rank for r in cat_scored) / denom,
        )

    return RetrievalMetrics(
        n=len(scored_rows),
        hit_at_1=hit1 / n_scored,
        hit_at_3=hit3 / n_scored,
        hit_at_5=hit5 / n_scored,
        mrr_at_10=rr_sum / n_scored,
        per_row=per_row,
        by_category=by_category,
        n_skipped=sum(1 for r in per_row if not r.scored),
    )


def load_eval_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows
