"""Expand leaf hits to their parent chunks (small-to-big retrieval).

After rerank we have precise leaf matches (~250 tokens each). The LLM
answers better with wider context, so here we replace each leaf with its
parent chunk's text. Multiple leaves pointing to the same parent are
deduped, keeping the best leaf's score. The leaf snippet is preserved in
metadata so the UI can still show the precise matching sentence.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..types import Hit


def _load_parents_by_id(chunks_path: Path) -> dict[str, dict]:
    parents: dict[str, dict] = {}
    if not chunks_path.exists():
        return parents
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("chunk_role") == "parent":
                parents[d["chunk_id"]] = d
    return parents


def expand_to_parents(hits: list[Hit], chunks_path: Path) -> list[Hit]:
    if not hits:
        return hits
    parents = _load_parents_by_id(chunks_path)
    if not parents:
        return hits  # legacy KB without parents: keep leaves as-is

    seen: dict[str, Hit] = {}
    for hit in hits:
        parent_id = hit.metadata.get("parent_id") if hit.metadata else None
        if not parent_id or parent_id not in parents:
            # Leaf without a parent (legacy or orphan) -> keep as-is.
            key = hit.chunk_id
            if key not in seen or seen[key].score < hit.score:
                seen[key] = hit
            continue

        parent = parents[parent_id]
        key = parent_id
        existing = seen.get(key)
        if existing and existing.score >= hit.score:
            # Already have a better leaf pointing to this parent; keep it but
            # update the snippet if this leaf is a stronger match.
            continue

        leaf_snippet = hit.text
        new_hit = Hit(
            chunk_id=parent_id,
            score=hit.score,
            text=parent["text"],
            title=parent["title"],
            source_id=parent["source_id"],
            source_path=parent.get("source_path", ""),
            section_path=parent.get("section_path", []),
            retrieval_source=hit.retrieval_source + "+parent",
            metadata={
                **hit.metadata,
                "leaf_chunk_id": hit.chunk_id,
                "leaf_snippet": leaf_snippet,
            },
        )
        seen[key] = new_hit

    # Preserve original order by score
    return sorted(seen.values(), key=lambda h: h.score, reverse=True)
