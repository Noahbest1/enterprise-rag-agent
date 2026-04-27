"""Retrofit parent chunks onto an existing flat-chunks KB.

Use when the raw source docs are gone but you still have a flat chunks.jsonl.
Each source_id becomes one parent chunk (concatenated text). Existing chunks
are marked as leaves and linked via parent_id. Then BM25 + vector indexes
are rebuilt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.index.build import build_indexes
from rag.ingest.chunking import approx_tokens
from rag.ingest.pipeline import read_chunks_jsonl, write_chunks_jsonl
from rag.knowledge_base import get_kb
from rag.types import Chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kb_id")
    args = parser.parse_args()

    kb = get_kb(args.kb_id)
    existing = read_chunks_jsonl(kb.chunks_path)
    if not existing:
        raise SystemExit(f"no chunks at {kb.chunks_path}")

    # Group leaves by source_id. Keep input order so parent text reads naturally.
    by_source: dict[str, list[Chunk]] = {}
    for c in existing:
        by_source.setdefault(c.source_id, []).append(c)

    new_chunks: list[Chunk] = []
    parents_created = 0
    leaves_linked = 0

    for source_id, leaves in by_source.items():
        leaves.sort(key=lambda x: x.order)
        parent_id = f"{source_id}::parent"
        parent_text = "\n\n".join(leaf.text for leaf in leaves).strip()
        first = leaves[0]

        new_chunks.append(
            Chunk(
                chunk_id=parent_id,
                kb_id=first.kb_id,
                source_id=source_id,
                source_path=first.source_path,
                title=first.title,
                section_path=first.section_path,
                text=parent_text,
                token_count=approx_tokens(parent_text),
                order=-1,
                metadata={**first.metadata, "synthesized_parent": True},
                parent_id=None,
                chunk_role="parent",
            )
        )
        parents_created += 1

        for leaf in leaves:
            new_chunks.append(
                Chunk(
                    chunk_id=leaf.chunk_id,
                    kb_id=leaf.kb_id,
                    source_id=leaf.source_id,
                    source_path=leaf.source_path,
                    title=leaf.title,
                    section_path=leaf.section_path,
                    text=leaf.text,
                    token_count=leaf.token_count,
                    order=leaf.order,
                    metadata=leaf.metadata,
                    parent_id=parent_id,
                    chunk_role="leaf",
                )
            )
            leaves_linked += 1

    write_chunks_jsonl(new_chunks, kb.chunks_path)
    print(f"[parentify] parents={parents_created} leaves={leaves_linked} total={len(new_chunks)}")

    print("[index] rebuilding BM25 + vector (leaves only)...")
    result = build_indexes(kb.root, args.kb_id)
    print(f"[index] done: indexed {result.chunk_count} chunks (leaves)")


if __name__ == "__main__":
    main()
