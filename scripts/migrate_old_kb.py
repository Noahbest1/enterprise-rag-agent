"""Migrate an old-format chunks.jsonl into a new-format KB.

Usage:
    python scripts_new/migrate_old_kb.py <old_chunks_jsonl> <new_kb_id>

Reads the old schema (chunk_id, source_id, title, section_title, text,
parent_section_title, ...) and emits new chunks with section_path etc.
Then builds BM25 + vector indexes via the normal builder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.index.build import build_indexes
from rag.ingest.chunking import approx_tokens
from rag.ingest.pipeline import write_chunks_jsonl
from rag.knowledge_base import create_kb
from rag.types import Chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("old_chunks_jsonl")
    parser.add_argument("new_kb_id")
    args = parser.parse_args()

    src = Path(args.old_chunks_jsonl).resolve()
    if not src.exists():
        raise SystemExit(f"not found: {src}")

    kb = create_kb(args.new_kb_id)

    new_chunks: list[Chunk] = []
    order_by_source: dict[str, int] = {}

    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            source_id = d.get("source_id") or d.get("parent_source_id") or d["chunk_id"]
            idx = order_by_source.get(source_id, 0)
            order_by_source[source_id] = idx + 1

            section_path: list[str] = []
            parent_section = d.get("parent_section_title")
            section = d.get("section_title")
            if parent_section and parent_section != section:
                section_path.append(parent_section)
            if section:
                section_path.append(section)

            text = d.get("text") or ""
            title = d.get("title") or source_id
            new_chunks.append(
                Chunk(
                    chunk_id=f"{args.new_kb_id}::{d['chunk_id']}",
                    kb_id=args.new_kb_id,
                    source_id=f"{args.new_kb_id}::{source_id}",
                    source_path=d.get("canonical_url") or source_id,
                    title=title,
                    section_path=section_path,
                    text=text,
                    token_count=approx_tokens(text),
                    order=idx,
                    metadata={
                        "source_type": d.get("source_type", "doc"),
                        "language": d.get("language"),
                        "tags": d.get("tags") or [],
                    },
                )
            )

    write_chunks_jsonl(new_chunks, kb.chunks_path)
    print(f"[migrate] {len(new_chunks)} chunks -> {kb.chunks_path}")

    print("[index] building BM25 + vector...")
    result = build_indexes(kb.root, args.new_kb_id)
    print(f"[index] done: {result.chunk_count} chunks")
    print(f"[ok] KB '{args.new_kb_id}' at {kb.root}")


if __name__ == "__main__":
    main()
