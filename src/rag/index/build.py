"""Build BM25 + vector indexes for a KB directory.

Picks the vector backend from ``settings.vector_backend``. FAISS writes
artifacts into the KB dir; Qdrant writes into its own collection. Either
way the manifest records which backend was used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import settings
from ..ingest.pipeline import read_chunks_jsonl
from .bm25 import BM25Index
from .vectorstore import get_vector_store


@dataclass
class BuildResult:
    kb_id: str
    chunk_count: int
    bm25_path: Path
    vector_backend: str
    indexed_chunks: int


def build_indexes(kb_dir: Path, kb_id: str) -> BuildResult:
    chunks_path = kb_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found at {chunks_path}")
    chunks = read_chunks_jsonl(chunks_path)

    bm25_path = kb_dir / "bm25.sqlite"
    BM25Index(bm25_path).build(chunks)

    store = get_vector_store(kb_dir, kb_id)
    indexed = store.build(chunks)

    manifest = {
        "kb_id": kb_id,
        "chunk_count": len(chunks),
        "indexed_leaves": indexed,
        "bm25_path": bm25_path.name,
        "vector_backend": settings.vector_backend,
    }
    (kb_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return BuildResult(
        kb_id=kb_id,
        chunk_count=len(chunks),
        bm25_path=bm25_path,
        vector_backend=settings.vector_backend,
        indexed_chunks=indexed,
    )
