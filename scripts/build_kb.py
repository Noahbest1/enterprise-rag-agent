"""Build (or rebuild) a knowledge base from files on disk.

Usage:
    python scripts/build_kb.py <kb_id> [<source_dir>] [--with-context] [--backend faiss|qdrant]

If <source_dir> is given, its files are copied into data/kb/<kb_id>/raw/ first.
Otherwise we just index whatever is already in data/kb/<kb_id>/raw/.

Flags:
    --with-context       Prepend LLM-generated context to each chunk before
                         indexing (Anthropic Contextual Retrieval). Caches
                         per chunk content hash so reruns are free.
    --backend faiss|qdrant
                         Override VECTOR_BACKEND for this build only.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kb_id")
    parser.add_argument("source_dir", nargs="?", default=None)
    parser.add_argument("--no-copy", action="store_true", help="Do not copy source files; index in place")
    parser.add_argument("--with-context", action="store_true",
                        help="Prepend LLM-generated context to each chunk before indexing.")
    parser.add_argument("--backend", choices=["faiss", "qdrant"], default=None,
                        help="Override VECTOR_BACKEND for this build only.")
    args = parser.parse_args()

    # Apply overrides BEFORE importing anything that reads settings.
    if args.backend:
        os.environ["VECTOR_BACKEND"] = args.backend
    if args.with_context:
        os.environ["ENABLE_CONTEXTUAL_RETRIEVAL"] = "true"

    # Force a fresh settings read so the env overrides above take effect.
    from rag.config import get_settings
    get_settings.cache_clear()
    from rag.config import settings

    from rag.index.build import build_indexes
    from rag.ingest.contextual import annotate_chunks_with_context
    from rag.ingest.loaders import supported_extensions
    from rag.ingest.pipeline import ingest_directory, write_chunks_jsonl
    from rag.knowledge_base import create_kb

    kb = create_kb(args.kb_id)
    raw_dir = kb.root / "raw"

    if args.source_dir and not args.no_copy:
        src = Path(args.source_dir).resolve()
        if not src.exists():
            raise SystemExit(f"source_dir does not exist: {src}")
        exts = supported_extensions()
        for path in src.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                dest = raw_dir / path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
                print(f"[copy] {path} -> {dest}")

    source_for_ingest = Path(args.source_dir).resolve() if (args.source_dir and args.no_copy) else raw_dir
    print(f"[ingest] source={source_for_ingest}")
    chunks = ingest_directory(source_for_ingest, args.kb_id)
    if not chunks:
        raise SystemExit(f"No chunks produced from {source_for_ingest}")
    print(f"[ingest] {len(chunks)} chunks")

    if args.with_context:
        leaf_n = sum(1 for c in chunks if c.chunk_role == 'leaf')
        print(f"[context] annotating {leaf_n} leaves with LLM context (cache-aware)...")
        chunks = annotate_chunks_with_context(chunks, kb.root)
        print("[context] done")

    write_chunks_jsonl(chunks, kb.chunks_path)
    print(f"[write] {kb.chunks_path}")

    print(f"[index] building BM25 + vector (backend={settings.vector_backend})...")
    result = build_indexes(kb.root, args.kb_id)
    print(f"[index] done: {result.chunk_count} chunks ({result.indexed_chunks} leaves indexed), backend={result.vector_backend}")

    print(f"[ok] KB '{args.kb_id}' built at {kb.root}")


if __name__ == "__main__":
    main()
