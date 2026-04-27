"""End-to-end: file/directory -> chunks.jsonl for a knowledge base."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import settings
from ..types import Chunk
from .chunking import approx_tokens, chunk_document
from .loaders import load_file, supported_extensions


def _source_id(kb_id: str, path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{kb_id}::{path.stem}::{digest}"


def ingest_file(
    path: Path,
    kb_id: str,
    *,
    chunks_out: list[Chunk] | None = None,
) -> list[Chunk]:
    """Load a file, chunk into (parent + leaves), return Chunk list."""
    title, text = load_file(path)
    if not text.strip():
        return chunks_out or []

    source_id = _source_id(kb_id, path)
    payloads = chunk_document(
        text,
        target_tokens=settings.chunk_target_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )

    # Lift parent_local_id -> global parent chunk_id so leaves can reference them.
    parent_global: dict[str, str] = {}
    for i, p in enumerate(payloads):
        if p.role == "parent" and p.parent_local_id:
            parent_global[p.parent_local_id] = f"{source_id}::p{i:04d}"

    results: list[Chunk] = []
    for i, p in enumerate(payloads):
        if p.role == "parent":
            chunk_id = parent_global[p.parent_local_id]
            parent_id = None
        else:
            chunk_id = f"{source_id}::l{i:04d}"
            parent_id = parent_global.get(p.parent_local_id) if p.parent_local_id else None

        results.append(
            Chunk(
                chunk_id=chunk_id,
                kb_id=kb_id,
                source_id=source_id,
                source_path=str(path.resolve()),
                title=title,
                section_path=p.section_path,
                text=p.text,
                token_count=approx_tokens(p.text),
                order=i,
                metadata={"file_suffix": path.suffix.lower()},
                parent_id=parent_id,
                chunk_role=p.role,
            )
        )

    if chunks_out is not None:
        chunks_out.extend(results)
    return results


def ingest_directory(
    root: Path,
    kb_id: str,
    *,
    recursive: bool = True,
) -> list[Chunk]:
    exts = supported_extensions()
    iterator = root.rglob("*") if recursive else root.glob("*")
    collected: list[Chunk] = []
    for path in sorted(iterator):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        try:
            ingest_file(path, kb_id, chunks_out=collected)
        except Exception as e:
            print(f"[ingest] skip {path}: {e}")
    return collected


def write_chunks_jsonl(chunks: list[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def read_chunks_jsonl(path: Path) -> list[Chunk]:
    out: list[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(Chunk(**d))
    return out
