"""Knowledge Base registry.

A KB is a directory under ``data/kb/<kb_id>/`` containing:
    raw/             -- optional, original source files
    chunks.jsonl     -- all chunks for this KB
    bm25.sqlite      -- BM25 FTS5 index
    vector.faiss     -- FAISS flat-IP vector index
    vector_meta.jsonl
    manifest.json    -- summary metadata

The registry is just a thin filesystem wrapper. No database required for
Phase 1; the list-KB endpoint iterates directories.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import settings


@dataclass
class KnowledgeBase:
    kb_id: str
    root: Path
    chunk_count: int = 0
    description: str = ""

    @property
    def chunks_path(self) -> Path:
        return self.root / "chunks.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["root"] = str(self.root)
        return d

    def is_built(self) -> bool:
        """True iff BM25 + vector indexes are both ready.

        Vector readiness depends on the configured backend:
          - faiss:  vector.faiss file on disk
          - qdrant: collection exists on the server / in-memory client
        """
        if not (self.root / "bm25.sqlite").exists():
            return False

        backend = getattr(settings, "vector_backend", "faiss")
        if backend == "faiss":
            return (self.root / "vector.faiss").exists()
        if backend == "qdrant":
            try:
                from .index.qdrant_store import _collection_name, get_client
                return get_client().collection_exists(_collection_name(self.kb_id))
            except Exception:
                return False
        return False


def kb_dir(kb_id: str) -> Path:
    return settings.kb_root / kb_id


def get_kb(kb_id: str) -> KnowledgeBase:
    root = kb_dir(kb_id)
    if not root.exists():
        raise FileNotFoundError(f"Knowledge base not found: {kb_id} (at {root})")
    manifest: dict = {}
    mf = root / "manifest.json"
    if mf.exists():
        manifest = json.loads(mf.read_text(encoding="utf-8"))
    return KnowledgeBase(
        kb_id=kb_id,
        root=root,
        chunk_count=manifest.get("chunk_count", 0),
        description=manifest.get("description", ""),
    )


def create_kb(kb_id: str, description: str = "") -> KnowledgeBase:
    root = kb_dir(kb_id)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    kb = KnowledgeBase(kb_id=kb_id, root=root, description=description)
    if not kb.manifest_path.exists():
        kb.manifest_path.write_text(
            json.dumps(
                {"kb_id": kb_id, "chunk_count": 0, "description": description},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return kb


def list_kbs() -> list[KnowledgeBase]:
    root = settings.kb_root
    if not root.exists():
        return []
    out: list[KnowledgeBase] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        try:
            out.append(get_kb(child.name))
        except FileNotFoundError:
            continue
    return out
