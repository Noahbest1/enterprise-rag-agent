"""Abstract VectorStore interface.

Three implementations live next to this file:

  * ``FaissStore``    -- zero-dep, flat IP, single-file artifacts. Good
    for dev, eval, and <1M chunk corpora.
  * ``QdrantStore``   -- production store with native dense+sparse hybrid,
    payload filtering, incremental updates, horizontal scale.
  * ``PGVectorStore`` -- Postgres + pgvector, leans on the existing DB's
    SQL filtering / backup / replication story.

Switching is a config flag (``VECTOR_BACKEND``) so eval, API, and CLI all
work against the same abstraction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from ..types import Chunk, Hit


class VectorStore(ABC):
    """Minimum surface the retrieval layer calls."""

    @abstractmethod
    def build(self, chunks: Iterable[Chunk]) -> int:
        """Drop-and-rebuild from a chunk iterable. Only leaves are indexed."""

    @abstractmethod
    def search(self, query: str, limit: int) -> list[Hit]:
        """Return top ``limit`` hits ranked by semantic similarity."""

    @abstractmethod
    def delete_by_source_id(self, source_id: str) -> int:
        """Delete all chunks (leaves) whose ``source_id`` matches. Returns rows deleted."""

    @abstractmethod
    def upsert_chunks(self, chunks: Iterable[Chunk]) -> int:
        """Add or replace a batch of chunks (leaves only). Parent chunks are skipped."""


def get_vector_store(kb_dir: Path, kb_id: str) -> VectorStore:
    """Factory: pick backend based on settings.vector_backend.

    Local import avoids importing qdrant-client when only faiss is used
    and vice-versa.
    """
    from ..config import settings

    backend = getattr(settings, "vector_backend", "faiss").lower()
    if backend == "qdrant":
        from .qdrant_store import QdrantStore
        return QdrantStore(kb_id=kb_id)
    if backend == "pgvector":
        from .pgvector_store import PGVectorStore
        return PGVectorStore(kb_id=kb_id)
    if backend == "faiss":
        from .faiss_store import FaissStore
        return FaissStore(
            index_path=kb_dir / "vector.faiss",
            meta_path=kb_dir / "vector_meta.jsonl",
        )
    raise ValueError(f"Unknown VECTOR_BACKEND: {backend}")
