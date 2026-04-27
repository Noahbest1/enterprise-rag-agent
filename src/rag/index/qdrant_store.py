"""Qdrant-backed vector store.

One collection per KB (``kb_<kb_id>``) so:
  * different knowledge bases are hard-isolated
  * payload filters (source_type, section, parent_id) can stay cheap
  * listing / deleting a whole KB is one call

Connection mode is picked from ``settings.qdrant_url``:
  * ``:memory:``        -- in-process, zero-ops, great for unit tests
  * ``./path/to/dir``    -- on-disk local mode, survives restarts
  * ``http(s)://...``   -- real Qdrant server (Docker / Cloud)

We only embed leaves. Dense vectors use BGE-M3 (1024-dim, cosine).
Sparse hybrid is left as a follow-up: Qdrant supports native dense+sparse
fusion, but SPLADE-style sparse vectors need a separate encoder we're not
ready to ship today. For now we stay dense-only in Qdrant; the BM25 path
in ``bm25.py`` remains the sparse signal, and RRF in ``retrieval/hybrid.py``
fuses the two. Same rig as FAISS, just with Qdrant on the dense side.
"""
from __future__ import annotations

import hashlib
import os
import threading
from functools import lru_cache
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ..config import settings
from ..types import Chunk, Hit
from .faiss_store import embed_query_cached, embed_texts, get_embedder
from .vectorstore import VectorStore


_client_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Singleton Qdrant client. Re-use for every store instance."""
    url = settings.qdrant_url or ":memory:"
    if url.startswith(":memory:"):
        return QdrantClient(location=":memory:")
    # Disk-backed local mode lets on-disk indexes survive process restarts
    # without needing Docker. Useful for eval + dev.
    if url.startswith("local://"):
        path = url.replace("local://", "", 1) or str(settings.project_root / "data" / "qdrant_local")
        os.makedirs(path, exist_ok=True)
        return QdrantClient(path=path)
    # HTTP(S) to a real server -- the Docker Compose service hits this path.
    api_key = settings.qdrant_api_key or None
    return QdrantClient(url=url, api_key=api_key, prefer_grpc=False)


def _collection_name(kb_id: str) -> str:
    # Qdrant allows [a-zA-Z0-9_-], but KB IDs are already sanitized.
    return f"kb_{kb_id}"


def _stable_uuid(chunk_id: str) -> str:
    """Qdrant point IDs must be ints or UUIDs. Derive a stable UUID from chunk_id."""
    h = hashlib.md5(chunk_id.encode("utf-8")).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


class QdrantStore(VectorStore):
    def __init__(self, kb_id: str):
        self.kb_id = kb_id
        self.collection = _collection_name(kb_id)

    # ---------------- build ----------------

    def _recreate_collection(self, dim: int) -> None:
        client = get_client()
        with _client_lock:
            if client.collection_exists(self.collection):
                client.delete_collection(self.collection)
            client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
            )
            # Payload indexes make filtered search fast. All four are cheap.
            for field in ("source_id", "source_type", "parent_id", "kb_id"):
                client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )

    def build(self, chunks: Iterable[Chunk]) -> int:
        leaves = [c for c in chunks if c.chunk_role == "leaf"]
        if not leaves:
            return 0

        payloads = [
            f"{c.title}\n{' / '.join(c.section_path)}\n{c.text}".strip()
            for c in leaves
        ]
        vectors = embed_texts(payloads, batch_size=16)
        dim = int(vectors.shape[1])
        self._recreate_collection(dim)

        points: list[qm.PointStruct] = []
        for c, vec in zip(leaves, vectors):
            points.append(
                qm.PointStruct(
                    id=_stable_uuid(c.chunk_id),
                    vector=vec.tolist(),
                    payload={
                        "chunk_id": c.chunk_id,
                        "kb_id": c.kb_id,
                        "source_id": c.source_id,
                        "source_path": c.source_path,
                        "title": c.title,
                        "section_path": c.section_path,
                        "text": c.text,
                        "parent_id": c.parent_id,
                        "source_type": (c.metadata or {}).get("source_type", "doc"),
                    },
                )
            )

        client = get_client()
        # Batch upsert -- Qdrant client handles chunking internally.
        client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(leaves)

    # ---------------- incremental ----------------

    def delete_by_source_id(self, source_id: str) -> int:
        client = get_client()
        if not client.collection_exists(self.collection):
            return 0
        count_resp = client.count(
            collection_name=self.collection,
            count_filter=qm.Filter(must=[qm.FieldCondition(key="source_id", match=qm.MatchValue(value=source_id))]),
        )
        before = int(count_resp.count or 0)
        client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(must=[qm.FieldCondition(key="source_id", match=qm.MatchValue(value=source_id))])
            ),
            wait=True,
        )
        return before

    def upsert_chunks(self, chunks) -> int:
        """Add new chunks to an existing collection. ``build`` is the full-rebuild variant."""
        leaves = [c for c in chunks if c.chunk_role == "leaf"]
        if not leaves:
            return 0

        payloads = [
            f"{c.title}\n{' / '.join(c.section_path)}\n{c.text}".strip()
            for c in leaves
        ]
        vectors = embed_texts(payloads, batch_size=16)
        dim = int(vectors.shape[1])

        client = get_client()
        if not client.collection_exists(self.collection):
            self._recreate_collection(dim)

        points: list[qm.PointStruct] = []
        for c, vec in zip(leaves, vectors):
            points.append(
                qm.PointStruct(
                    id=_stable_uuid(c.chunk_id),
                    vector=vec.tolist(),
                    payload={
                        "chunk_id": c.chunk_id,
                        "kb_id": c.kb_id,
                        "source_id": c.source_id,
                        "source_path": c.source_path,
                        "title": c.title,
                        "section_path": c.section_path,
                        "text": c.text,
                        "parent_id": c.parent_id,
                        "source_type": (c.metadata or {}).get("source_type", "doc"),
                    },
                )
            )
        client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(leaves)

    # ---------------- search ----------------

    def search(self, query: str, limit: int) -> list[Hit]:
        client = get_client()
        if not client.collection_exists(self.collection):
            return []
        qvec = embed_query_cached(query)[0]

        # Use query_points (stable, newer API). Falls back gracefully if
        # the server is an older version that only knows .search().
        try:
            response = client.query_points(
                collection_name=self.collection,
                query=qvec.tolist(),
                limit=limit,
                with_payload=True,
            )
            points = response.points
        except Exception:
            points = client.search(
                collection_name=self.collection,
                query_vector=qvec.tolist(),
                limit=limit,
                with_payload=True,
            )

        hits: list[Hit] = []
        for p in points:
            payload = dict(p.payload or {})
            hits.append(
                Hit(
                    chunk_id=payload.get("chunk_id", str(p.id)),
                    score=float(p.score),
                    text=payload.get("text", ""),
                    title=payload.get("title", ""),
                    source_id=payload.get("source_id", ""),
                    source_path=payload.get("source_path", ""),
                    section_path=payload.get("section_path", []) or [],
                    retrieval_source="vector",
                    metadata={
                        "kb_id": payload.get("kb_id"),
                        "parent_id": payload.get("parent_id"),
                        "source_type": payload.get("source_type"),
                        "backend": "qdrant",
                    },
                )
            )
        return hits
