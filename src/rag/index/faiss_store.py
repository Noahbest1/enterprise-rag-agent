"""FAISS flat-IP vector store with BGE-M3 dense embeddings.

Kept as the zero-dependency default: builds a single ``vector.faiss`` +
``vector_meta.jsonl`` next to the KB, no external service needed. Good
for local dev, eval benchmarks, and small corpora (<1M chunks).
"""
from __future__ import annotations

import asyncio
import json
import threading
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from ..cache import cache_get, cache_set, stable_key
from ..config import settings
from ..logging import get_logger
from ..types import Chunk, Hit
from .vectorstore import VectorStore


log = get_logger(__name__)


_embedder_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str], *, batch_size: int = 16) -> np.ndarray:
    with _embedder_lock:
        model = get_embedder()
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    return vecs.astype(np.float32)


def _emb_key(text: str) -> str:
    return "emb:" + stable_key(settings.embedding_model, text.strip())


def embed_query_cached(text: str) -> np.ndarray:
    """Embed a single query, caching the 1024-d vector in Redis/local.

    Query embeddings are a perfect cache target: the BGE-M3 encoder takes
    150-300 ms per short query, and the same query from a chat UI during
    a typing session recomputes the same vector over and over again.

    We store the vector as a plain JSON list under a key derived from
    (model, text). TTL is 1 hour -- longer than a typing session but
    shorter than a model swap. Fall back to fresh encode on any error.
    """
    key = _emb_key(text)
    try:
        hit = asyncio.run(cache_get(key))
    except RuntimeError:
        hit = None  # already inside an event loop; skip sync peek
    if isinstance(hit, list):
        try:
            arr = np.asarray(hit, dtype=np.float32).reshape(1, -1)
            log.info("embedding_cache_hit", key=key[:16])
            return arr
        except Exception:
            pass
    vec = embed_texts([text], batch_size=1)
    try:
        asyncio.run(cache_set(key, vec[0].tolist(), ttl_seconds=3600))
    except RuntimeError:
        pass
    return vec


class FaissStore(VectorStore):
    def __init__(self, index_path: Path, meta_path: Path):
        self.index_path = index_path
        self.meta_path = meta_path

    def build(self, chunks: Iterable[Chunk]) -> int:
        import faiss
        chunks = [c for c in chunks if c.chunk_role == "leaf"]
        if not chunks:
            return 0

        payloads = [
            f"{c.title}\n{' / '.join(c.section_path)}\n{c.text}".strip()
            for c in chunks
        ]
        vectors = embed_texts(payloads, batch_size=16)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(
                    json.dumps(
                        {
                            "chunk_id": c.chunk_id,
                            "kb_id": c.kb_id,
                            "source_id": c.source_id,
                            "source_path": c.source_path,
                            "title": c.title,
                            "section_path": c.section_path,
                            "text": c.text,
                            "parent_id": c.parent_id,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return len(chunks)

    def _load_meta(self) -> list[dict]:
        if not self.meta_path.exists():
            return []
        out: list[dict] = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def delete_by_source_id(self, source_id: str) -> int:
        """Incremental delete: rewrite the FAISS flat index without the dropped rows.

        FAISS flat doesn't support in-place delete; rebuilding is the idiomatic
        path at our scale (we already do it every build). For much larger
        indexes switch to ``IndexIVFFlat`` + ``remove_ids`` or just use
        Qdrant, which deletes in place.
        """
        import faiss
        if not self.index_path.exists():
            return 0
        meta = self._load_meta()
        keep_rows = [i for i, m in enumerate(meta) if m.get("source_id") != source_id]
        dropped = len(meta) - len(keep_rows)
        if dropped == 0:
            return 0

        old_index = faiss.read_index(str(self.index_path))
        # Extract kept vectors -- pull back to numpy, rebuild a fresh flat index.
        kept_vecs = old_index.reconstruct_n(0, old_index.ntotal)
        kept_vecs = np.asarray(kept_vecs)[keep_rows]
        new_index = faiss.IndexFlatIP(old_index.d)
        if len(kept_vecs) > 0:
            new_index.add(kept_vecs)
        faiss.write_index(new_index, str(self.index_path))

        with self.meta_path.open("w", encoding="utf-8") as f:
            for i in keep_rows:
                f.write(json.dumps(meta[i], ensure_ascii=False) + "\n")
        return dropped

    def upsert_chunks(self, chunks) -> int:
        """Add new chunks (leaves) to the existing FAISS index.

        For flat IP we just extend the index; meta.jsonl gets appended.
        Caller is responsible for calling ``delete_by_source_id`` first if
        it wants 'replace' semantics (see ``incremental.sync_kb``).
        """
        import faiss
        leaves = [c for c in chunks if c.chunk_role == "leaf"]
        if not leaves:
            return 0

        payloads = [
            f"{c.title}\n{' / '.join(c.section_path)}\n{c.text}".strip()
            for c in leaves
        ]
        new_vecs = embed_texts(payloads, batch_size=16)
        dim = int(new_vecs.shape[1])

        if self.index_path.exists():
            index = faiss.read_index(str(self.index_path))
            if index.d != dim:
                raise RuntimeError(
                    f"embedding dim mismatch: existing={index.d} new={dim}"
                )
        else:
            index = faiss.IndexFlatIP(dim)
        index.add(new_vecs)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))

        with self.meta_path.open("a", encoding="utf-8") as f:
            for c in leaves:
                f.write(
                    json.dumps(
                        {
                            "chunk_id": c.chunk_id,
                            "kb_id": c.kb_id,
                            "source_id": c.source_id,
                            "source_path": c.source_path,
                            "title": c.title,
                            "section_path": c.section_path,
                            "text": c.text,
                            "parent_id": c.parent_id,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return len(leaves)

    def search(self, query: str, limit: int) -> list[Hit]:
        import faiss
        if not self.index_path.exists():
            return []
        index = faiss.read_index(str(self.index_path))
        meta = self._load_meta()
        if not meta:
            return []

        qvec = embed_query_cached(query)
        scores, ids = index.search(qvec, min(limit, index.ntotal))
        hits: list[Hit] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(meta):
                continue
            m = meta[idx]
            hits.append(
                Hit(
                    chunk_id=m["chunk_id"],
                    score=float(score),
                    text=m["text"],
                    title=m["title"],
                    source_id=m["source_id"],
                    source_path=m["source_path"],
                    section_path=m.get("section_path", []),
                    retrieval_source="vector",
                    metadata={"kb_id": m["kb_id"], "parent_id": m.get("parent_id"), "backend": "faiss"},
                )
            )
        return hits
