"""PGVector-backed vector store.

Parity with ``FaissStore`` / ``QdrantStore``: implements the ``VectorStore``
contract so the retrieval pipeline sees a single interface regardless of
backend. Activated via ``VECTOR_BACKEND=pgvector`` + ``PGVECTOR_URL=...``.

Why it exists
-------------
FAISS is great for dev / <1M chunks but has no SQL filtering and no cross-
process concurrency. Qdrant is great but needs Docker. PGVector is great
when the team already runs Postgres (which is often), and leans on the DB's
native SQL + backup / replication / ACID story. For résumé / interview
optics, it demonstrates multi-backend fluency.

Single-table design, all KBs share it
-------------------------------------
Schema (created lazily via ``ensure_schema``)::

    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS vector_chunks (
        chunk_id      TEXT PRIMARY KEY,
        kb_id         TEXT NOT NULL,
        source_id     TEXT NOT NULL,
        source_path   TEXT,
        title         TEXT,
        section_path  TEXT[],
        text          TEXT NOT NULL,
        parent_id     TEXT,
        source_type   TEXT DEFAULT 'doc',
        embedding     vector(1024),
        created_at    TIMESTAMPTZ DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS ix_vector_chunks_kb        ON vector_chunks(kb_id);
    CREATE INDEX IF NOT EXISTS ix_vector_chunks_source_id ON vector_chunks(source_id);
    -- HNSW over cosine:
    CREATE INDEX IF NOT EXISTS ix_vector_chunks_emb_hnsw
        ON vector_chunks USING hnsw (embedding vector_cosine_ops);

One table keyed by (kb_id, chunk_id) -- not one table per KB -- because
PGVector indexes are DB-global and splitting per KB prevents the HNSW graph
from amortising across small KBs. KB isolation is enforced at query time
via ``WHERE kb_id = %s``.

Cosine similarity
-----------------
Postgres ``<=>`` returns cosine DISTANCE (0 = identical, 2 = antipodal). We
convert to SIMILARITY = ``1 - distance`` so it's directly comparable to
FAISS (cosine, higher-is-better) and Qdrant (Distance.COSINE, the score
field is already similarity-ish).
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Iterable

from ..config import settings
from ..types import Chunk, Hit
from .faiss_store import embed_query_cached, embed_texts
from .vectorstore import VectorStore


_SCHEMA_LOCK = threading.Lock()
_SCHEMA_READY: dict[str, bool] = {}


@lru_cache(maxsize=1)
def _get_conninfo() -> str:
    """Resolve the Postgres connection string.

    Prefer ``PGVECTOR_URL`` env var (Step D-specific), fall back to the
    general ``database_url`` if it starts with ``postgres``. Empty string
    means the store is unusable -- callers get a clear error rather than
    a random psycopg traceback.
    """
    env = os.getenv("PGVECTOR_URL") or getattr(settings, "pgvector_url", "")
    if env:
        return env
    db_url = getattr(settings, "database_url", "") or ""
    if db_url.startswith("postgres"):
        return db_url
    return ""


def _connect():
    """Open a fresh psycopg3 connection with pgvector types registered.

    psycopg is imported lazily so users on the FAISS / Qdrant path don't
    need the driver installed.
    """
    import psycopg  # lazy
    from pgvector.psycopg import register_vector  # lazy

    conninfo = _get_conninfo()
    if not conninfo:
        raise RuntimeError(
            "PGVectorStore requires PGVECTOR_URL (or DATABASE_URL=postgres://...) "
            "but none is configured."
        )
    conn = psycopg.connect(conninfo, autocommit=False)
    # Make sure the vector extension + adapter are loaded before registering.
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)
    return conn


def _ensure_schema(conn, dim: int) -> None:
    """Create the table + indexes on first use for this connection's DB.

    Keyed on ``(conninfo, dim)`` so different embedding dims (e.g. switching
    from BGE-M3 1024 to BGE-small 512) fail loudly at CREATE rather than
    silently mis-index.
    """
    key = f"{_get_conninfo()}|{dim}"
    with _SCHEMA_LOCK:
        if _SCHEMA_READY.get(key):
            return
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS vector_chunks (
                    chunk_id      TEXT PRIMARY KEY,
                    kb_id         TEXT NOT NULL,
                    source_id     TEXT NOT NULL,
                    source_path   TEXT,
                    title         TEXT,
                    section_path  TEXT[],
                    text          TEXT NOT NULL,
                    parent_id     TEXT,
                    source_type   TEXT DEFAULT 'doc',
                    embedding     vector({dim}),
                    created_at    TIMESTAMPTZ DEFAULT now()
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS ix_vector_chunks_kb ON vector_chunks(kb_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS ix_vector_chunks_source_id ON vector_chunks(source_id);")
            # HNSW index -- ignored gracefully if pgvector is too old for HNSW
            # (falls back to no ANN index = exact scan, still correct).
            try:
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS ix_vector_chunks_emb_hnsw "
                    "ON vector_chunks USING hnsw (embedding vector_cosine_ops);"
                )
            except Exception:  # pragma: no cover -- old pgvector
                conn.rollback()
        conn.commit()
        _SCHEMA_READY[key] = True


def _leaves(chunks: Iterable[Chunk]) -> list[Chunk]:
    return [c for c in chunks if c.chunk_role == "leaf"]


def _payload_text(c: Chunk) -> str:
    return f"{c.title}\n{' / '.join(c.section_path)}\n{c.text}".strip()


class PGVectorStore(VectorStore):
    """Postgres + pgvector backend.

    One shared ``vector_chunks`` table; KBs are logically separated by a
    ``kb_id`` column (unlike Qdrant's per-KB collections). This keeps the
    HNSW index dense enough to be useful even when you have many small KBs.
    """

    def __init__(self, kb_id: str):
        self.kb_id = kb_id

    # ---------------- build (drop + rebuild) ----------------

    def build(self, chunks: Iterable[Chunk]) -> int:
        leaves = _leaves(chunks)
        if not leaves:
            return 0
        payloads = [_payload_text(c) for c in leaves]
        vectors = embed_texts(payloads, batch_size=16)
        dim = int(vectors.shape[1])

        conn = _connect()
        try:
            _ensure_schema(conn, dim)
            with conn.cursor() as cur:
                # Wipe this KB only; other KBs on the same table are left alone.
                cur.execute("DELETE FROM vector_chunks WHERE kb_id = %s;", (self.kb_id,))
                rows = [
                    (
                        c.chunk_id,
                        c.kb_id,
                        c.source_id,
                        c.source_path,
                        c.title,
                        list(c.section_path or []),
                        c.text,
                        c.parent_id,
                        (c.metadata or {}).get("source_type", "doc"),
                        vec.tolist(),
                    )
                    for c, vec in zip(leaves, vectors)
                ]
                cur.executemany(
                    """
                    INSERT INTO vector_chunks
                        (chunk_id, kb_id, source_id, source_path, title,
                         section_path, text, parent_id, source_type, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    rows,
                )
            conn.commit()
        finally:
            conn.close()
        return len(leaves)

    # ---------------- incremental ----------------

    def delete_by_source_id(self, source_id: str) -> int:
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM vector_chunks WHERE kb_id = %s AND source_id = %s;",
                    (self.kb_id, source_id),
                )
                deleted = cur.rowcount
            conn.commit()
        finally:
            conn.close()
        return int(deleted or 0)

    def upsert_chunks(self, chunks: Iterable[Chunk]) -> int:
        leaves = _leaves(chunks)
        if not leaves:
            return 0
        payloads = [_payload_text(c) for c in leaves]
        vectors = embed_texts(payloads, batch_size=16)
        dim = int(vectors.shape[1])

        conn = _connect()
        try:
            _ensure_schema(conn, dim)
            rows = [
                (
                    c.chunk_id,
                    c.kb_id,
                    c.source_id,
                    c.source_path,
                    c.title,
                    list(c.section_path or []),
                    c.text,
                    c.parent_id,
                    (c.metadata or {}).get("source_type", "doc"),
                    vec.tolist(),
                )
                for c, vec in zip(leaves, vectors)
            ]
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO vector_chunks
                        (chunk_id, kb_id, source_id, source_path, title,
                         section_path, text, parent_id, source_type, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        kb_id = EXCLUDED.kb_id,
                        source_id = EXCLUDED.source_id,
                        source_path = EXCLUDED.source_path,
                        title = EXCLUDED.title,
                        section_path = EXCLUDED.section_path,
                        text = EXCLUDED.text,
                        parent_id = EXCLUDED.parent_id,
                        source_type = EXCLUDED.source_type,
                        embedding = EXCLUDED.embedding;
                    """,
                    rows,
                )
            conn.commit()
        finally:
            conn.close()
        return len(leaves)

    # ---------------- search ----------------

    def search(self, query: str, limit: int) -> list[Hit]:
        qvec = embed_query_cached(query)[0]
        conn = _connect()
        try:
            with conn.cursor() as cur:
                # <=> is pgvector's cosine distance operator.
                # We map distance d ∈ [0, 2] back to similarity = 1 - d so
                # callers compare apples to apples with FAISS / Qdrant.
                cur.execute(
                    """
                    SELECT chunk_id, kb_id, source_id, source_path, title,
                           section_path, text, parent_id, source_type,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM vector_chunks
                    WHERE kb_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (qvec.tolist(), self.kb_id, qvec.tolist(), int(limit)),
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        hits: list[Hit] = []
        for r in rows:
            (
                chunk_id, kb_id, source_id, source_path, title,
                section_path, text, parent_id, source_type, similarity,
            ) = r
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    score=float(similarity),
                    text=text or "",
                    title=title or "",
                    source_id=source_id or "",
                    source_path=source_path or "",
                    section_path=list(section_path or []),
                    retrieval_source="vector",
                    metadata={
                        "kb_id": kb_id,
                        "parent_id": parent_id,
                        "source_type": source_type,
                        "backend": "pgvector",
                    },
                )
            )
        return hits
