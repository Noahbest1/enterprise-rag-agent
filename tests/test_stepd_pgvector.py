"""Step D: PGVector VectorStore backend.

Two layers of coverage:

1. **Unit tests (always run)** — factory dispatch + connection-string
   resolution + the SQL statements the store would emit (with psycopg
   stubbed). These catch wire-up bugs without needing Postgres.

2. **Integration smoke (gated on PGVECTOR_URL)** — full round-trip:
   build → search → upsert → delete against a real Postgres + pgvector
   instance. Skipped automatically in CI and local dev.

The integration test sets up its own `vector_chunks` table via the store's
own `_ensure_schema`, so running it against any reachable Postgres with
the pgvector extension installed "just works".
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.index.vectorstore import get_vector_store
from rag.types import Chunk


_PGVECTOR_URL = os.getenv("PGVECTOR_URL", "").strip()
_INTEGRATION = pytest.mark.skipif(
    not _PGVECTOR_URL,
    reason="PGVECTOR_URL not set — set to a reachable Postgres+pgvector URL to run",
)


# ---------- factory tests ----------

def test_factory_dispatches_to_pgvector(monkeypatch):
    """Pointing VECTOR_BACKEND at pgvector returns a PGVectorStore."""
    from rag.config import settings
    from rag.index.pgvector_store import PGVectorStore

    monkeypatch.setattr(settings, "vector_backend", "pgvector")
    store = get_vector_store(Path("/tmp/ignored"), kb_id="kb_unit")
    assert isinstance(store, PGVectorStore)
    assert store.kb_id == "kb_unit"


def test_factory_rejects_unknown_backend(monkeypatch):
    from rag.config import settings

    monkeypatch.setattr(settings, "vector_backend", "nonesuch")
    with pytest.raises(ValueError, match="Unknown VECTOR_BACKEND"):
        get_vector_store(Path("/tmp/ignored"), kb_id="x")


def test_pgvector_url_resolution_prefers_env_var(monkeypatch):
    """`_get_conninfo` prefers PGVECTOR_URL over settings.database_url."""
    from rag.index import pgvector_store

    # Clear lru_cache so this test's env var is honoured.
    pgvector_store._get_conninfo.cache_clear()
    monkeypatch.setenv("PGVECTOR_URL", "postgresql://via-env:5432/x")
    assert pgvector_store._get_conninfo() == "postgresql://via-env:5432/x"
    pgvector_store._get_conninfo.cache_clear()


def test_pgvector_url_falls_back_to_postgres_database_url(monkeypatch):
    """When PGVECTOR_URL is empty but DATABASE_URL is a Postgres URL, use it."""
    from rag.config import settings
    from rag.index import pgvector_store

    pgvector_store._get_conninfo.cache_clear()
    monkeypatch.delenv("PGVECTOR_URL", raising=False)
    monkeypatch.setattr(settings, "pgvector_url", "")
    monkeypatch.setattr(settings, "database_url", "postgresql://fallback:5432/r")
    assert pgvector_store._get_conninfo() == "postgresql://fallback:5432/r"
    pgvector_store._get_conninfo.cache_clear()


def test_pgvector_raises_cleanly_when_unconfigured(monkeypatch):
    """With no PGVECTOR_URL and SQLite DATABASE_URL, connection fails loudly."""
    from rag.config import settings
    from rag.index import pgvector_store

    pgvector_store._get_conninfo.cache_clear()
    monkeypatch.delenv("PGVECTOR_URL", raising=False)
    monkeypatch.setattr(settings, "pgvector_url", "")
    monkeypatch.setattr(settings, "database_url", "sqlite:///./x.db")
    with pytest.raises(RuntimeError, match="PGVectorStore requires PGVECTOR_URL"):
        pgvector_store._connect()
    pgvector_store._get_conninfo.cache_clear()


# ---------- SQL shape tests (psycopg mocked) ----------

def _fake_chunk(i: int, kb="kb_unit", source_id="doc1") -> Chunk:
    return Chunk(
        chunk_id=f"c{i}",
        kb_id=kb,
        source_id=source_id,
        source_path=f"/tmp/{source_id}.md",
        title=f"Title {i}",
        section_path=["Root", f"Sec {i}"],
        text=f"Body text number {i} with something to embed",
        token_count=10,
        order=i,
        metadata={"source_type": "doc"},
        parent_id=None,
        chunk_role="leaf",
    )


class _FakeCursor:
    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []
        self.rowcount = 0
        self._rows: list[tuple] = []

    def execute(self, sql, params=()):
        self.calls.append((sql.strip(), params))
        # Emulate DELETE rowcount for delete_by_source_id.
        if "DELETE" in sql.upper() and "source_id" in sql:
            self.rowcount = 3
        return self

    def executemany(self, sql, rows):
        self.calls.append((sql.strip(), f"<{len(rows)} rows>"))

    def fetchall(self):
        return self._rows

    def __enter__(self): return self

    def __exit__(self, *a): return False


class _FakeConn:
    def __init__(self):
        self.cur = _FakeCursor()
        self.committed = 0
        self.closed = False

    def cursor(self): return self.cur

    def commit(self): self.committed += 1

    def rollback(self): pass

    def close(self): self.closed = True


@pytest.fixture()
def fake_store(monkeypatch):
    """Patch psycopg + embedder so we can test SQL wiring without PG or GPU."""
    import numpy as np
    from rag.index import pgvector_store

    fake_conn = _FakeConn()

    def fake_connect(*args, **kwargs):
        return fake_conn

    # embed_texts returns a (n, 1024) zero matrix -- valid shape for SQL.
    def fake_embed_texts(texts, *, batch_size=16):
        return np.zeros((len(texts), 1024), dtype="float32")

    def fake_embed_query_cached(text):
        return np.zeros((1, 1024), dtype="float32")

    # Schema "ready" bypasses the _ensure_schema psycopg-specific CREATE calls
    # by marking the current connection key as already prepared.
    pgvector_store._SCHEMA_READY.clear()
    pgvector_store._SCHEMA_READY[f"{pgvector_store._get_conninfo()}|1024"] = True

    monkeypatch.setattr(pgvector_store, "_connect", fake_connect)
    monkeypatch.setattr(pgvector_store, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(pgvector_store, "embed_query_cached", fake_embed_query_cached)
    yield pgvector_store, fake_conn


def test_build_deletes_then_inserts(fake_store):
    pgvector_store, conn = fake_store
    store = pgvector_store.PGVectorStore(kb_id="kb_unit")
    n = store.build([_fake_chunk(i) for i in range(3)])
    assert n == 3
    sqls = [c[0] for c in conn.cur.calls]
    # First a DELETE on this KB, then an INSERT executemany.
    assert any("DELETE FROM vector_chunks WHERE kb_id" in s for s in sqls)
    assert any(s.upper().startswith("INSERT INTO VECTOR_CHUNKS") for s in sqls)
    assert conn.committed == 1
    assert conn.closed is True


def test_build_skips_non_leaves(fake_store):
    pgvector_store, conn = fake_store
    parent = _fake_chunk(0)
    parent.chunk_role = "parent"
    store = pgvector_store.PGVectorStore(kb_id="kb_unit")
    n = store.build([parent])
    assert n == 0
    # No SQL at all when there are no leaves.
    assert conn.cur.calls == []


def test_delete_by_source_id_returns_rowcount(fake_store):
    pgvector_store, conn = fake_store
    store = pgvector_store.PGVectorStore(kb_id="kb_unit")
    n = store.delete_by_source_id("doc1")
    assert n == 3  # fake cursor reports rowcount=3
    assert any("DELETE FROM vector_chunks" in c[0] for c in conn.cur.calls)


def test_upsert_uses_on_conflict(fake_store):
    pgvector_store, conn = fake_store
    store = pgvector_store.PGVectorStore(kb_id="kb_unit")
    n = store.upsert_chunks([_fake_chunk(i) for i in range(2)])
    assert n == 2
    sql = next(c[0] for c in conn.cur.calls if c[0].upper().startswith("INSERT"))
    assert "ON CONFLICT (chunk_id) DO UPDATE" in sql


def test_search_emits_cosine_distance_ordering(fake_store):
    pgvector_store, conn = fake_store
    # Prime a row so fetchall returns one Hit.
    conn.cur._rows = [
        (
            "c1", "kb_unit", "doc1", "/tmp/doc1.md", "Title",
            ["Root", "Sec"], "body", None, "doc", 0.87,
        )
    ]
    store = pgvector_store.PGVectorStore(kb_id="kb_unit")
    hits = store.search("what", limit=5)
    assert len(hits) == 1
    assert hits[0].score == pytest.approx(0.87)
    assert hits[0].metadata["backend"] == "pgvector"
    sql = next(c[0] for c in conn.cur.calls if "ORDER BY" in c[0])
    assert "embedding <=> %s::vector" in sql
    # KB isolation must be in the WHERE clause.
    assert "WHERE kb_id = %s" in sql


# ---------- integration smoke (gated) ----------

@_INTEGRATION
def test_pgvector_integration_round_trip(monkeypatch):  # pragma: no cover
    """Full round trip against a real Postgres+pgvector server.

    Skipped when PGVECTOR_URL is unset. To run::

        PGVECTOR_URL=postgresql://user:pass@localhost:5432/test_rag \
        ./.venv/bin/pytest tests/test_stepd_pgvector.py::test_pgvector_integration_round_trip -v
    """
    from rag.config import settings
    from rag.index.pgvector_store import PGVectorStore, _get_conninfo

    _get_conninfo.cache_clear()
    monkeypatch.setattr(settings, "pgvector_url", _PGVECTOR_URL)

    store = PGVectorStore(kb_id="pgvector_itest")
    chunks = [_fake_chunk(i, kb="pgvector_itest") for i in range(3)]

    # build
    assert store.build(chunks) == 3
    hits = store.search("something to embed", limit=5)
    assert len(hits) >= 1
    # All hits must belong to THIS kb -- other KBs share the same table.
    for h in hits:
        assert h.metadata["kb_id"] == "pgvector_itest"

    # upsert (add one more)
    more = [_fake_chunk(99, kb="pgvector_itest")]
    assert store.upsert_chunks(more) == 1

    # delete_by_source_id removes what it should
    deleted = store.delete_by_source_id("doc1")
    assert deleted >= 3  # all 3 original leaves share source_id="doc1"

    # cleanup
    store.delete_by_source_id("doc1")
    _get_conninfo.cache_clear()
