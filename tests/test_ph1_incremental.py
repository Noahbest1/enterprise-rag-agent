"""PH1 acceptance tests: connectors + incremental ingest.

We use the FAISS backend (zero-dep) + a temporary directory as source.
Qdrant tests live in a different file; the interface is identical.

Cases covered:
  - sync adds new docs, builds both BM25 and vector indexes
  - a second sync with no changes is a no-op (skipped == all)
  - editing a file re-embeds that file only (added=0, updated=1)
  - deleting a file drops its chunks from both indexes
  - a retrieval round-trip works after each of the above
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture()
def kb_dir(tmp_path):
    return tmp_path / "kb"


@pytest.fixture()
def source_dir(tmp_path):
    d = tmp_path / "src_docs"
    d.mkdir()
    return d


def _write(p: Path, text: str):
    p.write_text(text, encoding="utf-8")


def _ids_in_bm25(kb_dir: Path) -> set[str]:
    import sqlite3
    db = kb_dir / "bm25.sqlite"
    if not db.exists():
        return set()
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute("SELECT DISTINCT source_id FROM chunks_fts").fetchall()
        return {r[0] for r in rows}
    finally:
        conn.close()


def _ids_in_vector(kb_dir: Path) -> set[str]:
    import json
    meta = kb_dir / "vector_meta.jsonl"
    if not meta.exists():
        return set()
    out: set[str] = set()
    for line in meta.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.add(json.loads(line)["source_id"])
    return out


def _run_sync(kb_id: str, src: Path, kb_dir: Path):
    from rag.ingest.connectors import LocalDirConnector
    from rag.ingest.incremental import sync_kb
    kb_dir.mkdir(parents=True, exist_ok=True)
    return sync_kb(kb_id, LocalDirConnector(kb_id, src), kb_dir)


def test_initial_sync_adds_docs(seeded_db, source_dir, kb_dir, monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "faiss")
    _write(source_dir / "a.md", "# Alpha\nThis is the alpha document about cats.")
    _write(source_dir / "b.md", "# Bravo\nThis is the bravo document about dogs.")

    stats = _run_sync("ph1_kb", source_dir, kb_dir)
    assert stats.summary()["added"] == 2
    assert stats.summary()["updated"] == 0
    assert stats.summary()["deleted"] == 0

    # Both backends know about both docs
    bm25_ids = _ids_in_bm25(kb_dir)
    vec_ids = _ids_in_vector(kb_dir)
    assert len(bm25_ids) == 2
    assert bm25_ids == vec_ids


def test_second_sync_is_noop(seeded_db, source_dir, kb_dir, monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "faiss")
    _write(source_dir / "a.md", "# Alpha\nsome content")
    _run_sync("ph1_kb2", source_dir, kb_dir)

    # Touch mtime but same content
    p = source_dir / "a.md"
    stat = p.stat()
    os.utime(p, (stat.st_atime, stat.st_mtime + 10))

    stats = _run_sync("ph1_kb2", source_dir, kb_dir)
    # Either skipped (cheap_hash same) or content-hash same -> still skipped
    summary = stats.summary()
    assert summary["added"] == 0
    assert summary["updated"] == 0


def test_edit_triggers_update_only_for_that_doc(seeded_db, source_dir, kb_dir, monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "faiss")
    _write(source_dir / "a.md", "# Alpha\noriginal content")
    _write(source_dir / "b.md", "# Bravo\nunchanged content")
    _run_sync("ph1_kb3", source_dir, kb_dir)

    bm25_before = _ids_in_bm25(kb_dir)
    assert len(bm25_before) == 2

    # Edit only a.md
    _write(source_dir / "a.md", "# Alpha\nNEW longer content that chunks differently")
    stats = _run_sync("ph1_kb3", source_dir, kb_dir)
    summary = stats.summary()
    assert summary["updated"] == 1, f"expected 1 update, got {summary}"
    assert summary["added"] == 0

    # Both docs still present
    bm25_after = _ids_in_bm25(kb_dir)
    assert bm25_after == bm25_before


def test_delete_removes_chunks_from_both_backends(seeded_db, source_dir, kb_dir, monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "faiss")
    _write(source_dir / "a.md", "# Alpha\ncontent A")
    _write(source_dir / "b.md", "# Bravo\ncontent B")
    _run_sync("ph1_kb4", source_dir, kb_dir)

    assert len(_ids_in_bm25(kb_dir)) == 2

    # Remove b.md upstream
    (source_dir / "b.md").unlink()
    stats = _run_sync("ph1_kb4", source_dir, kb_dir)
    assert stats.summary()["deleted"] == 1

    bm25_ids = _ids_in_bm25(kb_dir)
    vec_ids = _ids_in_vector(kb_dir)
    assert len(bm25_ids) == 1
    assert bm25_ids == vec_ids


def test_retrieval_works_after_incremental_sync(seeded_db, source_dir, kb_dir, monkeypatch):
    """End-to-end: after a sync the KB is actually queryable."""
    monkeypatch.setenv("VECTOR_BACKEND", "faiss")
    _write(
        source_dir / "cats.md",
        "# Cats\nCats purr when content. They groom themselves regularly.",
    )
    _run_sync("ph1_kb_rt", source_dir, kb_dir)

    from rag.retrieval.hybrid import retrieve
    hits = retrieve("why do cats purr", kb_dir, rerank=False)
    assert any("cats" in h.title.lower() or "purr" in h.text.lower() for h in hits), \
        f"no cat-ish hit found in {[h.title for h in hits]}"


def test_kb_document_registry_is_maintained(seeded_db, source_dir, kb_dir, monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "faiss")
    _write(source_dir / "a.md", "# A\nsomething")
    _run_sync("ph1_kb_reg", source_dir, kb_dir)

    from rag.db.base import SessionLocal
    from rag.db.models import KBDocument
    with SessionLocal() as s:
        rows = s.query(KBDocument).filter_by(kb_id="ph1_kb_reg").all()
        assert len(rows) == 1
        row = rows[0]
        assert row.connector == "local"
        assert row.content_hash
        assert row.chunk_count > 0


def test_stable_source_id_across_runs(seeded_db, source_dir, kb_dir, monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "faiss")
    _write(source_dir / "a.md", "# A\ncontent")
    s1 = _run_sync("ph1_kb_sid", source_dir, kb_dir)
    # Same content, same uri -> same source_id across runs
    s2 = _run_sync("ph1_kb_sid", source_dir, kb_dir)

    # s2 should skip s1's doc (no-op)
    assert s2.summary()["added"] == 0
    assert len(s1.added) == 1


def test_http_connector_builds_source_doc():
    from rag.ingest.connectors import HttpPageConnector
    c = HttpPageConnector("kb_x", ["https://example.com/a", "https://example.com/b"])
    docs = list(c.list_documents())
    assert len(docs) == 2
    assert all(d.suffix == ".html" for d in docs)
    assert all(d.source_id.startswith("kb_x::") for d in docs)
    # Stable source_id: same URL → same id on re-run
    docs2 = list(c.list_documents())
    assert {d.source_id for d in docs} == {d.source_id for d in docs2}
