"""Incremental ingest: diff a connector's current document set against the
KBDocument registry, and add/update/delete accordingly.

Flow
----
1. List documents from the connector (cheap enumeration, no fetch yet).
2. Index existing KBDocument rows by source_id.
3. For each live doc:
     - not in registry        → FETCH, chunk, insert rows/chunks, write registry.
     - in registry, same hash → SKIP (no network, no LLM).
     - in registry, new hash  → FETCH, DELETE old chunks, chunk+insert, bump registry.
4. For each registry row whose source_id isn't in live docs:
     - DELETE chunks from BM25 + vector, drop registry row.

"Hash" here is ``sha256(bytes)``. We prefer the connector's ``cheap_hash``
(mtime+size for local; ETag for HTTP) to decide whether to even fetch;
the real content hash goes into the registry so follow-up runs are still
correct if someone modifies cheap hashes by hand.

Persistence story
-----------------
All registry mutations are through ``SessionLocal`` (Postgres in Docker,
SQLite locally). All chunk mutations go through the VectorStore + BM25
``upsert_chunks`` / ``delete_by_source_id`` pair, which we added in
alongside this module.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from sqlalchemy import select

from ..config import settings
from ..db.base import SessionLocal
from ..db.models import KBDocument
from ..index.bm25 import BM25Index
from ..index.vectorstore import get_vector_store
from ..ingest.chunking import approx_tokens, chunk_document
from ..ingest.clean import (
    MinHashDeduper,
    is_low_info_chunk,
    redact_pii,
    strip_html_boilerplate,
)
from ..ingest.loaders import LOADERS
from ..logging import get_logger
from ..types import Chunk
from .connectors.base import Connector, SourceDoc


log = get_logger(__name__)


@dataclass
class SyncStats:
    kb_id: str
    added: list[str] = field(default_factory=list)      # source_ids
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)  # (source_id, error)

    # PH2 cleaning counters -- aggregate across docs for an ingest summary.
    pii_redactions: int = 0
    low_info_dropped: int = 0
    dedup_dropped: int = 0

    def summary(self) -> dict[str, int]:
        return {
            "added": len(self.added),
            "updated": len(self.updated),
            "skipped": len(self.skipped),
            "deleted": len(self.deleted),
            "errors": len(self.errors),
            "pii_redactions": self.pii_redactions,
            "low_info_dropped": self.low_info_dropped,
            "dedup_dropped": self.dedup_dropped,
        }


@dataclass
class CleanConfig:
    """Toggles for the PH2 cleaning passes. Keep all off = back-compat."""
    html_boilerplate: bool = False
    low_info_filter: bool = False
    low_info_threshold: float = 0.7
    pii_redact: bool = False
    dedup: bool = False
    dedup_threshold: float = 0.85


def _content_hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _parse_bytes_to_title_and_text(doc: SourceDoc, raw: bytes) -> tuple[str, str]:
    """Route suffix → loader. Loaders take ``Path``; we materialise a temp."""
    loader = LOADERS.get(doc.suffix)
    if loader is None:
        raise ValueError(f"unsupported suffix: {doc.suffix}")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=doc.suffix, delete=False) as f:
        f.write(raw)
        tmp = Path(f.name)
    try:
        return loader(tmp)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _chunks_for_doc(
    doc: SourceDoc,
    raw: bytes,
    kb_id: str,
    *,
    clean_cfg: CleanConfig | None = None,
    dedup: MinHashDeduper | None = None,
    stats: SyncStats | None = None,
) -> list[Chunk]:
    """Parse bytes → (optionally clean) → chunk → (optionally filter) → Chunk[].

    Cleaning passes (all opt-in via ``clean_cfg``):
      1. HTML boilerplate strip (only for .html / .htm docs)
      2. PII redaction on doc text
      3. Per-chunk low-info filter
      4. Cross-doc MinHash dedup
    """
    from ..config import settings
    cfg = clean_cfg or CleanConfig()

    title, text = _parse_bytes_to_title_and_text(doc, raw)

    if cfg.html_boilerplate and doc.suffix in {".html", ".htm"}:
        try:
            html_title, cleaned = strip_html_boilerplate(raw.decode("utf-8", errors="ignore"))
            if cleaned.strip():
                text = cleaned
                title = html_title or title
        except Exception as e:  # noqa: BLE001 -- never block ingest on cleaner failure
            log.warning("html_clean_failed", source_id=doc.source_id, error=str(e))

    if cfg.pii_redact:
        text, report = redact_pii(text)
        if stats is not None:
            stats.pii_redactions += report.total
        if report.total:
            log.info("pii_redacted", source_id=doc.source_id, **{k: v for k, v in report.__dict__.items() if k not in {"samples"} and isinstance(v, int) and v > 0})

    if not text.strip():
        return []

    payloads = chunk_document(
        text,
        target_tokens=settings.chunk_target_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )

    source_id = doc.source_id
    parent_global: dict[str, str] = {}
    for i, p in enumerate(payloads):
        if p.role == "parent" and p.parent_local_id:
            parent_global[p.parent_local_id] = f"{source_id}::p{i:04d}"

    results: list[Chunk] = []
    for i, p in enumerate(payloads):
        if p.role == "parent":
            chunk_id = parent_global[p.parent_local_id]
            parent_id = None
            # Parents are never indexed, so don't run low-info / dedup on them.
            results.append(
                Chunk(
                    chunk_id=chunk_id, kb_id=kb_id, source_id=source_id, source_path=doc.uri,
                    title=doc.title or title, section_path=p.section_path, text=p.text,
                    token_count=approx_tokens(p.text), order=i,
                    metadata={"connector": "via_bytes", "suffix": doc.suffix},
                    parent_id=parent_id, chunk_role=p.role,
                )
            )
            continue

        chunk_id = f"{source_id}::l{i:04d}"
        parent_id = parent_global.get(p.parent_local_id) if p.parent_local_id else None

        if cfg.low_info_filter and is_low_info_chunk(p.text, threshold=cfg.low_info_threshold):
            if stats is not None:
                stats.low_info_dropped += 1
            log.info("low_info_dropped", source_id=source_id, order=i, preview=p.text[:80])
            continue

        if cfg.dedup and dedup is not None:
            decision = dedup.decide(p.text)
            if decision.is_dup:
                if stats is not None:
                    stats.dedup_dropped += 1
                log.info("dedup_dropped", source_id=source_id, order=i, matched=decision.matched_key, jaccard=round(decision.jaccard_estimate, 3))
                continue
            dedup.add(p.text)

        results.append(
            Chunk(
                chunk_id=chunk_id, kb_id=kb_id, source_id=source_id, source_path=doc.uri,
                title=doc.title or title, section_path=p.section_path, text=p.text,
                token_count=approx_tokens(p.text), order=i,
                metadata={"connector": "via_bytes", "suffix": doc.suffix},
                parent_id=parent_id, chunk_role=p.role,
            )
        )
    return results


def sync_kb(
    kb_id: str,
    connector: Connector,
    kb_dir: Path,
    *,
    clean_cfg: CleanConfig | None = None,
    enrich_metadata: bool | None = None,
) -> SyncStats:
    """Bring ``kb_id``'s chunks + registry into sync with the connector.

    ``clean_cfg`` toggles PH2 cleaning passes (HTML boilerplate strip,
    low-info filter, PII redact, dedup). Omit / leave default = back-compat
    with PH1 behaviour.

    ``enrich_metadata`` toggles the sprint A.5 LLM-based metadata extraction
    (entities / topics / date) per leaf chunk. Defaults to
    ``settings.enable_metadata_enrichment`` (False unless env flipped).

    Side effects:
      - mutates ``kb_documents`` table (insert / update / delete)
      - mutates BM25 DB and vector store at the chunk level
    """
    cfg = clean_cfg or CleanConfig()
    enrich = enrich_metadata if enrich_metadata is not None else settings.enable_metadata_enrichment
    vector = get_vector_store(kb_dir, kb_id)
    bm25 = BM25Index(kb_dir / "bm25.sqlite")
    stats = SyncStats(kb_id=kb_id)
    # Dedup state is per-run (cross-doc within one sync). Cross-run dedup
    # would persist MinHashes; out of PH2 scope.
    dedup_state = MinHashDeduper(threshold=cfg.dedup_threshold) if cfg.dedup else None

    # Load existing registry for this kb.
    with SessionLocal() as s:
        existing_rows = list(s.execute(select(KBDocument).where(KBDocument.kb_id == kb_id)).scalars())
    existing: dict[str, KBDocument] = {r.source_id: r for r in existing_rows}

    live_ids: set[str] = set()
    for doc in connector.list_documents():
        live_ids.add(doc.source_id)
        prev = existing.get(doc.source_id)

        try:
            # Cheap-hash short-circuit: if the connector says nothing's changed
            # AND we already have a row, skip without fetching.
            if prev is not None and doc.cheap_hash and prev.extra.get("cheap_hash") == doc.cheap_hash:
                stats.skipped.append(doc.source_id)
                continue

            raw = connector.fetch_bytes(doc)
            chash = _content_hash(raw)

            if prev is not None and prev.content_hash == chash:
                # Same bytes despite different cheap hash (e.g. mtime bumped by `touch`)
                # Still worth refreshing cheap_hash so next run short-circuits.
                with SessionLocal() as s:
                    row = s.get(KBDocument, doc.source_id)
                    if row is not None:
                        row.extra = {**(row.extra or {}), "cheap_hash": doc.cheap_hash}
                        s.commit()
                stats.skipped.append(doc.source_id)
                continue

            new_chunks = _chunks_for_doc(doc, raw, kb_id, clean_cfg=cfg, dedup=dedup_state, stats=stats)
            if enrich and new_chunks:
                from .metadata_enrich import enrich_chunks
                new_chunks = enrich_chunks(new_chunks, kb_dir)
            # Replace semantics: delete old, insert new.
            vector.delete_by_source_id(doc.source_id)
            bm25.delete_by_source_id(doc.source_id)
            if new_chunks:
                vector.upsert_chunks(new_chunks)
                bm25.upsert_chunks(new_chunks)

            with SessionLocal() as s:
                row = s.get(KBDocument, doc.source_id)
                if row is None:
                    s.add(KBDocument(
                        source_id=doc.source_id,
                        kb_id=kb_id,
                        connector=connector.name,
                        source_uri=doc.uri,
                        content_hash=chash,
                        title=doc.title,
                        chunk_count=sum(1 for c in new_chunks if c.chunk_role == "leaf"),
                        extra={"cheap_hash": doc.cheap_hash, "suffix": doc.suffix, **(doc.extra or {})},
                    ))
                    s.commit()
                    stats.added.append(doc.source_id)
                else:
                    row.content_hash = chash
                    row.title = doc.title or row.title
                    row.source_uri = doc.uri
                    row.chunk_count = sum(1 for c in new_chunks if c.chunk_role == "leaf")
                    row.extra = {**(row.extra or {}), "cheap_hash": doc.cheap_hash, "suffix": doc.suffix, **(doc.extra or {})}
                    s.commit()
                    stats.updated.append(doc.source_id)

        except Exception as e:  # noqa: BLE001 -- keep syncing siblings
            log.warning("sync_doc_error", source_id=doc.source_id, uri=doc.uri, error=str(e))
            stats.errors.append((doc.source_id, str(e)))

    # Deletions: registry has rows the connector didn't yield.
    gone = [sid for sid in existing if sid not in live_ids]
    for sid in gone:
        try:
            vector.delete_by_source_id(sid)
            bm25.delete_by_source_id(sid)
            with SessionLocal() as s:
                row = s.get(KBDocument, sid)
                if row is not None:
                    s.delete(row)
                    s.commit()
            stats.deleted.append(sid)
        except Exception as e:  # noqa: BLE001
            log.warning("sync_delete_error", source_id=sid, error=str(e))
            stats.errors.append((sid, str(e)))

    log.info("sync_kb_done", kb_id=kb_id, **stats.summary())
    return stats
