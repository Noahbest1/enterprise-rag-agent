"""Connector abstraction: the thing that turns "some external source" into
a stream of ``SourceDoc`` records we can incrementally ingest.

A connector has two primitives:

  * ``list_documents()`` -- enumerate candidate documents in the source.
    Returns lightweight descriptors (id, uri, content hash if cheap, etc).
  * ``fetch_bytes(doc)`` -- pull the actual bytes for one document.

We separate the two so the incremental ingest logic can cheaply diff the
DB against the connector output (hash comparison) and only fetch what it
actually needs to (re-)process.

``SourceDoc.source_id`` is the key the chunk-level schema uses: same
across BM25 chunks + vector rows + ``KBDocument`` registry. Keep it
stable across runs for the same document so updates don't orphan chunks.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class SourceDoc:
    source_id: str           # stable primary key across runs
    uri: str                 # /path/to/file.md, https://..., notion://page/abc, ...
    suffix: str              # ".md" / ".html" / ".pdf" / ".txt"
    title: str | None = None
    cheap_hash: str | None = None  # optional (e.g. mtime+size) -- avoids fetch if unchanged
    extra: dict = field(default_factory=dict)


def stable_source_id(kb_id: str, canonical_key: str) -> str:
    """Derive a deterministic source_id from kb + any canonical identifier.

    ``canonical_key`` is what uniquely identifies a document within its source
    (e.g. absolute file path; normalised URL; Notion page id). Changing the
    canonical_key means a different document -- so choose something the
    upstream system wouldn't rewrite on reflow.
    """
    h = hashlib.sha1(canonical_key.encode("utf-8")).hexdigest()[:16]
    return f"{kb_id}::{h}"


class Connector(ABC):
    """Pull-mode connector base class."""

    name: str = "base"

    @abstractmethod
    def list_documents(self) -> Iterable[SourceDoc]:
        ...

    @abstractmethod
    def fetch_bytes(self, doc: SourceDoc) -> bytes:
        ...

    def describe(self) -> dict:
        """Short self-descriptor for audit logs / CLI prints."""
        return {"connector": self.name}
