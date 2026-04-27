"""Local directory connector.

Walks a directory tree and emits one ``SourceDoc`` per supported file.
The ``cheap_hash`` is ``mtime:size`` so incremental ingest can skip files
that obviously haven't changed without opening them; the full content
hash is computed only when ``cheap_hash`` differs.

Safe on binary files for the supported suffixes (we never decode before
hashing -- the chunker later does the decoding and fails loudly on
unsupported formats).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from ..loaders import supported_extensions
from .base import Connector, SourceDoc, stable_source_id


class LocalDirConnector(Connector):
    name = "local"

    def __init__(self, kb_id: str, root: str | Path, *, recursive: bool = True):
        self.kb_id = kb_id
        self.root = Path(root).expanduser().resolve()
        self.recursive = recursive

    def describe(self) -> dict:
        return {"connector": self.name, "root": str(self.root), "recursive": self.recursive}

    def _iter_paths(self) -> Iterator[Path]:
        exts = supported_extensions()
        iterator = self.root.rglob("*") if self.recursive else self.root.glob("*")
        for p in iterator:
            if p.is_file() and p.suffix.lower() in exts:
                yield p

    def list_documents(self) -> Iterable[SourceDoc]:
        for p in self._iter_paths():
            try:
                stat = p.stat()
                cheap = f"{int(stat.st_mtime_ns)}:{stat.st_size}"
            except OSError:
                cheap = None
            yield SourceDoc(
                source_id=stable_source_id(self.kb_id, str(p.resolve())),
                uri=str(p.resolve()),
                suffix=p.suffix.lower(),
                title=p.stem,
                cheap_hash=cheap,
            )

    def fetch_bytes(self, doc: SourceDoc) -> bytes:
        return Path(doc.uri).read_bytes()
