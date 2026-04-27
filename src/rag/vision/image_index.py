"""Simple per-KB image index.

Stores one entry per indexed image:
    {source_id, title, source_path, embedding: [float] * 512}

Persisted as JSONL at ``<kb_dir>/image_index.jsonl`` so rebuilds don't
lose work. Search is linear scan (``np.dot`` on a single 2-D matrix);
fine up to ~100k images per KB. Swap for Faiss flat-IP when we cross
that threshold -- ``VectorStore`` already has the pattern.

API:
    idx = ImageIndex(kb_dir)
    idx.add(source_id, title, source_path, embedding)
    idx.save()
    idx = ImageIndex.load(kb_dir)     # classmethod
    hits = idx.search(query_vec, top_k=5)
    idx.remove(source_id)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


_FILE = "image_index.jsonl"


@dataclass
class ImageHit:
    source_id: str
    title: str
    source_path: str
    similarity: float


class ImageIndex:
    def __init__(self, kb_dir: Path):
        self.kb_dir = Path(kb_dir)
        self._entries: list[dict] = []

    # ---------- persistence ----------

    @classmethod
    def load(cls, kb_dir: Path) -> "ImageIndex":
        idx = cls(kb_dir)
        path = Path(kb_dir) / _FILE
        if not path.exists():
            return idx
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "source_id" in rec and "embedding" in rec:
                    idx._entries.append(rec)
        return idx

    def save(self) -> None:
        path = self.kb_dir / _FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for rec in self._entries:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------- mutation ----------

    def add(self, source_id: str, title: str, source_path: str, embedding: Iterable[float]) -> None:
        # Replace existing entry with the same source_id (upsert semantics).
        vec = [float(x) for x in embedding]
        self._entries = [e for e in self._entries if e.get("source_id") != source_id]
        self._entries.append({
            "source_id": source_id,
            "title": title,
            "source_path": source_path,
            "embedding": vec,
        })

    def remove(self, source_id: str) -> int:
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.get("source_id") != source_id]
        return before - len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    # ---------- search ----------

    def search(self, query_vec: np.ndarray, *, top_k: int = 5) -> list[ImageHit]:
        if not self._entries:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        qn = float(np.linalg.norm(q))
        if qn == 0.0:
            return []
        q = q / qn

        mat = np.asarray([e["embedding"] for e in self._entries], dtype=np.float32)
        # Re-normalise row-wise in case a caller stored unnormalised vectors.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        mat = mat / norms

        sims = mat @ q  # (N,)
        order = np.argsort(-sims)[:top_k]
        out: list[ImageHit] = []
        for i in order:
            rec = self._entries[int(i)]
            out.append(ImageHit(
                source_id=rec["source_id"],
                title=rec.get("title", ""),
                source_path=rec.get("source_path", ""),
                similarity=float(sims[int(i)]),
            ))
        return out
