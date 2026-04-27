"""Near-dup detection via MinHash LSH.

Usage pattern during incremental ingest:

    dedup = MinHashDeduper(threshold=0.85)
    for chunk_text in new_chunks:
        if dedup.is_dup(chunk_text):
            continue
        dedup.add(chunk_text)
        yield chunk_text

- ``threshold=0.85`` is the Jaccard similarity above which we call two
  chunks duplicates. 0.85 is empirically tight enough to catch pagination
  cruft without flagging "two paragraphs about the same topic" as dups.
- ``num_perm=128`` is the standard MinHash permutation count; doubling it
  is marginally more accurate and slower.
- Shingles are 5-gram at the character level, matching the ``datasketch``
  recipe for multilingual corpora.

Persistence is caller's problem. For a single ingest run we keep an
in-memory LSH. For cross-run dedup, a future upgrade could persist the
LSH to Redis or a SQLite ``lsh`` table — not needed for PH2.
"""
from __future__ import annotations

from dataclasses import dataclass

from datasketch import MinHash, MinHashLSH


def _shingles(text: str, k: int = 5) -> list[str]:
    if not text:
        return []
    norm = "".join(text.split())
    if len(norm) <= k:
        return [norm]
    return [norm[i : i + k] for i in range(len(norm) - k + 1)]


def _minhash(text: str, num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in _shingles(text):
        m.update(sh.encode("utf-8"))
    return m


@dataclass
class DedupeDecision:
    is_dup: bool
    matched_key: str | None = None
    jaccard_estimate: float = 0.0


class MinHashDeduper:
    """Track seen chunks; answer is_dup? for new ones."""

    def __init__(self, threshold: float = 0.85, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._hashes: dict[str, MinHash] = {}
        self._next_id = 0

    def decide(self, text: str) -> DedupeDecision:
        if not text.strip():
            return DedupeDecision(is_dup=True, matched_key=None)
        m = _minhash(text, self.num_perm)
        candidates = self._lsh.query(m)
        if candidates:
            key = candidates[0]
            prev = self._hashes[key]
            # Cheap estimate of true Jaccard via minhash overlap
            est = float(sum(1 for a, b in zip(m.hashvalues, prev.hashvalues) if a == b)) / self.num_perm
            return DedupeDecision(is_dup=True, matched_key=key, jaccard_estimate=est)
        return DedupeDecision(is_dup=False)

    def add(self, text: str) -> str:
        """Record this chunk. Returns the internal key used for LSH."""
        m = _minhash(text, self.num_perm)
        key = f"k{self._next_id}"
        self._next_id += 1
        self._hashes[key] = m
        self._lsh.insert(key, m)
        return key

    def is_dup(self, text: str) -> bool:
        return self.decide(text).is_dup

    def stats(self) -> dict[str, int]:
        return {"seen": len(self._hashes)}
