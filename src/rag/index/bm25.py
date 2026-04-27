"""BM25 via SQLite FTS5 with CJK-aware tokenization.

Key design choices that matter for accuracy:

1. FTS5 table uses ``unicode61`` with ``categories 'L* N* Co'`` so it keeps
   digits, letters (including CJK), and private-use chars. Default FTS5
   tokenizer strips CJK; we fix that here.

2. Indexing side does character-level split on CJK: "数据连接器" is stored
   as four co-occurring tokens "数 据 连 接 器". FTS5 queries joined by AND
   then match on overlapping character subsequences -- a cheap substitute
   for a proper Chinese analyzer, and good enough at chunk granularity.

3. Query side tokenizes both ASCII words and CJK characters, joins ASCII
   tokens with ``OR`` (prefix matched) and CJK tokens into phrase blocks so
   partial-word matches still rank.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterable

from ..types import Chunk, Hit


CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]")
ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

# Small, conservative stoplist. BM25 handles rare-term weighting itself, so
# we only strip genuinely noisy function words.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "have",
    "i", "if", "in", "into", "is", "it", "of", "on", "or", "that", "the",
    "to", "was", "were", "will", "with",
}


def _cjk_split(text: str) -> str:
    """Insert spaces between CJK characters so unicode61 treats each as a token."""
    return CJK_RE.sub(lambda m: f" {m.group(0)} ", text)


def _prepare_for_index(text: str) -> str:
    return _cjk_split(text.lower())


def _tokenize_query(query: str) -> tuple[list[str], list[str]]:
    """Return (ascii_tokens, cjk_chars)."""
    q = query.lower()
    ascii_tokens: list[str] = []
    seen_ascii: set[str] = set()
    for tok in ASCII_TOKEN_RE.findall(q):
        if tok in STOPWORDS or len(tok) <= 1:
            continue
        if tok not in seen_ascii:
            seen_ascii.add(tok)
            ascii_tokens.append(tok)

    cjk_chars: list[str] = []
    for ch in q:
        if CJK_RE.match(ch):
            cjk_chars.append(ch)
    return ascii_tokens, cjk_chars


def compile_match_query(query: str) -> str:
    ascii_tokens, cjk_chars = _tokenize_query(query)
    parts: list[str] = []
    if ascii_tokens:
        parts.append("(" + " OR ".join(f"{t}*" for t in ascii_tokens) + ")")
    if cjk_chars:
        # Characters as AND group: roughly requires all CJK chars to appear in chunk.
        parts.append("(" + " AND ".join(cjk_chars) + ")")
    if not parts:
        return ""
    return " OR ".join(parts)


class BM25Index:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def delete_by_source_id(self, source_id: str) -> int:
        """Remove all rows for a doc. FTS5 supports WHERE on UNINDEXED cols."""
        if not self.db_path.exists():
            return 0
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.execute("DELETE FROM chunks_fts WHERE source_id = ?", (source_id,))
            conn.commit()
            return cur.rowcount or 0
        finally:
            conn.close()

    def upsert_chunks(self, chunks) -> int:
        """Append leaves to the FTS5 table; caller ensures dedupe by source_id first."""
        from .vectorstore import VectorStore  # noqa: F401  (circular-safe lazy)
        leaves = [c for c in chunks if c.chunk_role == "leaf"]
        if not leaves:
            return 0
        if not self.db_path.exists():
            # First write -- bootstrap the FTS5 schema using a single-leaf build.
            return self.build(leaves)

        conn = sqlite3.connect(str(self.db_path))
        try:
            with conn:
                for c in leaves:
                    conn.execute(
                        "INSERT INTO chunks_fts (chunk_id, kb_id, source_id, source_path, parent_id, title, section_path, text) VALUES (?,?,?,?,?,?,?,?)",
                        (
                            c.chunk_id,
                            c.kb_id,
                            c.source_id,
                            c.source_path,
                            c.parent_id or "",
                            _prepare_for_index(c.title),
                            _prepare_for_index(" / ".join(c.section_path)),
                            _prepare_for_index(c.text),
                        ),
                    )
            return len(leaves)
        finally:
            conn.close()

    def build(self, chunks: Iterable[Chunk]) -> int:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.db_path.exists():
            self.db_path.unlink()
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA journal_mode=DELETE;")
            conn.execute(
                """
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    kb_id UNINDEXED,
                    source_id UNINDEXED,
                    source_path UNINDEXED,
                    parent_id UNINDEXED,
                    title,
                    section_path,
                    text,
                    tokenize = "unicode61 categories 'L* N* Co'"
                );
                """
            )
            count = 0
            with conn:
                for c in chunks:
                    # Index only leaves. Parents exist for LLM context only.
                    if c.chunk_role != "leaf":
                        continue
                    conn.execute(
                        "INSERT INTO chunks_fts (chunk_id, kb_id, source_id, source_path, parent_id, title, section_path, text) VALUES (?,?,?,?,?,?,?,?)",
                        (
                            c.chunk_id,
                            c.kb_id,
                            c.source_id,
                            c.source_path,
                            c.parent_id or "",
                            _prepare_for_index(c.title),
                            _prepare_for_index(" / ".join(c.section_path)),
                            _prepare_for_index(c.text),
                        ),
                    )
                    count += 1
            return count
        finally:
            conn.close()

    def search(self, query: str, limit: int) -> list[Hit]:
        match = compile_match_query(query)
        if not match:
            return []
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.execute(
                """
                SELECT chunk_id, kb_id, source_id, source_path, parent_id, title, section_path, text,
                       bm25(chunks_fts) AS score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (match, limit),
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

        hits: list[Hit] = []
        for chunk_id, kb_id, source_id, source_path, parent_id, title, section_path, text, raw_score in rows:
            # sqlite bm25() returns negative values; lower is better. Map to positive.
            score = -float(raw_score) if raw_score is not None else 0.0
            section = [s for s in section_path.split(" / ") if s] if section_path else []
            hits.append(
                Hit(
                    chunk_id=chunk_id,
                    score=score,
                    text=text,
                    title=title,
                    source_id=source_id,
                    source_path=source_path,
                    section_path=section,
                    retrieval_source="bm25",
                    metadata={"kb_id": kb_id, "parent_id": parent_id or None},
                )
            )
        return hits
