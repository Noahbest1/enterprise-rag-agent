from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    kb_id: str
    source_id: str
    source_path: str
    title: str
    section_path: list[str]
    text: str
    token_count: int
    order: int
    metadata: dict[str, Any] = field(default_factory=dict)
    # Parent-child (small-to-big) retrieval:
    # - chunk_role="leaf": indexed by BM25 + vector; small, precise match
    # - chunk_role="parent": NOT indexed; returned to LLM for wider context
    parent_id: str | None = None
    chunk_role: str = "leaf"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Hit:
    chunk_id: str
    score: float
    text: str
    title: str
    source_id: str
    source_path: str
    section_path: list[str]
    retrieval_source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    n: int
    source_id: str
    source_path: str
    title: str
    section_path: list[str]
    snippet: str


@dataclass
class Answer:
    query: str
    rewritten_query: str
    text: str
    citations: list[Citation]
    abstained: bool
    reason: str
    hits: list[Hit]
    latency_ms: int
    trace: dict[str, Any] = field(default_factory=dict)
