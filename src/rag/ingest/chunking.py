"""Structural chunker.

Strategy:
1. Split by markdown headings (H1/H2/H3) -> sections.
2. Within each section, split into paragraphs, then pack into windows of
   ~target_tokens with sliding overlap for long sections.
3. Each chunk carries section_path (breadcrumb) so answers can cite precise locations.

Token count is approximated by whitespace split + 1.3x for CJK character density.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
CJK_RE = re.compile(r"[一-鿿]")


def approx_tokens(text: str) -> int:
    """Cheap token estimate. Good enough for chunking thresholds."""
    if not text:
        return 0
    ascii_tokens = len(re.findall(r"[A-Za-z0-9_]+", text))
    cjk_chars = len(CJK_RE.findall(text))
    # CJK chars roughly 1 token each for BGE-M3 tokenizer.
    return ascii_tokens + cjk_chars


@dataclass
class Section:
    path: list[str]
    text: str


def split_sections(text: str) -> list[Section]:
    """Split markdown/text into sections by headings. No heading -> single section."""
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [Section(path=[], text=text.strip())]

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, heading)

    if matches[0].start() > 0:
        preface = text[: matches[0].start()].strip()
        if preface:
            sections.append(Section(path=[], text=preface))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))
        path = [h for _, h in stack]

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append(Section(path=list(path), text=body))

    return sections


def pack_paragraphs(
    paragraphs: list[str], target_tokens: int, overlap_tokens: int
) -> list[str]:
    """Greedy pack paragraphs into chunks near target_tokens, with tail overlap."""
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if current:
            chunks.append("\n\n".join(current).strip())
            current, current_tokens = [], 0

    for para in paragraphs:
        t = approx_tokens(para)
        if t >= target_tokens * 1.5:
            # Oversized paragraph: hard-split by sentences.
            flush()
            for piece in _split_long(para, target_tokens):
                chunks.append(piece)
            continue
        if current_tokens + t > target_tokens and current:
            flush()
        current.append(para)
        current_tokens += t

    flush()

    if overlap_tokens > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for prev, nxt in zip(chunks, chunks[1:]):
            tail = _take_tail(prev, overlap_tokens)
            overlapped.append(f"{tail}\n\n{nxt}" if tail else nxt)
        chunks = overlapped

    return chunks


def _split_long(text: str, target_tokens: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?。!?])\s+|\n+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    pieces: list[str] = []
    buf: list[str] = []
    tokens = 0
    for s in sentences:
        t = approx_tokens(s)
        if tokens + t > target_tokens and buf:
            pieces.append(" ".join(buf))
            buf, tokens = [], 0
        buf.append(s)
        tokens += t
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def _take_tail(text: str, tokens: int) -> str:
    words = text.split()
    if approx_tokens(text) <= tokens:
        return text
    tail: list[str] = []
    total = 0
    for w in reversed(words):
        t = approx_tokens(w) or 1
        if total + t > tokens:
            break
        tail.insert(0, w)
        total += t
    return " ".join(tail)


@dataclass
class ChunkPayload:
    section_path: list[str]
    text: str
    parent_local_id: str | None  # "sec0", "sec1" etc. — caller turns this into global parent_id
    role: str                     # "leaf" or "parent"


def chunk_document(
    text: str,
    *,
    target_tokens: int,
    overlap_tokens: int,
    parent_target_tokens: int = 900,
) -> list[ChunkPayload]:
    """Parent-child chunker.

    For each heading-bounded section we emit ONE parent chunk holding the full
    section (capped at ``parent_target_tokens`` -- long sections become multiple
    parents). Then we split that section into small leaves (``target_tokens``
    with sliding overlap). Each leaf is linked to its parent by ``parent_local_id``.

    Indexing code must only embed leaves; parents exist solely to be handed to
    the LLM at answer-time so the generator sees wider context than the
    precision-matched leaf snippet.
    """
    sections = split_sections(text)
    out: list[ChunkPayload] = []
    parent_counter = 0

    for section in sections:
        # Long sections get split into multiple parent windows so we don't
        # hand the LLM an entire 5k-token chapter.
        section_tokens = approx_tokens(section.text)
        if section_tokens <= parent_target_tokens:
            parent_windows = [section.text]
        else:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section.text) if p.strip()]
            parent_windows = pack_paragraphs(paragraphs, parent_target_tokens, overlap_tokens=0)

        for parent_text in parent_windows:
            parent_local_id = f"p{parent_counter}"
            parent_counter += 1
            out.append(
                ChunkPayload(
                    section_path=section.path,
                    text=parent_text,
                    parent_local_id=parent_local_id,
                    role="parent",
                )
            )
            # Leaves inside this parent
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", parent_text) if p.strip()]
            leaves = pack_paragraphs(paragraphs, target_tokens, overlap_tokens)
            for leaf_text in leaves:
                out.append(
                    ChunkPayload(
                        section_path=section.path,
                        text=leaf_text,
                        parent_local_id=parent_local_id,
                        role="leaf",
                    )
                )

    return out
