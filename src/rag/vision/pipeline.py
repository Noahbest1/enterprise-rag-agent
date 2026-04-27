"""End-to-end image → chunks pipeline.

Use cases:
  1. Ingest an image into a KB at build time.
     → preprocess → layout analyse → emit region-typed chunks
     → chunks carry metadata.region_type so later retrieval can filter
  2. Answer a user question about an uploaded image at query time.
     → preprocess → cached describe → return description

Caching is keyed by pixel hash so "same image again" (rename, recompress,
re-upload) is a cache hit for describe / layout / OCR.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..logging import get_logger
from ..types import Chunk
from .cache import cached_vision_call
from .layout import Region, analyse_layout
from .preprocess import ProcessedImage, preprocess_image
from .vlm import describe_image, extract_text_from_image


log = get_logger(__name__)


def _regions_to_json(regions: list[Region]) -> list[dict]:
    return [asdict(r) for r in regions]


def describe_image_cached(raw: bytes, question: str = "") -> dict:
    """Preprocess + cached VLM describe. Returns a dict with description + metadata."""
    pimg = preprocess_image(raw)

    def _run():
        text = describe_image(pimg.bytes_, question)
        return {"description": text, "pixel_hash": pimg.pixel_hash, "width": pimg.width, "height": pimg.height}

    return cached_vision_call(pimg.pixel_hash, task=f"describe:{hash(question) & 0xffffffff}", runner=_run)


def ocr_image_cached(raw: bytes) -> dict:
    """Preprocess + cached OCR transcription."""
    pimg = preprocess_image(raw)

    def _run():
        text = extract_text_from_image(pimg.bytes_)
        return {"ocr_text": text, "pixel_hash": pimg.pixel_hash}

    return cached_vision_call(pimg.pixel_hash, task="ocr", runner=_run)


def analyse_layout_cached(raw: bytes) -> dict:
    """Preprocess + cached layout-aware region analysis."""
    pimg = preprocess_image(raw)

    def _run():
        regions = analyse_layout(pimg.bytes_)
        return {"regions": _regions_to_json(regions), "pixel_hash": pimg.pixel_hash}

    return cached_vision_call(pimg.pixel_hash, task="layout", runner=_run)


def image_to_chunks(
    raw: bytes,
    *,
    kb_id: str,
    source_id: str,
    source_uri: str,
    title: str | None = None,
) -> list[Chunk]:
    """Turn one image into region-typed ``Chunk`` objects ready for indexing.

    - Tables are kept as a single chunk each (atomic), text rendered as
      their markdown so BM25/vector can match cell content.
    - Figures are indexed via their VLM caption; the caption is the text
      BM25 sees, so "查图表里红色柱" can match on the caption's words.
    - Titles and text regions become regular chunks.
    - Code regions are indexed but kept atomic (no cross-block splits).

    For truly long images that produce many regions we still emit each as
    its own chunk (no further packing) -- region-level is the meaningful
    granularity for this use case.
    """
    from ..config import settings

    pimg = preprocess_image(raw)
    payload = cached_vision_call(
        pimg.pixel_hash,
        task="layout_for_ingest",
        runner=lambda: {"regions": _regions_to_json(analyse_layout(pimg.bytes_))},
    )
    regions_raw = payload.get("regions") or []
    title = title or "(image)"

    # Emit a synthetic parent chunk that concatenates all regions -- lets
    # the parent-expand retrieval return the full page when any region hits.
    parent_text_parts: list[str] = []
    chunks: list[Chunk] = []
    parent_id = f"{source_id}::p0000"

    for i, r in enumerate(regions_raw):
        rtype = (r.get("type") or "text").lower()
        if rtype == "table":
            body = r.get("markdown") or r.get("text") or ""
        elif rtype == "figure":
            caption = r.get("text") or ""
            body = f"[Figure] {caption}".strip()
        elif rtype == "code":
            lang = r.get("language") or ""
            body = f"[Code:{lang}]\n{r.get('text') or ''}"
        elif rtype == "title":
            body = f"# {r.get('text') or ''}".strip()
        else:  # text / unknown
            body = r.get("text") or ""

        body = body.strip()
        if not body:
            continue

        parent_text_parts.append(body)
        chunks.append(Chunk(
            chunk_id=f"{source_id}::l{i:04d}",
            kb_id=kb_id,
            source_id=source_id,
            source_path=source_uri,
            title=title,
            section_path=[],
            text=body,
            token_count=len(body.split()),
            order=i,
            metadata={
                "region_type": rtype,
                "pixel_hash": pimg.pixel_hash,
                "suffix": ".image",
                "applied": pimg.applied,
            },
            parent_id=parent_id,
            chunk_role="leaf",
        ))

    if chunks:
        parent_text = "\n\n".join(parent_text_parts)
        parent = Chunk(
            chunk_id=parent_id,
            kb_id=kb_id,
            source_id=source_id,
            source_path=source_uri,
            title=title,
            section_path=[],
            text=parent_text,
            token_count=len(parent_text.split()),
            order=-1,
            metadata={"region_type": "image_page", "pixel_hash": pimg.pixel_hash},
            parent_id=None,
            chunk_role="parent",
        )
        return [parent, *chunks]

    return chunks
