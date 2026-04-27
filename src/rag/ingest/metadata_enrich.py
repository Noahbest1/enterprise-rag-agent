"""LLM-based metadata enrichment at ingest time.

For each leaf chunk we ask a small LLM to extract a trio of structured facts:

- ``entities`` -- proper nouns mentioned (products, versions, people, places)
- ``topics``   -- 1-3 short topic tags (e.g. ["return_policy", "shipping"])
- ``date``     -- ISO-8601 date mentioned in the chunk (YYYY-MM-DD), or null

Why this matters:
- Metadata filter ("entities_contains: iPhone 16 Pro") becomes useful only
  after chunks actually carry entity metadata.
- Temporal filter ("date_after: 2025-01-01") unlocks release-note queries.

This step is OPT-IN because it costs one LLM call per chunk at build time.
Toggle via ``settings.enable_metadata_enrichment`` or the ``enrich_metadata``
kwarg on ``sync_kb``. Results are cached per-chunk in
``<kb_dir>/metadata_enrich_cache.jsonl`` so rebuilds don't re-pay.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..answer.llm_client import LLMError, chat_once
from ..config import settings
from ..logging import get_logger
from ..types import Chunk


log = get_logger(__name__)

SYSTEM_PROMPT = """You extract structured metadata from a chunk of text. Return ONLY valid JSON with these three fields:

  "entities": a list of 0-6 proper nouns mentioned in the text (product names, versions, people, places, specific feature names). Do NOT include generic nouns like "user" or "file". Deduplicate case-insensitively.
  "topics":   a list of 1-3 short, lowercase, underscore_separated topic tags summarising what the chunk is about (e.g. "return_policy", "plus_membership", "shipping_rules").
  "date":     an ISO-8601 date (YYYY-MM-DD) if a specific calendar date appears in the text; otherwise null.

Rules:
- Output JSON only. No prose, no markdown fences, no explanation.
- Keep entities short (<= 40 chars each).
- If the chunk is boilerplate (navigation, copyright) return empty lists and null date.
"""


def _cache_path(kb_dir: Path) -> Path:
    return kb_dir / "metadata_enrich_cache.jsonl"


def _cache_key(chunk: Chunk) -> str:
    h = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:16]
    return f"{chunk.chunk_id}:{h}"


def _load_cache(kb_dir: Path) -> dict[str, dict]:
    path = _cache_path(kb_dir)
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[rec["key"]] = rec["value"]
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def _append_cache(kb_dir: Path, key: str, value: dict) -> None:
    path = _cache_path(kb_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def _parse_json_loose(text: str) -> dict | None:
    """Parse JSON even if the LLM wrapped it in a code fence."""
    text = text.strip()
    if text.startswith("```"):
        # Drop fences
        lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try grabbing the first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None


def _normalise(data: dict | None) -> dict:
    """Coerce model output into our schema. Returns safe defaults on bad input."""
    if not isinstance(data, dict):
        return {"entities": [], "topics": [], "date": None}
    raw_ents = data.get("entities") or []
    raw_topics = data.get("topics") or []
    date = data.get("date")

    entities: list[str] = []
    if isinstance(raw_ents, list):
        seen: set[str] = set()
        for e in raw_ents:
            if not isinstance(e, str):
                continue
            s = e.strip()
            if not s or len(s) > 40:
                continue
            lower = s.lower()
            if lower in seen:
                continue
            seen.add(lower)
            entities.append(s)
            if len(entities) >= 6:
                break

    topics: list[str] = []
    if isinstance(raw_topics, list):
        for t in raw_topics[:3]:
            if not isinstance(t, str):
                continue
            # lowercase + whitespace -> underscore
            tag = "_".join(t.strip().lower().split())
            if tag and len(tag) <= 40:
                topics.append(tag)

    if isinstance(date, str):
        # Basic ISO-8601 YYYY-MM-DD shape check
        if len(date) == 10 and date[4] == "-" and date[7] == "-":
            try:
                int(date[:4]); int(date[5:7]); int(date[8:10])
            except ValueError:
                date = None
        else:
            date = None
    else:
        date = None

    return {"entities": entities, "topics": topics, "date": date}


def enrich_chunk_metadata(chunk: Chunk) -> dict[str, Any]:
    """One-shot LLM extract. Returns ``{"entities", "topics", "date"}``.

    On LLM error / no key / bad JSON returns safe defaults (empty / null) so
    ingest never hard-fails.
    """
    user = (
        f"Title: {chunk.title}\n"
        f"Section: {' / '.join(chunk.section_path) if chunk.section_path else ''}\n"
        f"Text:\n{chunk.text[:2000]}\n\n"
        f"Return the JSON."
    )
    try:
        reply = chat_once(system=SYSTEM_PROMPT, user=user, max_output_tokens=200,
                          temperature=0.0, task="intent")
    except LLMError as e:
        log.warning("metadata_enrich_llm_failed", chunk_id=chunk.chunk_id, error=str(e))
        return {"entities": [], "topics": [], "date": None}
    return _normalise(_parse_json_loose(reply))


def enrich_chunks(chunks: list[Chunk], kb_dir: Path) -> list[Chunk]:
    """Return ``chunks`` with per-chunk metadata merged in. Leaf-only.

    Uses a per-chunk disk cache so reruns are free. Writes are append-only
    so a crash mid-run doesn't corrupt earlier work.
    """
    if not chunks:
        return chunks
    cache = _load_cache(kb_dir)
    out: list[Chunk] = []
    done = 0
    total = sum(1 for c in chunks if c.chunk_role == "leaf")
    for c in chunks:
        if c.chunk_role != "leaf":
            out.append(c)
            continue
        key = _cache_key(c)
        value = cache.get(key)
        if value is None:
            value = enrich_chunk_metadata(c)
            if value and (value.get("entities") or value.get("topics") or value.get("date")):
                _append_cache(kb_dir, key, value)
                cache[key] = value
        new_meta = dict(c.metadata or {})
        new_meta["entities"] = value.get("entities") or []
        new_meta["topics"] = value.get("topics") or []
        if value.get("date"):
            new_meta["date"] = value["date"]
        out.append(
            Chunk(
                chunk_id=c.chunk_id,
                kb_id=c.kb_id,
                source_id=c.source_id,
                source_path=c.source_path,
                title=c.title,
                section_path=c.section_path,
                text=c.text,
                token_count=c.token_count,
                order=c.order,
                metadata=new_meta,
                parent_id=c.parent_id,
                chunk_role=c.chunk_role,
            )
        )
        done += 1
        if done % 20 == 0:
            log.info("metadata_enrich_progress", done=done, total=total)
    return out
