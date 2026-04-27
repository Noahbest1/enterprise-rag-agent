"""Sprint A.5 metadata enrichment + filter wiring tests.

Covers:
- _parse_json_loose: handles plain JSON, fenced JSON, JSON with prose around
- _normalise: clamps entities (max 6, len 40), topics (max 3, underscore form),
  rejects bad dates
- enrich_chunk_metadata: returns safe defaults on LLMError
- enrich_chunks: writes / reuses disk cache
- _hit_matches_filter: entities_contains / topics_contains / date_after /
  date_before all AND-combined with existing filters
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from rag.ingest.metadata_enrich import (
    _cache_path,
    _normalise,
    _parse_json_loose,
    enrich_chunk_metadata,
    enrich_chunks,
)
from rag.retrieval.hybrid import _hit_matches_filter
from rag.types import Chunk, Hit


# ---------- parse / normalise ----------

def test_parse_json_loose_plain():
    assert _parse_json_loose('{"a": 1}') == {"a": 1}


def test_parse_json_loose_code_fence():
    s = "```json\n{\"entities\": [\"X\"]}\n```"
    assert _parse_json_loose(s) == {"entities": ["X"]}


def test_parse_json_loose_with_prose():
    s = "Sure, here is the JSON: {\"entities\": [\"Y\"]} thanks!"
    assert _parse_json_loose(s) == {"entities": ["Y"]}


def test_parse_json_loose_invalid():
    assert _parse_json_loose("definitely not JSON") is None


def test_normalise_happy_path():
    out = _normalise({
        "entities": ["Airbyte", "v1.5", "Postgres"],
        "topics": ["Release Notes", "ETL Pipeline"],
        "date": "2025-02-20",
    })
    assert out["entities"] == ["Airbyte", "v1.5", "Postgres"]
    assert out["topics"] == ["release_notes", "etl_pipeline"]
    assert out["date"] == "2025-02-20"


def test_normalise_clamps_entities():
    out = _normalise({"entities": ["A"] * 20})  # dedupes case-insensitive
    assert len(out["entities"]) == 1


def test_normalise_drops_long_entities():
    out = _normalise({"entities": ["X" * 60, "ok"]})
    assert "ok" in out["entities"]
    assert all(len(e) <= 40 for e in out["entities"])


def test_normalise_bad_date():
    out = _normalise({"date": "not-a-date"})
    assert out["date"] is None
    out2 = _normalise({"date": "2025/03/01"})
    assert out2["date"] is None


def test_normalise_missing_fields():
    out = _normalise({})
    assert out == {"entities": [], "topics": [], "date": None}


def test_normalise_non_dict():
    assert _normalise(None) == {"entities": [], "topics": [], "date": None}


# ---------- enrichment with mocked LLM ----------

def _chunk(i: int = 0, text: str = "京东 PLUS 年卡 148 元,2025-02-20 起生效。") -> Chunk:
    return Chunk(
        chunk_id=f"c{i}", kb_id="kb", source_id="s1", source_path="/p",
        title="title", section_path=[], text=text, token_count=10, order=i,
        chunk_role="leaf",
    )


def test_enrich_chunk_metadata_llm_error_returns_defaults():
    from rag.answer.llm_client import LLMError
    with patch("rag.ingest.metadata_enrich.chat_once", side_effect=LLMError("no key")):
        out = enrich_chunk_metadata(_chunk())
    assert out == {"entities": [], "topics": [], "date": None}


def test_enrich_chunk_metadata_happy(monkeypatch):
    with patch("rag.ingest.metadata_enrich.chat_once",
               return_value='{"entities": ["PLUS", "京东"], "topics": ["membership"], "date": "2025-02-20"}'):
        out = enrich_chunk_metadata(_chunk())
    assert out["entities"] == ["PLUS", "京东"]
    assert out["topics"] == ["membership"]
    assert out["date"] == "2025-02-20"


def test_enrich_chunks_writes_cache(tmp_path: Path):
    chunks = [_chunk(0), _chunk(1, text="不同内容 different content")]
    with patch("rag.ingest.metadata_enrich.chat_once",
               return_value='{"entities": ["X"], "topics": ["t1"], "date": null}'):
        out = enrich_chunks(chunks, tmp_path)
    assert all(c.metadata.get("entities") == ["X"] for c in out)
    assert _cache_path(tmp_path).exists()


def test_enrich_chunks_reuses_cache(tmp_path: Path):
    chunks = [_chunk(0)]
    # First pass: write cache
    with patch("rag.ingest.metadata_enrich.chat_once",
               return_value='{"entities": ["FirstCall"], "topics": ["t"], "date": null}') as m:
        enrich_chunks(chunks, tmp_path)
        assert m.call_count == 1
    # Second pass: cache hit -> NO new LLM call
    with patch("rag.ingest.metadata_enrich.chat_once",
               return_value='{"entities": ["ShouldNotBeSeen"], "topics": [], "date": null}') as m2:
        out = enrich_chunks(chunks, tmp_path)
        assert m2.call_count == 0
    # Metadata came from cache, not the second mock
    assert out[0].metadata["entities"] == ["FirstCall"]


def test_enrich_chunks_skips_parents(tmp_path: Path):
    parent = Chunk(
        chunk_id="p", kb_id="kb", source_id="s", source_path="/p", title="t",
        section_path=[], text="parent text", token_count=5, order=0,
        chunk_role="parent",
    )
    leaf = _chunk(1)
    with patch("rag.ingest.metadata_enrich.chat_once",
               return_value='{"entities": ["E"], "topics": [], "date": null}') as m:
        out = enrich_chunks([parent, leaf], tmp_path)
    assert m.call_count == 1  # only the leaf got a call
    # parent metadata left untouched; leaf got enriched
    assert out[0].metadata == {} or out[0].metadata.get("entities") is None
    assert out[1].metadata["entities"] == ["E"]


# ---------- filter ----------

def _hit(text: str = "", metadata: dict | None = None) -> Hit:
    return Hit(
        chunk_id="c1", score=1.0, text=text, title="t", source_id="s1",
        source_path="/p", section_path=[], retrieval_source="test",
        metadata=metadata or {},
    )


def test_entities_contains_matches_any():
    h = _hit(metadata={"entities": ["iPhone 16 Pro", "A18 Pro"]})
    assert _hit_matches_filter(h, {"entities_contains": ["iphone"]})
    assert _hit_matches_filter(h, {"entities_contains": ["a18 pro", "samsung"]})


def test_entities_contains_no_match():
    h = _hit(metadata={"entities": ["Airbyte"]})
    assert not _hit_matches_filter(h, {"entities_contains": ["nothing"]})


def test_entities_contains_missing_metadata():
    h = _hit(metadata={})
    assert not _hit_matches_filter(h, {"entities_contains": ["X"]})


def test_topics_contains():
    h = _hit(metadata={"topics": ["return_policy", "shipping"]})
    assert _hit_matches_filter(h, {"topics_contains": ["return_policy"]})
    assert _hit_matches_filter(h, {"topics_contains": ["ship"]})  # substring
    assert not _hit_matches_filter(h, {"topics_contains": ["membership"]})


def test_date_after():
    h = _hit(metadata={"date": "2025-06-01"})
    assert _hit_matches_filter(h, {"date_after": "2025-01-01"})
    assert not _hit_matches_filter(h, {"date_after": "2026-01-01"})


def test_date_before():
    h = _hit(metadata={"date": "2025-06-01"})
    assert _hit_matches_filter(h, {"date_before": "2026-01-01"})
    assert not _hit_matches_filter(h, {"date_before": "2025-01-01"})


def test_date_range():
    h = _hit(metadata={"date": "2025-06-01"})
    assert _hit_matches_filter(h, {"date_after": "2025-01-01", "date_before": "2025-12-31"})


def test_date_missing_fails_date_filter():
    h = _hit(metadata={})  # no date
    assert not _hit_matches_filter(h, {"date_after": "2025-01-01"})


def test_filters_and_combined():
    h = _hit(metadata={"entities": ["iPhone"], "topics": ["products"], "date": "2025-06-01"})
    # All pass -> match
    assert _hit_matches_filter(h, {
        "entities_contains": ["iphone"],
        "topics_contains": ["products"],
        "date_after": "2025-01-01",
    })
    # One fails -> overall fail
    assert not _hit_matches_filter(h, {
        "entities_contains": ["iphone"],
        "topics_contains": ["nomatch"],
    })
