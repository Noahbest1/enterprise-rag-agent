"""PH3 acceptance: preprocess + cache + layout + multi-image endpoint + ingest."""
from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ---------- helpers ----------

def _make_rgb_png(w=200, h=100, color=(255, 0, 0)) -> bytes:
    im = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _make_exif_rotated_jpeg() -> bytes:
    """Build a minimal JPEG with EXIF orientation = 6 (rotate 90° CW)."""
    im = Image.new("RGB", (200, 100), (0, 255, 0))
    buf = io.BytesIO()
    # PIL doesn't trivially inject EXIF orientation; use piexif? skip --
    # just assert the code path handles a normal image without crashing.
    im.save(buf, format="JPEG")
    return buf.getvalue()


def _make_tiny_image() -> bytes:
    return _make_rgb_png(50, 50)


def _make_huge_image() -> bytes:
    return _make_rgb_png(6000, 4000)


# ---------- preprocess ----------

def test_preprocess_normalises_to_jpeg():
    from rag.vision.preprocess import preprocess_image
    out = preprocess_image(_make_rgb_png())
    assert out.format == "JPEG"
    assert out.bytes_[:3] == b"\xff\xd8\xff"
    assert out.width == 1000 or 200 <= out.width <= 4096  # upscaled


def test_preprocess_upscales_tiny_image():
    from rag.vision.preprocess import preprocess_image
    out = preprocess_image(_make_tiny_image())
    assert max(out.width, out.height) >= 1000
    assert any("upscaled" in a for a in out.applied)


def test_preprocess_downscales_huge_image():
    from rag.vision.preprocess import preprocess_image
    out = preprocess_image(_make_huge_image())
    assert max(out.width, out.height) <= 4096
    assert any("downscaled" in a for a in out.applied)


def test_preprocess_empty_raises():
    from rag.vision.preprocess import ImagePreprocessError, preprocess_image
    with pytest.raises(ImagePreprocessError):
        preprocess_image(b"")


def test_preprocess_bogus_bytes_raises():
    from rag.vision.preprocess import ImagePreprocessError, preprocess_image
    with pytest.raises(ImagePreprocessError):
        preprocess_image(b"this is not an image at all, just text bytes here")


def test_pixel_hash_deterministic():
    from rag.vision.preprocess import preprocess_image
    img = _make_rgb_png()
    h1 = preprocess_image(img).pixel_hash
    h2 = preprocess_image(img).pixel_hash
    assert h1 == h2
    # Different image → different hash
    h3 = preprocess_image(_make_rgb_png(color=(0, 0, 255))).pixel_hash
    assert h3 != h1


# ---------- cache ----------

def test_cache_hit_skips_runner():
    from rag import cache as rag_cache
    from rag.vision.cache import cached_vision_call

    rag_cache.reset_for_tests()

    calls = {"n": 0}

    def runner():
        calls["n"] += 1
        return {"value": "fresh"}

    r1 = cached_vision_call("abc123" + "0" * 58, "describe", runner)
    r2 = cached_vision_call("abc123" + "0" * 58, "describe", runner)
    assert r1 == r2 == {"value": "fresh"}
    assert calls["n"] == 1


def test_cache_different_task_keys_isolated():
    from rag import cache as rag_cache
    from rag.vision.cache import cached_vision_call

    rag_cache.reset_for_tests()
    h = "abc" + "0" * 61
    r1 = cached_vision_call(h, "describe", lambda: {"k": "d"})
    r2 = cached_vision_call(h, "ocr", lambda: {"k": "o"})
    assert r1 != r2
    assert r1["k"] == "d"
    assert r2["k"] == "o"


# ---------- layout ----------

def test_layout_parse_regions_valid_json():
    from rag.vision.layout import _parse_regions
    reply = '{"regions": [{"type":"title","text":"Hello"},{"type":"table","markdown":"| a |\\n|---|\\n| 1 |"}]}'
    regions = _parse_regions(reply)
    assert len(regions) == 2
    assert regions[0].type == "title"
    assert regions[1].type == "table"
    assert regions[1].markdown


def test_layout_parse_falls_back_on_non_json():
    from rag.vision.layout import _parse_regions
    regions = _parse_regions("sorry, I can't analyse this image")
    assert len(regions) == 1
    assert regions[0].type == "text"


def test_layout_parse_invalid_type_becomes_unknown():
    from rag.vision.layout import _parse_regions
    reply = '{"regions": [{"type": "banana", "text": "hi"}]}'
    regions = _parse_regions(reply)
    assert regions[0].type == "unknown"


# ---------- pipeline (integration, with VLM mocked) ----------

def test_image_to_chunks_emits_typed_chunks(monkeypatch):
    from rag.vision import pipeline as pvis
    from rag.vision import layout as plyt

    # Mock analyse_layout to return structured regions
    def fake_layout(bytes_):
        return [
            plyt.Region(type="title", text="Q3 Report"),
            plyt.Region(type="text", text="Revenue grew by 15%."),
            plyt.Region(type="table", markdown="| Region | Rev |\n|---|---|\n| NA | 10 |"),
            plyt.Region(type="figure", text="Bar chart showing regional breakdown."),
        ]
    monkeypatch.setattr(pvis, "analyse_layout", fake_layout)

    # reset cache so this test doesn't pick up a cached result
    from rag import cache as rag_cache
    rag_cache.reset_for_tests()

    chunks = pvis.image_to_chunks(
        _make_rgb_png(), kb_id="kb_img", source_id="kb_img::abc", source_uri="unit.png", title="Test Image",
    )

    # Expect 1 parent + 4 leaves
    parents = [c for c in chunks if c.chunk_role == "parent"]
    leaves = [c for c in chunks if c.chunk_role == "leaf"]
    assert len(parents) == 1
    assert len(leaves) == 4

    # Types preserved in metadata
    types = {c.metadata["region_type"] for c in leaves}
    assert types == {"title", "text", "table", "figure"}

    # Figure chunk has [Figure] prefix so it's retrievable via BM25
    fig_chunk = next(c for c in leaves if c.metadata["region_type"] == "figure")
    assert "Figure" in fig_chunk.text

    # Table is atomic: its markdown is in one chunk
    tbl = next(c for c in leaves if c.metadata["region_type"] == "table")
    assert "| Region | Rev |" in tbl.text


# ---------- endpoints ----------

@pytest.fixture()
def client(db_session):
    from rag_api.main import app
    return TestClient(app)


def test_endpoint_describe_single(client, monkeypatch):
    from rag.vision import pipeline as pvis
    from rag import cache as rag_cache
    rag_cache.reset_for_tests()
    monkeypatch.setattr(pvis, "describe_image", lambda b, q="", **kw: "mocked description")

    r = client.post(
        "/vision/describe",
        files={"image": ("a.png", io.BytesIO(_make_rgb_png()), "image/png")},
        data={"question": "what is this"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 1
    assert body["items"][0]["description"] == "mocked description"
    assert body["items"][0]["pixel_hash"]
    assert body["items"][0]["filename"] == "a.png"


def test_endpoint_describe_multi_image(client, monkeypatch):
    from rag.vision import pipeline as pvis
    from rag import cache as rag_cache
    rag_cache.reset_for_tests()
    monkeypatch.setattr(pvis, "describe_image", lambda b, q="", **kw: f"desc-{hash(b) & 0xff}")

    files = [
        ("images", ("a.png", io.BytesIO(_make_rgb_png(color=(255, 0, 0))), "image/png")),
        ("images", ("b.png", io.BytesIO(_make_rgb_png(color=(0, 255, 0))), "image/png")),
        ("images", ("c.png", io.BytesIO(_make_rgb_png(color=(0, 0, 255))), "image/png")),
    ]
    r = client.post("/vision/describe", files=files, data={"question": "compare"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 3
    assert {it["filename"] for it in body["items"]} == {"a.png", "b.png", "c.png"}


def test_endpoint_describe_rejects_empty(client):
    r = client.post(
        "/vision/describe",
        files={"image": ("empty.png", io.BytesIO(b""), "image/png")},
    )
    assert r.status_code == 400


def test_endpoint_describe_rejects_no_image(client):
    r = client.post("/vision/describe", data={"question": "hi"})
    assert r.status_code == 400


def test_endpoint_layout_happy_path(client, monkeypatch):
    from rag.vision import pipeline as pvis, layout as plyt
    from rag import cache as rag_cache
    rag_cache.reset_for_tests()
    monkeypatch.setattr(
        pvis, "analyse_layout",
        lambda b: [plyt.Region(type="text", text="stubbed layout region")],
    )

    r = client.post(
        "/vision/layout",
        files={"image": ("a.png", io.BytesIO(_make_rgb_png()), "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["regions"][0]["text"] == "stubbed layout region"
    assert body["filename"] == "a.png"


def test_endpoint_ocr_happy_path(client, monkeypatch):
    from rag.vision import pipeline as pvis
    from rag import cache as rag_cache
    rag_cache.reset_for_tests()
    monkeypatch.setattr(pvis, "extract_text_from_image", lambda b, **kw: "MOCK OCR RESULT")

    r = client.post(
        "/vision/ocr",
        files={"image": ("a.png", io.BytesIO(_make_rgb_png()), "image/png")},
    )
    assert r.status_code == 200
    assert r.json()["ocr_text"] == "MOCK OCR RESULT"


def test_endpoint_ask_multi_image_composes_query(client, monkeypatch):
    """Multi-image /vision/ask: all descriptions make it into the composed query."""
    from rag.vision import pipeline as pvis
    from rag import cache as rag_cache
    import rag_api.vision_routes as vroutes

    rag_cache.reset_for_tests()
    monkeypatch.setattr(pvis, "describe_image", lambda b, q="", **kw: f"pic-{hash(b) & 0xff:x}")

    from rag.types import Answer
    captured = {}

    async def fake_answer(query, kb_id, **kw):
        captured["query"] = query
        return Answer(
            query=query, rewritten_query=query, text="OK.",
            citations=[], abstained=False, reason="", hits=[], latency_ms=5, trace={},
        )
    monkeypatch.setattr(vroutes, "answer_query_async", fake_answer)

    files = [
        ("images", ("a.png", io.BytesIO(_make_rgb_png(color=(10, 0, 0))), "image/png")),
        ("images", ("b.png", io.BytesIO(_make_rgb_png(color=(0, 10, 0))), "image/png")),
    ]
    r = client.post(
        "/vision/ask",
        files=files,
        data={"question": "compare them", "kb_id": "jd_demo"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image_count"] == 2
    assert len(body["image_descriptions"]) == 2
    # The composed query must mention both images' descriptions
    q = captured["query"]
    assert "图片 1" in q and "图片 2" in q


def test_endpoint_rejects_too_many_images(client):
    files = [
        ("images", (f"{i}.png", io.BytesIO(_make_rgb_png()), "image/png"))
        for i in range(6)
    ]
    r = client.post("/vision/describe", files=files, data={"question": ""})
    assert r.status_code == 413
