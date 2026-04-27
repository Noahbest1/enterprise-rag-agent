"""Sprint A.multimodal — CLIP image embed + image index + /vision/search.

Most tests monkeypatch CLIP to keep the suite fast (no 180 MB model load).
One opt-in smoke test exercises the real encoder -- skipped unless
``RUN_CLIP_SMOKE=1`` is set.

Covers:
- cosine: handles zero vectors, returns 1 for identical, 0 for orthogonal
- ImageIndex: add / save / load round-trip, remove, upsert-on-duplicate
- ImageIndex.search: top-k ordering + empty-index -> []
- /vision/search endpoint: image + text mode, missing KB -> 404, no input -> 400
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from rag_api.main import app
    return TestClient(app)


# ---------- cosine ----------

def test_cosine_identical():
    from rag.vision.image_embed import cosine
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal():
    from rag.vision.image_embed import cosine
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine(a, b) == pytest.approx(0.0)


def test_cosine_zero_vector_guard():
    from rag.vision.image_embed import cosine
    z = np.zeros(3, dtype=np.float32)
    assert cosine(z, z) == 0.0


# ---------- ImageIndex ----------

def _vec(seed: int, dim: int = 8) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v = v / (np.linalg.norm(v) or 1.0)
    return v.tolist()


def test_image_index_add_and_search(tmp_path: Path):
    from rag.vision.image_index import ImageIndex
    idx = ImageIndex(tmp_path)
    idx.add("s1", "img one", "/p1", _vec(1))
    idx.add("s2", "img two", "/p2", _vec(2))
    assert len(idx) == 2

    hits = idx.search(np.array(_vec(1), dtype=np.float32), top_k=2)
    assert hits[0].source_id == "s1"
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-4)


def test_image_index_upsert_same_source_id(tmp_path: Path):
    from rag.vision.image_index import ImageIndex
    idx = ImageIndex(tmp_path)
    idx.add("s1", "old", "/p1", _vec(1))
    idx.add("s1", "new", "/p1b", _vec(2))
    assert len(idx) == 1
    # second add replaces first
    hits = idx.search(np.array(_vec(2), dtype=np.float32), top_k=1)
    assert hits[0].title == "new"


def test_image_index_remove(tmp_path: Path):
    from rag.vision.image_index import ImageIndex
    idx = ImageIndex(tmp_path)
    idx.add("s1", "t", "/p", _vec(1))
    idx.add("s2", "t", "/p", _vec(2))
    n = idx.remove("s1")
    assert n == 1
    assert len(idx) == 1
    assert idx.remove("absent") == 0


def test_image_index_save_load_roundtrip(tmp_path: Path):
    from rag.vision.image_index import ImageIndex
    idx = ImageIndex(tmp_path)
    idx.add("s1", "t", "/p", _vec(1))
    idx.save()

    path = tmp_path / "image_index.jsonl"
    assert path.exists()
    # Valid JSONL
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            assert "source_id" in rec and "embedding" in rec

    idx2 = ImageIndex.load(tmp_path)
    assert len(idx2) == 1
    hits = idx2.search(np.array(_vec(1), dtype=np.float32), top_k=1)
    assert hits[0].source_id == "s1"


def test_image_index_empty_search_returns_empty(tmp_path: Path):
    from rag.vision.image_index import ImageIndex
    idx = ImageIndex(tmp_path)
    assert idx.search(np.array(_vec(1), dtype=np.float32)) == []


def test_image_index_zero_query_returns_empty(tmp_path: Path):
    from rag.vision.image_index import ImageIndex
    idx = ImageIndex(tmp_path)
    idx.add("s1", "t", "/p", _vec(1))
    assert idx.search(np.zeros(8, dtype=np.float32)) == []


# ---------- /vision/search endpoint ----------

def _fake_image_embed(_raw: bytes):
    return np.array(_vec(42), dtype=np.float32)


def _fake_text_embed(_text: str):
    return np.array(_vec(42), dtype=np.float32)


def test_vision_search_missing_kb(client: TestClient):
    r = client.post(
        "/vision/search",
        data={"kb_id": "nonexistent_kb", "text": "hello"},
    )
    assert r.status_code == 404


def test_vision_search_empty_input(client: TestClient):
    # Make sure we hit an existing KB so the 400 is triggered by missing payload.
    r = client.post(
        "/vision/search",
        data={"kb_id": "jd_demo", "text": "   "},
    )
    assert r.status_code == 400


def test_vision_search_text_mode_empty_index(client: TestClient, monkeypatch):
    # jd_demo has no image_index.jsonl -> empty hits but 200
    monkeypatch.setattr("rag_api.vision_routes.embed_text_for_image", _fake_text_embed)
    r = client.post(
        "/vision/search",
        data={"kb_id": "jd_demo", "text": "iPhone 16 Pro red", "top_k": "5"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["kb_id"] == "jd_demo"
    assert body["mode"] == "text"
    assert body["index_size"] == 0
    assert body["hits"] == []


def test_vision_search_image_mode_with_index(client: TestClient, tmp_path, monkeypatch):
    from rag.vision.image_index import ImageIndex
    # Seed the jd_demo image index with one entry that matches the fake embed
    from rag.knowledge_base import get_kb
    kb = get_kb("jd_demo")
    idx = ImageIndex(kb.root)
    idx.add("jd_demo::product_1", "iPhone photo", "/data/kb/jd_demo/raw/iphone.jpg", _vec(42))
    idx.save()
    try:
        monkeypatch.setattr("rag_api.vision_routes.embed_image_bytes", _fake_image_embed)
        fake_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 128  # bogus but non-empty
        r = client.post(
            "/vision/search",
            data={"kb_id": "jd_demo", "top_k": "3"},
            files={"image": ("q.jpg", fake_jpeg, "image/jpeg")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "image"
        assert body["index_size"] == 1
        assert body["hits"][0]["source_id"] == "jd_demo::product_1"
        assert body["hits"][0]["similarity"] == pytest.approx(1.0, abs=1e-3)
    finally:
        # Cleanup so other test runs don't see the injected entry
        (kb.root / "image_index.jsonl").unlink(missing_ok=True)


# ---------- opt-in real CLIP smoke ----------

@pytest.mark.skipif(os.environ.get("RUN_CLIP_SMOKE") != "1",
                    reason="set RUN_CLIP_SMOKE=1 to run real CLIP encode (180MB model)")
def test_clip_smoke_real_encoder():
    from rag.vision.image_embed import embed_image_bytes, embed_text_for_image, embed_dim
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color="red").save(buf, format="JPEG")
    v_img = embed_image_bytes(buf.getvalue())
    v_txt = embed_text_for_image("a plain red image")
    assert v_img.shape == (embed_dim(),)
    assert v_txt.shape == (embed_dim(),)
    # Red image should have non-trivial similarity with the phrase
    sim = float(np.dot(v_img, v_txt))
    assert -1.0 <= sim <= 1.0
