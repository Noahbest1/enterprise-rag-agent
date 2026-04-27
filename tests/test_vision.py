"""Vision endpoints + VLM client. We mock the VLM call so tests don't
burn image-generation tokens and run deterministically.
"""
from __future__ import annotations

from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


TINY_JPEG = (
    # 1x1 black JPEG minimal bytes
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00" + b"\x08" * 64 +
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xc4\x00\x1f\x00" + b"\x00" * 30 +
    b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xd2\xff\xd9"
)


def test_detect_mime_variants():
    from rag.vision.vlm import _detect_mime
    assert _detect_mime(b"\xff\xd8\xff").startswith("image/jpeg")
    assert _detect_mime(b"\x89PNG\r\n\x1a\n") == "image/png"
    assert _detect_mime(b"GIF8xxx") == "image/gif"
    assert _detect_mime(b"RIFF\x00\x00\x00\x00WEBPxxx") == "image/webp"


def test_data_url_builds_correctly():
    from rag.vision.vlm import _data_url
    url = _data_url(TINY_JPEG)
    assert url.startswith("data:image/jpeg;base64,")


def test_describe_image_calls_vlm(monkeypatch):
    from rag.vision import vlm
    monkeypatch.setattr(vlm, "_call_vlm", lambda *a, **kw: "a small black square")
    monkeypatch.setattr(vlm.settings, "qwen_api_key", "fake", raising=False)
    out = vlm.describe_image(TINY_JPEG, question="what do you see?")
    assert out == "a small black square"


def test_vision_describe_endpoint(db_session, monkeypatch):
    """Endpoint now uses multi-image payload shape: {count, items:[{description,...}]}."""
    from rag_api.main import app
    client = TestClient(app)

    from rag.vision import pipeline as pvis
    from rag import cache as rag_cache
    rag_cache.reset_for_tests()
    monkeypatch.setattr(pvis, "describe_image", lambda b, q="", **kw: "mocked description")

    r = client.post(
        "/vision/describe",
        files={"image": ("test.jpg", BytesIO(TINY_JPEG), "image/jpeg")},
        data={"question": "is this broken?"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 1
    assert body["items"][0]["description"] == "mocked description"
    assert body["items"][0]["filename"] == "test.jpg"


def test_vision_describe_rejects_empty(db_session):
    from rag_api.main import app
    client = TestClient(app)
    r = client.post(
        "/vision/describe",
        files={"image": ("empty.jpg", BytesIO(b""), "image/jpeg")},
    )
    assert r.status_code == 400


def test_vision_ask_combines_vlm_and_rag(db_session, monkeypatch):
    """vision/ask calls VLM per image then feeds composed query through answer_query_async."""
    from rag_api.main import app
    from rag.vision import pipeline as pvis
    from rag import cache as rag_cache
    import rag_api.vision_routes as vroutes

    client = TestClient(app)

    rag_cache.reset_for_tests()
    monkeypatch.setattr(pvis, "describe_image", lambda b, q="", **kw: "screenshot of mysql connection error")

    from rag.types import Answer
    async def fake_answer(query, kb_id, **kw):
        assert "screenshot" in query.lower() or "mysql" in query.lower()
        return Answer(
            query=query, rewritten_query=query, text="Check host + port first.",
            citations=[], abstained=False, reason="", hits=[], latency_ms=10, trace={},
        )
    monkeypatch.setattr(vroutes, "answer_query_async", fake_answer)

    r = client.post(
        "/vision/ask",
        files={"image": ("err.jpg", BytesIO(TINY_JPEG), "image/jpeg")},
        data={"question": "what error is this?", "kb_id": "airbyte_demo"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image_count"] == 1
    assert body["image_descriptions"][0]["description"] == "screenshot of mysql connection error"
    assert body["answer"] == "Check host + port first."
