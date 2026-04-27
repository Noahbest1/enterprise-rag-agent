"""POST /kbs/{kb_id}/upload — multipart upload + build endpoint.

Covers:
- Happy path: text file uploads, KB is auto-created, indexes are built,
  chunk_count is reported.
- Multiple files in one request.
- Unsupported extension is skipped (not 400'd) but reported in `skipped`.
- Empty file is skipped.
- Files exceeding 8 MB cap are skipped with a clear reason.
- Path-traversal filenames are sanitized to basename.
- create_if_missing=false on a missing KB returns 404 (no auto-create).
- All-skipped request returns 400 with the skipped list in detail.

This is the ONLY browser-reachable path to add a new KB after we deleted
App.tsx; it's the upload demo's load-bearing API.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag.db.base import Base, engine
from rag_api.main import app


@pytest.fixture(autouse=True)
def _fresh_db():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Redirect KB root to tmp_path so tests don't pollute data/kb/.
    from rag.config import settings
    monkeypatch.setattr(settings, "kb_root", tmp_path)
    return TestClient(app)


def _txt(name: str, body: str) -> tuple[str, io.BytesIO, str]:
    return (name, io.BytesIO(body.encode("utf-8")), "text/plain")


def test_upload_creates_kb_and_indexes(client):
    files = [
        ("files", _txt("a.md", "# Title A\n\nThis is paragraph one about apples.\n\nParagraph two.")),
        ("files", _txt("b.md", "# Title B\n\nBananas are yellow. Carrots are orange.")),
    ]
    r = client.post(
        "/kbs/upload_test_kb/upload",
        files=files,
        data={"create_if_missing": "true", "description": "unit test KB"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kb_id"] == "upload_test_kb"
    assert body["chunk_count"] >= 2
    assert body["indexed_chunks"] >= 1
    assert {s["filename"] for s in body["saved"]} == {"a.md", "b.md"}
    assert body["skipped"] == []


def test_upload_skips_unsupported_extensions(client):
    files = [
        ("files", _txt("doc.md", "# Heading\n\nHello.")),
        ("files", ("evil.exe", io.BytesIO(b"binary"), "application/octet-stream")),
        ("files", ("notes.foo", io.BytesIO(b"weird"), "application/octet-stream")),
    ]
    r = client.post("/kbs/skip_test/upload", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    saved_names = {s["filename"] for s in body["saved"]}
    skipped_names = {s["filename"] for s in body["skipped"]}
    assert saved_names == {"doc.md"}
    assert skipped_names == {"evil.exe", "notes.foo"}
    for s in body["skipped"]:
        assert "unsupported extension" in s["reason"]


def test_upload_skips_empty_files(client):
    files = [
        ("files", _txt("good.md", "# OK\n\nReal content here.")),
        ("files", ("empty.md", io.BytesIO(b""), "text/plain")),
    ]
    r = client.post("/kbs/empty_test/upload", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "good.md" in {s["filename"] for s in body["saved"]}
    assert any(s["filename"] == "empty.md" and "empty" in s["reason"] for s in body["skipped"])


def test_upload_rejects_oversize(client):
    big = b"x" * (9 * 1024 * 1024)  # > 8 MB
    files = [
        ("files", _txt("ok.md", "# OK\n\nReal stuff.")),
        ("files", ("huge.md", io.BytesIO(big), "text/plain")),
    ]
    r = client.post("/kbs/oversize_test/upload", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "ok.md" in {s["filename"] for s in body["saved"]}
    assert any(s["filename"] == "huge.md" and "too large" in s["reason"] for s in body["skipped"])


def test_upload_sanitizes_path_traversal(client):
    # Try to escape into parent dirs via filename
    files = [
        ("files", ("../../etc/sneaky.md", io.BytesIO(b"# title\n\nbody"), "text/plain")),
    ]
    r = client.post("/kbs/traversal_test/upload", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    # The raw filename was reduced to its basename "sneaky.md"
    assert {s["filename"] for s in body["saved"]} == {"sneaky.md"}


def test_upload_404_when_create_disabled_and_kb_missing(client):
    files = [("files", _txt("hi.md", "# x\n\nbody"))]
    r = client.post(
        "/kbs/never_created/upload",
        files=files,
        data={"create_if_missing": "false"},
    )
    assert r.status_code == 404
    assert "never_created" in r.json()["detail"]


def test_upload_400_when_all_files_skipped(client):
    files = [
        ("files", ("a.exe", io.BytesIO(b"x"), "application/octet-stream")),
        ("files", ("b.dll", io.BytesIO(b"y"), "application/octet-stream")),
    ]
    r = client.post("/kbs/all_skipped/upload", files=files)
    assert r.status_code == 400
    detail = r.json()["detail"]
    # Detail is a structured dict here (we returned {"message": ..., "skipped": [...]})
    assert isinstance(detail, dict)
    assert detail["message"] == "no files saved"
    assert len(detail["skipped"]) == 2


def test_upload_uses_existing_kb_without_recreating(client, tmp_path):
    # First upload creates the KB
    files1 = [("files", _txt("v1.md", "# version one\n\nfirst body"))]
    r1 = client.post("/kbs/reupload_test/upload", files=files1)
    assert r1.status_code == 200

    # Second upload with create_if_missing=false should still succeed
    files2 = [("files", _txt("v2.md", "# version two\n\nsecond body"))]
    r2 = client.post(
        "/kbs/reupload_test/upload",
        files=files2,
        data={"create_if_missing": "false"},
    )
    assert r2.status_code == 200, r2.text
    body = r2.json()
    saved_names = {s["filename"] for s in body["saved"]}
    # Both files now sit on disk
    raw_dir = tmp_path / "reupload_test" / "raw"
    on_disk = {p.name for p in raw_dir.iterdir() if p.is_file()}
    assert "v1.md" in on_disk
    assert "v2.md" in on_disk
    assert saved_names == {"v2.md"}
