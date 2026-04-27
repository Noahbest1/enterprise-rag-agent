"""PH2 acceptance: boilerplate / low-info / PII / dedup + end-to-end ingest."""
from __future__ import annotations

from pathlib import Path

import pytest

from rag.ingest.clean import (
    MinHashDeduper,
    is_low_info_chunk,
    low_info_score,
    redact_pii,
    strip_html_boilerplate,
)


# --- boilerplate ---

def test_html_cleaner_drops_nav_and_scripts():
    html = """
    <html><head><title>Real Article</title></head>
    <body>
      <nav>Home | About | Contact | Sign in | Sign up</nav>
      <header><ul><li>Menu</li></ul></header>
      <main><article>
        <h1>Real Article</h1>
        <p>This is the actual content paragraph one.</p>
        <p>This is paragraph two with more substance to it.</p>
      </article></main>
      <script>window.trackingCode=1;</script>
      <style>.x{color:red}</style>
      <footer>© 2026 Nobody. All rights reserved.</footer>
    </body></html>
    """
    title, text = strip_html_boilerplate(html)
    assert "actual content" in text
    assert "tracking" not in text.lower()
    assert ".x{color:red" not in text


def test_html_cleaner_passthrough_non_html():
    title, text = strip_html_boilerplate("just plain text, nothing fancy")
    assert text == "just plain text, nothing fancy"


def test_html_cleaner_empty_input():
    title, text = strip_html_boilerplate("")
    assert text == ""


# --- low-info ---

def test_low_info_short_text_is_low_info():
    assert is_low_info_chunk("hi") is True


def test_low_info_boilerplate_phrases():
    assert is_low_info_chunk("Click here") is True
    assert is_low_info_chunk("下一页") is True
    assert is_low_info_chunk("Table of Contents") is True
    assert is_low_info_chunk("© 2026 All Rights Reserved") is True


def test_low_info_repetitive_text():
    text = "ok " * 40
    assert low_info_score(text) > 0.7


def test_low_info_real_content_passes():
    text = (
        "The quick brown fox jumps over the lazy dog. This sentence contains "
        "multiple distinct words and conveys real meaning about animals and motion."
    )
    assert is_low_info_chunk(text) is False


# --- PII ---

def test_pii_redacts_cn_mobile():
    text = "我的手机号是 13800138000,请联系我"
    safe, r = redact_pii(text)
    assert "13800138000" not in safe
    assert "[PHONE]" in safe
    assert r.phone == 1
    assert r.total == 1


def test_pii_redacts_cn_id():
    text = "身份证 11010519900307001X 仅供参考"
    safe, r = redact_pii(text)
    assert "11010519900307001X" not in safe
    assert "[ID]" in safe
    assert r.id_card == 1


def test_pii_redacts_email_and_ip():
    text = "联系 alice@example.com 或访问 192.168.1.100"
    safe, r = redact_pii(text)
    assert "alice@example.com" not in safe
    assert "192.168.1.100" not in safe
    assert r.email == 1
    assert r.ip == 1


def test_pii_redacts_valid_bankcard_only():
    # 4111111111111111 is the Visa test Luhn-valid dummy
    valid_card = "4111111111111111"
    text = f"卡号 {valid_card} 交易成功。订单号 99999999999999 不是卡号。"
    safe, r = redact_pii(text)
    assert valid_card not in safe
    assert "[BANKCARD]" in safe
    # The Luhn-invalid 14-digit "订单号" should NOT be redacted
    assert "99999999999999" in safe
    assert r.bank_card == 1


def test_pii_no_false_positive_on_clean_text():
    text = "PLUS 年卡 148 元, 每月 5 张免邮券"
    safe, r = redact_pii(text)
    assert safe == text
    assert r.total == 0


# --- Dedup ---

def test_dedup_flags_exact_duplicate():
    d = MinHashDeduper(threshold=0.85)
    a = "This is a substantial paragraph with enough words to shingle usefully."
    d.add(a)
    assert d.is_dup(a) is True


def test_dedup_allows_different_content():
    d = MinHashDeduper(threshold=0.85)
    d.add("The iPhone 16 Pro supports USB-C and has a titanium frame.")
    assert d.is_dup("MacBook Pro ships with an M4 Pro chip and 24GB RAM.") is False


def test_dedup_flags_near_dup():
    d = MinHashDeduper(threshold=0.6)  # looser threshold for the test
    a = "The iPhone 16 Pro supports USB-C and has a titanium frame and new camera."
    b = "The iPhone 16 Pro supports USB-C and has a titanium frame and new cameras."
    d.add(a)
    assert d.is_dup(b) is True


def test_dedup_empty_treated_as_dup():
    d = MinHashDeduper()
    assert d.is_dup("") is True


# --- end-to-end: dirty doc + all cleaners on ---

def _run_sync(kb_id, source_dir, kb_dir, *, clean_cfg=None):
    from rag.ingest.connectors import LocalDirConnector
    from rag.ingest.incremental import sync_kb
    kb_dir.mkdir(parents=True, exist_ok=True)
    return sync_kb(kb_id, LocalDirConnector(kb_id, source_dir), kb_dir, clean_cfg=clean_cfg)


def test_e2e_pii_redacted_before_index(seeded_db, tmp_path, monkeypatch):
    """End-to-end: PII in source doc must NOT appear in BM25 or vector index."""
    monkeypatch.setenv("VECTOR_BACKEND", "faiss")
    src = tmp_path / "src"
    src.mkdir()
    (src / "about.md").write_text(
        "# About\nContact 13800138000 anytime. Also email admin@corp.com.\n"
        "Our server IP is 10.0.0.5 for debugging.",
        encoding="utf-8",
    )
    kb_dir = tmp_path / "kb"

    from rag.ingest.incremental import CleanConfig
    stats = _run_sync("ph2_pii", src, kb_dir, clean_cfg=CleanConfig(pii_redact=True))
    assert stats.pii_redactions >= 2

    # grep chunks.jsonl-substitute (vector_meta.jsonl has all leaf text)
    meta = (kb_dir / "vector_meta.jsonl").read_text(encoding="utf-8")
    assert "13800138000" not in meta
    assert "admin@corp.com" not in meta
    assert "10.0.0.5" not in meta
    assert "[PHONE]" in meta or "[EMAIL]" in meta  # replacement visible


def test_e2e_low_info_dropped(seeded_db, tmp_path, monkeypatch):
    """Dirty sections (navigation / copyright) under their own headings must be
    dropped; real content under its heading survives."""
    monkeypatch.setenv("VECTOR_BACKEND", "faiss")
    src = tmp_path / "src"
    src.mkdir()
    # Each heading becomes its own section, which becomes at least one chunk.
    # Low-info sections get filtered; the real-content section stays.
    (src / "page.md").write_text(
        "# Navigation\n\nClick here\n\n"
        "# Footer\n\n© 2026 All Rights Reserved\n\n"
        "# Next page\n\n下一页\n\n"
        "# Real content\n\n"
        "Our flagship product runs on the A18 Pro chip, features a titanium frame, "
        "and ships with 256 GB of storage. Battery life is rated at 27 hours of video playback.",
        encoding="utf-8",
    )
    kb_dir = tmp_path / "kb"
    from rag.ingest.incremental import CleanConfig
    stats = _run_sync("ph2_lowinfo", src, kb_dir, clean_cfg=CleanConfig(low_info_filter=True))
    assert stats.low_info_dropped >= 1, stats.summary()

    # Real content should still be indexed
    meta = (kb_dir / "vector_meta.jsonl").read_text(encoding="utf-8")
    assert "A18 Pro" in meta


def test_e2e_dedup_dropped(seeded_db, tmp_path, monkeypatch):
    """If two files have the same content, dedup should drop the second's chunks within one sync."""
    monkeypatch.setenv("VECTOR_BACKEND", "faiss")
    src = tmp_path / "src"
    src.mkdir()
    content = (
        "# Refund policy\n\n"
        "Unused items can be returned within 7 days of delivery. Please keep "
        "the original packaging, all accessories, and the receipt together."
    )
    (src / "a.md").write_text(content, encoding="utf-8")
    (src / "b.md").write_text(content, encoding="utf-8")  # identical duplicate
    kb_dir = tmp_path / "kb"
    from rag.ingest.incremental import CleanConfig
    stats = _run_sync("ph2_dedup", src, kb_dir, clean_cfg=CleanConfig(dedup=True))
    assert stats.dedup_dropped >= 1, stats.summary()
