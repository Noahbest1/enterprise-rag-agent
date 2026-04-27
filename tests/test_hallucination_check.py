"""Hallucination guardrail tests.

Doesn't hit a real LLM — we patch ``chat_once`` to return canned verifier
JSON so the parsing / fail-open / flagged-text logic can be checked
deterministically. The integration test (real qwen-turbo) is opt-in via
DASHSCOPE_API_KEY in CI.
"""
from __future__ import annotations

import pytest

from rag.answer import hallucination_check as hc


def test_split_sentences_handles_cjk_and_ascii():
    text = "你好。这是第一句话。Here is the next one. And another."
    s = hc._split_sentences(text)
    assert len(s) >= 3


def test_split_sentences_empty():
    assert hc._split_sentences("") == []
    assert hc._split_sentences("   ") == []


def test_verify_all_supported(monkeypatch):
    """Verifier says everything is supported -> clean report."""
    canned = '[{"i":1,"supported":true,"reasoning":"直接引用"},{"i":2,"supported":true,"reasoning":"OK"}]'
    monkeypatch.setattr(hc, "chat_once", lambda **kwargs: canned)

    report = hc.verify_answer_grounding(
        "PLUS 会员年费 198 元[1]。运费立减 6 元[2]。",
        chunks=[{"title": "会员", "text": "PLUS 198 元"}, {"title": "运费", "text": "立减 6 元"}],
    )
    assert report.is_clean
    assert report.unsupported_count == 0
    assert all(v.supported for v in report.verified_sentences)


def test_verify_flags_unsupported(monkeypatch):
    canned = '[{"i":1,"supported":true,"reasoning":"OK"},{"i":2,"supported":false,"reasoning":"chunks 没说支持夜间客服"}]'
    monkeypatch.setattr(hc, "chat_once", lambda **kwargs: canned)

    report = hc.verify_answer_grounding(
        "PLUS 会员年费 198 元[1]。还提供 24 小时夜间专属客服[1]。",
        chunks=[{"title": "会员", "text": "PLUS 198 元/年"}],
    )
    assert not report.is_clean
    assert report.unsupported_count == 1
    # Unsupported sentence is prefixed with ⚠ in the flagged text.
    assert "⚠" in report.flagged_text


def test_verify_degrades_open_on_llm_error(monkeypatch):
    """Verifier failure should NOT block the answer — fall back to "all
    supported" + record verifier-unavailable in reasoning."""
    def boom(**kwargs):
        from rag.answer.llm_client import LLMError
        raise LLMError("network down")
    monkeypatch.setattr(hc, "chat_once", boom)

    report = hc.verify_answer_grounding(
        "Some answer.",
        chunks=[{"title": "x", "text": "y"}],
    )
    assert report.is_clean
    assert all(v.supported for v in report.verified_sentences)
    # Reasoning records the verifier was unavailable.
    assert any("unavailable" in v.reasoning for v in report.verified_sentences)


def test_verify_handles_garbage_llm_output(monkeypatch):
    """Verifier returns non-JSON -> degrade open + log."""
    monkeypatch.setattr(hc, "chat_once", lambda **kwargs: "I forgot the JSON, sorry.")

    report = hc.verify_answer_grounding(
        "Hello.",
        chunks=[{"title": "x", "text": "y"}],
    )
    assert report.is_clean


def test_verify_extracts_citation_numbers(monkeypatch):
    canned = '[{"i":1,"supported":true,"reasoning":"OK"}]'
    monkeypatch.setattr(hc, "chat_once", lambda **kwargs: canned)

    report = hc.verify_answer_grounding(
        "PLUS 会员年费 198 元[1] 享受优惠[2]。",
        chunks=[{"title": "x", "text": "y"}],
    )
    assert report.verified_sentences[0].cited_ns == [1, 2]


def test_is_enabled_default_off(monkeypatch):
    from rag.config import settings
    monkeypatch.setattr(settings, "enable_hallucination_check", False)
    assert hc.is_enabled() is False
    monkeypatch.setattr(settings, "enable_hallucination_check", True)
    assert hc.is_enabled() is True


def test_empty_answer_returns_empty_report():
    report = hc.verify_answer_grounding("", chunks=[])
    assert report.unsupported_count == 0
    assert report.verified_sentences == []
