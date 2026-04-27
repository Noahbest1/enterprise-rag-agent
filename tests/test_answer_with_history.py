"""Option-B tests: answer prompt builder + conversation history.

The 'B' fix lets the answerer SEE the last 2 turns to disambiguate
follow-ups like "那 PRO 呢", but the system prompt's HISTORY_RULE
forbids using history as a fact source. These tests verify both halves:
the history block is present when expected, and the strict rule is
attached to the system prompt so the LLM knows not to trust it.
"""
from __future__ import annotations

from rag.answer.compose import (
    HISTORY_RULE_EN,
    HISTORY_RULE_ZH,
    _format_history_block,
    build_answer_prompt,
)
from rag.types import Hit


def _hit(n: int = 1) -> Hit:
    return Hit(
        chunk_id=f"c{n}",
        score=0.9,
        text=f"chunk {n} body content",
        title=f"doc {n}",
        source_id=f"s{n}",
        source_path=f"/p/{n}.md",
        section_path=["root", f"sec{n}"],
        retrieval_source="hybrid",
    )


# ---------- 1. Format helper ----------

def test_format_history_block_empty():
    assert _format_history_block(None, max_turns=2, max_chars_per_turn=300) == ""
    assert _format_history_block([], max_turns=2, max_chars_per_turn=300) == ""


def test_format_history_block_keeps_last_n_pairs():
    convo = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
        {"role": "user", "content": "Q3"},
        {"role": "assistant", "content": "A3"},
    ]
    block = _format_history_block(convo, max_turns=2, max_chars_per_turn=300)
    # Only last 4 messages (2 pairs) survive.
    assert "Q1" not in block
    assert "A1" not in block
    assert "Q2" in block and "A2" in block
    assert "Q3" in block and "A3" in block


def test_format_history_block_truncates_long_messages():
    long_text = "x" * 800
    convo = [{"role": "assistant", "content": long_text}]
    block = _format_history_block(convo, max_turns=2, max_chars_per_turn=200)
    # Truncated + ellipsis suffix.
    assert " ..." in block
    # Body length is bounded.
    assert len(block) < 250


# ---------- 2. Prompt builder behavior ----------

def test_prompt_omits_history_when_no_conversation():
    system, user = build_answer_prompt("PLUS 多少钱", [_hit()], "zh", conversation=None)
    # Base zh system prompt only — no history rule.
    assert HISTORY_RULE_ZH not in system
    # User block has no "Recent conversation:" header.
    assert "Recent conversation" not in user


def test_prompt_includes_history_and_rule_when_conversation_present():
    convo = [
        {"role": "user", "content": "PLUS 会员价格"},
        {"role": "assistant", "content": "PLUS 年费 198 元[1]。"},
    ]
    system, user = build_answer_prompt("那 PRO 呢", [_hit()], "zh", conversation=convo)
    # System prompt now contains the strict-history rule.
    assert HISTORY_RULE_ZH in system
    # User block leads with the conversation.
    assert "Recent conversation:" in user
    assert "PLUS 年费 198 元" in user
    # Original chunk + question still present.
    assert "Question: 那 PRO 呢" in user
    assert "chunk 1 body content" in user


def test_prompt_english_history_rule():
    convo = [
        {"role": "user", "content": "What's the price of PLUS?"},
        {"role": "assistant", "content": "PLUS is 198 RMB/year [1]."},
    ]
    system, user = build_answer_prompt("And PRO?", [_hit()], "en", conversation=convo)
    assert HISTORY_RULE_EN in system
    assert "Recent conversation:" in user


def test_prompt_disabled_via_settings_flag(monkeypatch):
    """When enable_answer_with_history=False, history is dropped even if
    the caller passes a conversation. This is the safety valve."""
    from rag import config as cfg
    monkeypatch.setattr(cfg.settings, "enable_answer_with_history", False)

    convo = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    system, user = build_answer_prompt("Q", [_hit()], "zh", conversation=convo)
    assert HISTORY_RULE_ZH not in system
    assert "Recent conversation" not in user


# ---------- 3. Backward compatibility ----------

def test_prompt_signature_backward_compatible():
    """Existing callers that pass only (query, hits, language) still work
    — conversation is optional. Locks the public API."""
    system, user = build_answer_prompt("Q", [_hit()], "zh")
    assert "Question: Q" in user
    assert "[1]" in user  # numbered citation header present
