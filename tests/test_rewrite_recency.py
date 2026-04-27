"""Rewriter recency-bias tests.

Verifies the system prompt contains the explicit "most recent assistant
message wins" rule, and that conversation history is correctly assembled
into the user prompt for the LLM. The LLM call itself is mocked — we
test the contract between rewrite.py and llm_client.py, not the
underlying model's instruction-following (that is verified live).
"""
from __future__ import annotations

from rag.query import rewrite as rw


def test_system_prompt_has_recency_rule():
    """The SYSTEM prompt must explicitly tell the LLM to prefer the most
    recent assistant message — without this clause the LLM tends to grab
    the earliest user question's subject (the W02-L01 bug we're fixing)."""
    sys_p = rw.SYSTEM
    assert "Recency" in sys_p or "recency" in sys_p or "MOST RECENT" in sys_p
    # Concrete cue words from the rule.
    assert "most recent assistant" in sys_p.lower() or "MOST RECENT assistant" in sys_p


def test_system_prompt_has_translate_example():
    """An example covering the operate-on-prior-content case (translate /
    summarize) anchors the rule for the LLM."""
    sys_p = rw.SYSTEM
    assert "翻译" in sys_p
    # The good rewrite should reference Liskov etc., not "课件讲了什么".
    assert "Liskov" in sys_p


def test_format_conversation_keeps_assistant_turn_intact():
    """The history block must include the latest assistant message in full
    so the LLM can quote from it. We clip at 400 chars per turn — that's
    enough to carry the subject without blowing up token cost."""
    convo = [
        {"role": "user", "content": "课件讲了什么"},
        {"role": "assistant", "content": "讲了软件设计原则的概念。"},
        {"role": "user", "content": "具体讲讲"},
        {"role": "assistant", "content": "Liskov 替换原则、接口隔离原则、组合优于继承。"},
    ]
    block = rw._format_conversation(convo)
    # Both assistant turns survive; latest is the recency anchor.
    assert "Liskov" in block
    assert "组合优于继承" in block
    # Latest turn appears AFTER the earliest one.
    assert block.index("Liskov") > block.index("软件设计原则")


def test_rewriter_passes_history_to_llm(monkeypatch):
    """Sanity-check: the user_msg sent to chat_once must contain the
    latest assistant turn so the LLM can apply the recency rule."""
    captured = {}
    def fake_chat_once(**kwargs):
        captured["system"] = kwargs.get("system")
        captured["user"] = kwargs.get("user")
        return "Liskov substitution interface segregation translate English"
    monkeypatch.setattr(rw, "chat_once", fake_chat_once)

    convo = [
        {"role": "user", "content": "课件讲了什么"},
        {"role": "assistant", "content": "讲了软件设计原则。"},
        {"role": "user", "content": "具体讲讲"},
        {"role": "assistant", "content": "Liskov 替换原则、接口隔离、组合优于继承。"},
    ]
    out = rw.rewrite_query("翻译成英文", conversation=convo)
    # The assistant's latest content must be visible to the LLM.
    assert "Liskov" in captured["user"]
    assert "组合优于继承" in captured["user"]
    # And the recency rule must be in the system prompt.
    assert "MOST RECENT assistant" in captured["system"]
    # The mocked LLM output is returned verbatim when conversation is present.
    assert out == "Liskov substitution interface segregation translate English"
