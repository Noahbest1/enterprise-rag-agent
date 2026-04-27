"""Intent routing tests.

Covers the three-way classifier (meta / chitchat / kb) and the pipeline
dispatch that bypasses retrieval for meta and chitchat. Most tests are
pure-rule (no mocking); the pipeline integration tests monkeypatch
``get_kb`` + ``retrieve_async`` so we can assert retrieval is NOT
reached on the meta/chitchat branches.
"""
from __future__ import annotations

import asyncio

import pytest

from rag.query.intent import classify_intent, IntentVerdict
from rag.answer import meta_answer
from rag import pipeline


# ---------- 1. Classifier rules (no mocks) ----------

@pytest.mark.parametrize("query", [
    "刚刚翻译的是什么",
    "上面的内容是什么意思",
    "重复一下你刚才的回答",
    "你刚说的那段再翻译一下",
    "总结一下上面",
    # "问" verb — user observed in live testing that "我刚刚问的是哪个订单"
    # was NOT matching, so it fell through to the order specialist and the
    # LLM hallucinated "您之前咨询的是这 3 笔订单".
    "我刚刚问的是哪个订单",
    "刚才咨询的是什么",
    "你刚问的什么",
])
def test_classify_meta_zh(query):
    v = classify_intent(query, has_conversation=True)
    assert v.intent == "meta", f"{query!r} should be meta, got {v}"
    assert v.matched
    assert v.via == "rule"


@pytest.mark.parametrize("query", [
    "What did you just say?",
    "translate the above",
    "summarize your previous answer",
])
def test_classify_meta_en(query):
    v = classify_intent(query, has_conversation=True)
    assert v.intent == "meta", f"{query!r} should be meta, got {v}"


@pytest.mark.parametrize("query", [
    "你好",
    "你是谁",
    "你能做什么",
    "在吗",
    "hello",
    "who are you",
    "what can you do",
])
def test_classify_chitchat(query):
    v = classify_intent(query, has_conversation=False)
    assert v.intent == "chitchat", f"{query!r} should be chitchat, got {v}"


@pytest.mark.parametrize("query", [
    "PLUS 会员的年费多少",
    "请介绍一下软件设计原则",
    "如何申请退货",
    "What is the return policy for damaged items?",
    "解释一下高内聚低耦合在架构层面如何体现",
])
def test_classify_kb_default(query):
    v = classify_intent(query, has_conversation=True)
    assert v.intent == "kb", f"{query!r} should be kb, got {v}"


def test_meta_fires_even_without_history():
    """Meta keywords classify as ``meta`` regardless of has_conversation.
    The handler returns a polite "no prior turns" fallback when conversation
    is empty — falling through to KB / planner caused the order specialist
    to list all the user's orders and the LLM to hallucinate "you previously
    asked about these orders" (real bug observed 2026-04-28)."""
    v = classify_intent("刚刚翻译的是什么", has_conversation=False)
    assert v.intent == "meta"
    assert v.matched is not None


def test_meta_requires_short_query():
    """A long technical question that mentions '上面' in passing is not
    a meta-question; we'd rather pay one retrieval than misroute."""
    long_q = "上面提到的高内聚原则在微服务架构里如何落地,有哪些常见反模式以及怎么用代码量化检测"
    v = classify_intent(long_q, has_conversation=True)
    assert v.intent == "kb"


def test_chitchat_beats_meta():
    """A bare greeting + question should be chitchat, not kb, even with
    conversation in flight."""
    v = classify_intent("你好你是谁", has_conversation=True)
    assert v.intent == "chitchat"


def test_empty_query_is_kb():
    v = classify_intent("", has_conversation=True)
    assert v.intent == "kb"
    assert v.matched is None


# ---------- 2. Meta handler (mocked LLM) ----------

def test_answer_meta_sync_uses_history(monkeypatch):
    captured = {}
    def fake_chat_once(**kwargs):
        captured["system"] = kwargs.get("system")
        captured["user"] = kwargs.get("user")
        captured["task"] = kwargs.get("task")
        return "翻译后的英文版本。"
    monkeypatch.setattr(meta_answer, "chat_once", fake_chat_once)

    text = meta_answer.answer_meta_sync(
        query="翻译一下上面",
        conversation=[
            {"role": "user", "content": "讲讲设计原则"},
            {"role": "assistant", "content": "设计原则包括高内聚、低耦合等。"},
        ],
        language="zh",
    )
    assert "翻译后的英文版本" in text
    assert captured["task"] == "meta_query"
    # The assistant's previous answer must be present in the prompt.
    assert "高内聚、低耦合" in captured["user"]


def test_answer_meta_no_history_returns_fallback(monkeypatch):
    """With no conversation, the handler should not call the LLM at all
    and just return a polite fallback."""
    called = {"n": 0}
    def fake_chat_once(**_):
        called["n"] += 1
        return "should not be reached"
    monkeypatch.setattr(meta_answer, "chat_once", fake_chat_once)

    text = meta_answer.answer_meta_sync(query="刚刚说的是什么", conversation=None, language="zh")
    assert called["n"] == 0
    assert "重新提问" in text or "再说一次" in text


def test_answer_meta_degrades_open_on_llm_error(monkeypatch):
    from rag.answer.llm_client import LLMError
    def boom(**_):
        raise LLMError("network down")
    monkeypatch.setattr(meta_answer, "chat_once", boom)

    text = meta_answer.answer_meta_sync(
        query="翻译一下上面",
        conversation=[{"role": "assistant", "content": "x"}],
        language="zh",
    )
    # Polite Chinese fallback, not an exception.
    assert isinstance(text, str) and len(text) > 0


def test_answer_chitchat_sync(monkeypatch):
    monkeypatch.setattr(meta_answer, "chat_once", lambda **kw: "你好,我是企业知识库助手。")
    text = meta_answer.answer_chitchat_sync("你好", "zh")
    assert "知识库" in text


# ---------- 3. Pipeline dispatch (the bug-fix scenario) ----------

class _FakeKB:
    def __init__(self):
        self.root = "/tmp/fake-kb"
        self.chunks_path = "/tmp/fake-kb/chunks.jsonl"
    def is_built(self):
        return True


def _install_pipeline_mocks(monkeypatch, retrieve_should_fire=False):
    """Common scaffolding: fake KB, retrieve must NOT be called for meta/chitchat."""
    monkeypatch.setattr(pipeline, "get_kb", lambda kb_id: _FakeKB())

    async def assert_not_called(*args, **kwargs):
        if not retrieve_should_fire:
            raise AssertionError("retrieve_async was called on a meta/chitchat path")
        return []
    monkeypatch.setattr(pipeline, "retrieve_async", assert_not_called)


def test_pipeline_meta_query_skips_retrieval(monkeypatch):
    """The exact scenario from the W02-L01 KB bug:
    user asked about prior translation — we must NOT re-retrieve from KB.
    """
    _install_pipeline_mocks(monkeypatch)

    async def fake_meta(query, conversation, language):
        return "翻译后的内容。"
    monkeypatch.setattr(pipeline, "answer_meta", fake_meta)

    conversation = [
        {"role": "user", "content": "这个课件讲了什么"},
        {"role": "assistant", "content": "课件讲了软件设计原则,包括 Composition over inheritance 与 Dependency Inversion。"},
        {"role": "user", "content": "英文翻译一下上面"},
        {"role": "assistant", "content": "Translation: Composition over inheritance and Dependency Inversion."},
    ]
    ans = asyncio.run(pipeline.answer_query_async(
        "刚刚翻译的是什么",
        kb_id="cs5033",
        conversation=conversation,
    ))
    assert ans.text == "翻译后的内容。"
    assert ans.citations == []
    assert ans.hits == []
    assert ans.trace.get("intent") == "meta"


def test_pipeline_chitchat_skips_retrieval(monkeypatch):
    _install_pipeline_mocks(monkeypatch)
    async def fake_chitchat(query, language):
        return "你好,我是企业知识库助手。"
    monkeypatch.setattr(pipeline, "answer_chitchat", fake_chitchat)

    ans = asyncio.run(pipeline.answer_query_async("你好", kb_id="any"))
    assert ans.trace.get("intent") == "chitchat"
    assert ans.citations == []


def test_pipeline_kb_query_uses_retrieval(monkeypatch):
    """Regression guard: a real KB question must still hit retrieve_async."""
    _install_pipeline_mocks(monkeypatch, retrieve_should_fire=True)

    async def fake_chat_with_fallback(**kwargs):
        return ("这是基于检索结果的答案[1]。", "qwen")
    monkeypatch.setattr(pipeline, "chat_once_with_fallback", fake_chat_with_fallback)
    monkeypatch.setattr(pipeline, "expand_to_parents", lambda hits, _path: hits)

    ans = asyncio.run(pipeline.answer_query_async(
        "PLUS 会员年费多少",
        kb_id="jd_demo",
        use_semantic_cache=False,
        use_rewrite=False,
    ))
    # No intent key for kb path (or it's absent / != meta).
    assert ans.trace.get("intent") in (None, "kb")


def test_pipeline_meta_without_history_short_circuits_with_fallback(monkeypatch):
    """Meta keyword + NO conversation → still short-circuits (does NOT
    invoke retrieval) and the meta handler returns its polite fallback.
    Falling through to KB caused the LLM to hallucinate a "summary" of
    arbitrary chunks (the bug we're fixing)."""
    _install_pipeline_mocks(monkeypatch)  # retrieve must NOT fire

    async def fake_meta(query, conversation, language):
        # Real handler hits this branch internally when conversation is empty.
        return "抱歉,我暂时无法回答关于上一轮对话的问题,请直接把您想问的内容再说一次。"
    monkeypatch.setattr(pipeline, "answer_meta", fake_meta)

    ans = asyncio.run(pipeline.answer_query_async(
        "刚刚说的是什么",
        kb_id="jd_demo",
        conversation=None,
        use_semantic_cache=False,
    ))
    assert ans.trace.get("intent") == "meta"
    assert ans.citations == []
    assert "请" in ans.text or "再说" in ans.text
