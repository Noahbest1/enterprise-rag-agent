"""Sprint A fallback-chain tests.

Covers:
- ``chat_once_with_fallback`` happy path returns ``(text, "qwen")``
- disabled chain re-raises on Qwen failure
- Qwen fails + Claude configured + Claude succeeds -> returns ("claude")
- Qwen fails + Claude configured + Claude fails -> extractive
- Qwen fails + Claude NOT configured -> extractive straight away
- Extractive respects hit ordering, builds ``[n]`` citations, handles empty hits
- Claude adapter: parses Anthropic content blocks + records tokens + maps HTTP/Net errors to LLMError
"""
from __future__ import annotations

from unittest.mock import patch, AsyncMock

import httpx
import pytest

from rag.answer.fallback import (
    chat_once_with_fallback,
    claude_chat_async,
    extractive_answer,
)
from rag.answer.llm_client import LLMError
from rag.config import settings
from rag.types import Hit


def _hit(i: int, title: str, text: str) -> Hit:
    return Hit(
        chunk_id=f"c{i}",
        score=1.0 - 0.1 * i,
        text=text,
        title=title,
        source_id=f"s{i}",
        source_path="/p",
        section_path=[],
        retrieval_source="test",
    )


# ---------- extractive ----------

def test_extractive_empty_hits():
    out = extractive_answer("退货政策", [])
    assert "知识库内信息不足" in out


def test_extractive_formats_topk_with_citations():
    hits = [
        _hit(0, "7 天无理由", "京东自营商品支持 7 天无理由退换货。"),
        _hit(1, "退货流程", "登录京东 APP,进入我的订单页面。"),
        _hit(2, "退款时效", "退款 1-7 个工作日原路返回。"),
        _hit(3, "不该出现", "fourth hit should be trimmed"),
    ]
    out = extractive_answer("退货怎么办", hits)
    # Only top 3 shown
    assert "7 天无理由" in out
    assert "退款时效" in out
    assert "fourth hit" not in out
    # Includes numbered citations
    assert "[1]" in out and "[2]" in out and "[3]" in out
    assert "LLM 服务暂不可用" in out


# ---------- claude adapter ----------

@pytest.mark.asyncio
async def test_claude_chat_no_key(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    with pytest.raises(LLMError) as e:
        await claude_chat_async(system="sys", user="hi")
    assert "ANTHROPIC_API_KEY" in str(e.value)


@pytest.mark.asyncio
async def test_claude_chat_parses_content(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")

    # Build a fake httpx.AsyncClient that returns a canned Anthropic response.
    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {
                "content": [{"type": "text", "text": "claude says hi"}],
                "usage": {"input_tokens": 5, "output_tokens": 7},
            }

    class FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json, headers): return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    text = await claude_chat_async(system="sys", user="hi")
    assert text == "claude says hi"


@pytest.mark.asyncio
async def test_claude_chat_http_error_maps_to_llmerror(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")

    class FakeResp:
        status_code = 500
        text = "boom"
        def raise_for_status(self):
            raise httpx.HTTPStatusError("500", request=None, response=self)
        def json(self): return {}

    class FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json, headers): return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    with pytest.raises(LLMError) as e:
        await claude_chat_async(system="sys", user="hi")
    assert "Claude HTTP" in str(e.value)


# ---------- chain ----------

@pytest.mark.asyncio
async def test_chain_happy_path_returns_qwen(monkeypatch):
    monkeypatch.setattr(settings, "enable_fallback_chain", True)
    with patch(
        "rag.answer.fallback.chat_once_async",
        new=AsyncMock(return_value="qwen answer"),
    ):
        text, provider = await chat_once_with_fallback(system="s", user="u", task="generate")
    assert provider == "qwen"
    assert text == "qwen answer"


@pytest.mark.asyncio
async def test_chain_disabled_propagates_qwen_error(monkeypatch):
    monkeypatch.setattr(settings, "enable_fallback_chain", False)
    with patch(
        "rag.answer.fallback.chat_once_async",
        new=AsyncMock(side_effect=LLMError("down")),
    ):
        with pytest.raises(LLMError):
            await chat_once_with_fallback(system="s", user="u")


@pytest.mark.asyncio
async def test_chain_falls_back_to_claude(monkeypatch):
    monkeypatch.setattr(settings, "enable_fallback_chain", True)
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    with patch(
        "rag.answer.fallback.chat_once_async",
        new=AsyncMock(side_effect=LLMError("qwen down")),
    ), patch(
        "rag.answer.fallback.claude_chat_async",
        new=AsyncMock(return_value="claude covers"),
    ):
        text, provider = await chat_once_with_fallback(system="s", user="u")
    assert provider == "claude"
    assert text == "claude covers"


@pytest.mark.asyncio
async def test_chain_falls_through_to_extractive_when_both_fail(monkeypatch):
    monkeypatch.setattr(settings, "enable_fallback_chain", True)
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    hits = [_hit(0, "Policy", "policy body.")]
    with patch(
        "rag.answer.fallback.chat_once_async",
        new=AsyncMock(side_effect=LLMError("qwen down")),
    ), patch(
        "rag.answer.fallback.claude_chat_async",
        new=AsyncMock(side_effect=LLMError("claude down")),
    ):
        text, provider = await chat_once_with_fallback(
            system="s", user="u", hits=hits, user_query_for_extractive="what policy",
        )
    assert provider == "extractive"
    assert "[1]" in text
    assert "Policy" in text


@pytest.mark.asyncio
async def test_chain_skips_claude_when_no_key(monkeypatch):
    monkeypatch.setattr(settings, "enable_fallback_chain", True)
    monkeypatch.setattr(settings, "anthropic_api_key", "")  # unconfigured
    hits = [_hit(0, "t", "body body body.")]
    claude_mock = AsyncMock(return_value="should-not-be-called")
    with patch(
        "rag.answer.fallback.chat_once_async",
        new=AsyncMock(side_effect=LLMError("qwen down")),
    ), patch("rag.answer.fallback.claude_chat_async", new=claude_mock):
        text, provider = await chat_once_with_fallback(
            system="s", user="u", hits=hits, user_query_for_extractive="q",
        )
    assert provider == "extractive"
    claude_mock.assert_not_called()
