"""Native function-calling loop: unit-level (no LLM required)."""
from __future__ import annotations

from unittest.mock import patch

import httpx

from agent.tool_calling import Tool, run_tool_calling_loop


def _make_fake_chat(responses: list[dict]):
    """Build a fake httpx.Client that returns the given chat/completions payloads in order."""
    idx = {"i": 0}

    class FakeResp:
        status_code = 200
        def __init__(self, data):
            self._data = data
        def raise_for_status(self):
            pass
        def json(self):
            return self._data

    class FakeClient:
        def __init__(self, *a, **kw):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        def post(self, *a, **kw):
            r = FakeResp(responses[idx["i"]])
            idx["i"] += 1
            return r

    return FakeClient


def test_loop_returns_final_text_when_no_tool_calls():
    tools = [Tool(name="noop", description="", parameters={"type": "object", "properties": {}}, fn=lambda: "x")]
    responses = [
        {"choices": [{"message": {"role": "assistant", "content": "no tools needed", "tool_calls": []}}]}
    ]
    with patch.object(httpx, "Client", _make_fake_chat(responses)), \
         patch("agent.tool_calling.settings.qwen_api_key", "fake-key"):
        r = run_tool_calling_loop(
            system_prompt="s", user_prompt="u", tools=tools, task="intent",
        )
    assert r.final_text == "no tools needed"
    assert r.tool_calls == []


def test_loop_dispatches_tool_and_feeds_result_back():
    def add(a: int, b: int) -> int:
        return a + b

    tools = [Tool(
        name="add",
        description="add",
        parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]},
        fn=add,
    )]

    responses = [
        # Turn 1: LLM asks to call "add(2, 3)"
        {"choices": [{"message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": "{\"a\": 2, \"b\": 3}"},
            }],
        }}]},
        # Turn 2: LLM sees tool result and answers
        {"choices": [{"message": {
            "role": "assistant", "content": "The sum is 5.", "tool_calls": [],
        }}]},
    ]
    with patch.object(httpx, "Client", _make_fake_chat(responses)), \
         patch("agent.tool_calling.settings.qwen_api_key", "fake-key"):
        r = run_tool_calling_loop(
            system_prompt="s", user_prompt="what is 2+3?", tools=tools, task="intent",
        )
    assert r.final_text == "The sum is 5."
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].name == "add"
    assert r.tool_calls[0].result == 5
    assert r.tool_calls[0].error is None


def test_loop_surfaces_tool_error_to_llm():
    def boom(**kw):
        raise RuntimeError("tool down")

    tools = [Tool(
        name="boom",
        description="",
        parameters={"type": "object", "properties": {}},
        fn=boom,
    )]
    responses = [
        {"choices": [{"message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "boom", "arguments": "{}"}}],
        }}]},
        {"choices": [{"message": {"role": "assistant", "content": "tool failed, apologising.", "tool_calls": []}}]},
    ]
    with patch.object(httpx, "Client", _make_fake_chat(responses)), \
         patch("agent.tool_calling.settings.qwen_api_key", "fake-key"):
        r = run_tool_calling_loop(
            system_prompt="s", user_prompt="u", tools=tools, task="intent",
        )
    assert r.final_text == "tool failed, apologising."
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0].error and "tool down" in r.tool_calls[0].error


def test_loop_respects_max_steps():
    # LLM keeps asking to call tools forever -- loop must terminate.
    tools = [Tool(name="loop", description="", parameters={"type": "object", "properties": {}},
                  fn=lambda: "ok")]
    response = {"choices": [{"message": {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c", "type": "function", "function": {"name": "loop", "arguments": "{}"}}],
    }}]}
    responses = [response] * 10  # plenty
    with patch.object(httpx, "Client", _make_fake_chat(responses)), \
         patch("agent.tool_calling.settings.qwen_api_key", "fake-key"):
        r = run_tool_calling_loop(
            system_prompt="s", user_prompt="u", tools=tools, max_steps=3, task="intent",
        )
    assert len(r.tool_calls) == 3
