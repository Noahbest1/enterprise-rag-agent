"""Native function-calling loop for specialists that need LLM-driven reasoning.

This is a thin wrapper around Qwen's OpenAI-compatible tool_calls API.
Specialists register their available tools (JSON schema + Python callable)
and this loop runs:

    while not done and steps < max_steps:
        llm_response = chat_with_tools(messages, tools)
        if llm_response.tool_calls:
            for tc in tool_calls:
                result = dispatch(tc.name, tc.args)
                messages.append({"role":"tool", "tool_call_id":..., "content": result})
        else:
            return llm_response.content

Why keep the Planner / Coordinator separate from this?
The top-level graph (Plan-and-Execute) answers "what high-level skills do
we need"; this module is for "within a skill, which concrete tools does
the LLM want to use in what order". Different granularity, different
failure modes, simpler to reason about.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from rag.answer.llm_client import LLMError, _headers
from rag.config import settings
from rag.logging import get_logger
from rag.model_routing import resolve_model


log = get_logger(__name__)


@dataclass
class Tool:
    """A tool the LLM can choose to call via native function calling."""
    name: str
    description: str
    parameters: dict  # JSON Schema
    fn: Callable[..., Any]  # returns JSON-serialisable

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCallTrace:
    name: str
    args: dict
    result: Any
    error: str | None = None


@dataclass
class ToolCallingResult:
    final_text: str
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    raw_messages: list[dict] = field(default_factory=list)


def run_tool_calling_loop(
    *,
    system_prompt: str,
    user_prompt: str,
    tools: list[Tool],
    max_steps: int = 4,
    temperature: float = 0.1,
    task: str = "intent",
    timeout: int | None = None,
) -> ToolCallingResult:
    """Execute a bounded native function-calling loop.

    Returns structured trace of every tool call + final assistant text.
    Raises LLMError on upstream failure. Bounds loops hard (``max_steps``)
    to avoid runaway costs if the LLM keeps asking to call tools.
    """
    if not settings.qwen_api_key:
        raise LLMError("DASHSCOPE_API_KEY not set")
    tools_by_name = {t.name: t for t in tools}
    tool_schemas = [t.to_openai_schema() for t in tools]

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    trace: list[ToolCallTrace] = []

    with httpx.Client(timeout=timeout or settings.llm_timeout_seconds) as client:
        for step in range(max_steps):
            payload = {
                "model": resolve_model(task),  # type: ignore[arg-type]
                "messages": messages,
                "tools": tool_schemas,
                "tool_choice": "auto",
                "temperature": temperature,
                "max_tokens": settings.llm_max_output_tokens,
            }
            try:
                resp = client.post(settings.qwen_chat_url, json=payload, headers=_headers())
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as e:
                raise LLMError(f"tool-calling HTTP {e.response.status_code}: {e.response.text[:300]}") from e
            except httpx.RequestError as e:
                raise LLMError(f"tool-calling network error: {e}") from e

            msg = data["choices"][0]["message"]
            messages.append(msg)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                content = msg.get("content") or ""
                return ToolCallingResult(final_text=content.strip(), tool_calls=trace, raw_messages=messages)

            # Dispatch every requested tool this turn.
            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}

                tool = tools_by_name.get(name)
                if tool is None:
                    err = f"unknown tool: {name}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps({"error": err}),
                    })
                    trace.append(ToolCallTrace(name=name, args=args, result=None, error=err))
                    continue

                try:
                    result = tool.fn(**args) if args else tool.fn()
                    trace.append(ToolCallTrace(name=name, args=args, result=result))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    })
                except Exception as e:  # noqa: BLE001 -- surface tool errors to LLM
                    err = f"{type(e).__name__}: {e}"
                    log.warning("tool_call_error", tool=name, error=err)
                    trace.append(ToolCallTrace(name=name, args=args, result=None, error=err))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps({"error": err}),
                    })

    # Hit max_steps without the model settling. Return the last assistant content (if any).
    last_assistant = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant" and m.get("content")),
        "",
    )
    return ToolCallingResult(
        final_text=(last_assistant or "(未在最大步数内完成推理)").strip(),
        tool_calls=trace,
        raw_messages=messages,
    )
