"""Answer handlers for non-KB intents: meta-query and chitchat.

Both bypass retrieval and call the LLM directly. The meta handler is the
important one — it answers questions ABOUT the conversation history
("translate what you just said") without re-running RAG, which was the
defect that motivated this whole module.

Failure mode: on LLM error, return a polite fixed string. We never raise
out of these handlers, because intent routing has already committed the
caller to skipping retrieval; a raise here would leave the user with no
answer at all.
"""
from __future__ import annotations

from .llm_client import LLMError, chat_once, chat_once_async, chat_stream_async


_META_SYSTEM_ZH = """你是一个对话助手。用户的最新问题是关于上一轮对话内容的(例如「刚刚说的」「翻译一下上面」「重复一下你刚才的回答」)。

规则:
- 你的回答必须基于下面给出的对话历史中已经出现的内容,不要引入新的事实。
- 如果用户要求翻译、总结、重复、解释上一轮内容,请直接对历史中最后一条 Assistant 消息进行操作。
- 如果对话历史里找不到用户提到的内容,礼貌说明并请用户重新提问。
- 不要使用 [n] 这种引用标记;这一轮没有检索结果。
- 用与用户相同的语言作答。
"""

_META_SYSTEM_EN = """You are a conversational assistant. The user's latest question refers to PRIOR turns of this conversation (e.g. "what did you just say", "translate that", "summarize the above").

Rules:
- Base your answer ONLY on content that already appears in the conversation history below. Do not introduce new facts.
- If the user asks to translate / summarize / repeat / explain the previous turn, operate on the last Assistant message in the history.
- If the referenced content is not in the history, politely say so and ask the user to repeat.
- Do NOT use [n] citation markers — there is no retrieval this turn.
- Reply in the same language the user used.
"""

_CHITCHAT_SYSTEM_ZH = """你是某企业 RAG + Agent 客服平台的对话助手。用户在跟你打招呼或闲聊。

规则:
- 简短回答(2-3 句),自然口语化。
- 必要时简单介绍自己:可以回答上传到知识库的内容、可以代为查订单/办退货/开发票/投诉等。
- 不要编造产品名;不要使用 [n] 引用。
"""

_CHITCHAT_SYSTEM_EN = """You are a friendly assistant for an enterprise RAG + Agent customer-service platform. The user is greeting or making small talk.

Rules:
- Reply in 2-3 short sentences.
- If asked who you are or what you can do, mention you can answer questions from uploaded knowledge bases and handle order / return / invoice / complaint actions.
- Do NOT invent product names. Do NOT use [n] citation markers.
"""

_META_FALLBACK_ZH = "抱歉,我暂时无法回答关于上一轮对话的问题,请直接把您想问的内容再说一次。"
_META_FALLBACK_EN = "Sorry, I can't reference our previous turn right now — could you restate the question?"
_CHITCHAT_FALLBACK_ZH = "你好!我是企业知识库助手,可以回答上传过的文档,或帮你查订单、办退货、开发票、提工单。"
_CHITCHAT_FALLBACK_EN = "Hi! I'm an enterprise knowledge-base assistant. I can answer questions about uploaded documents, or help with orders, returns, invoices and complaints."


def _format_history(conversation: list[dict] | None, *, tail: int = 6) -> str:
    if not conversation:
        return ""
    lines: list[str] = []
    for turn in conversation[-tail:]:
        role = (turn.get("role") or "user").lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        tag = "User" if role == "user" else "Assistant"
        # Don't truncate meta-answer history as aggressively as rewrite does;
        # the assistant's previous full answer is what we're operating on.
        if len(content) > 1500:
            content = content[:1500].rstrip() + " ..."
        lines.append(f"{tag}: {content}")
    return "\n".join(lines)


def answer_meta_sync(
    query: str,
    conversation: list[dict] | None,
    language: str,
) -> str:
    """Sync variant for the legacy answer_query() entry point."""
    history = _format_history(conversation)
    if not history:
        return _META_FALLBACK_ZH if language == "zh" else _META_FALLBACK_EN
    system = _META_SYSTEM_ZH if language == "zh" else _META_SYSTEM_EN
    user = f"Conversation so far:\n{history}\n\nUser's latest question: {query.strip()}"
    try:
        return chat_once(
            system=system, user=user, task="meta_query",
            temperature=0.2, max_output_tokens=600,
        ).strip()
    except LLMError:
        return _META_FALLBACK_ZH if language == "zh" else _META_FALLBACK_EN


def answer_chitchat_sync(query: str, language: str) -> str:
    system = _CHITCHAT_SYSTEM_ZH if language == "zh" else _CHITCHAT_SYSTEM_EN
    try:
        return chat_once(
            system=system, user=query.strip(), task="chitchat",
            temperature=0.3, max_output_tokens=200,
        ).strip()
    except LLMError:
        return _CHITCHAT_FALLBACK_ZH if language == "zh" else _CHITCHAT_FALLBACK_EN


async def answer_meta(
    query: str,
    conversation: list[dict] | None,
    language: str,
) -> str:
    """Answer a meta-question by reading the conversation history."""
    history = _format_history(conversation)
    if not history:
        return _META_FALLBACK_ZH if language == "zh" else _META_FALLBACK_EN

    system = _META_SYSTEM_ZH if language == "zh" else _META_SYSTEM_EN
    user = f"Conversation so far:\n{history}\n\nUser's latest question: {query.strip()}"
    try:
        return (await chat_once_async(
            system=system,
            user=user,
            task="meta_query",
            temperature=0.2,
            max_output_tokens=600,
        )).strip()
    except LLMError:
        return _META_FALLBACK_ZH if language == "zh" else _META_FALLBACK_EN


async def answer_chitchat(query: str, language: str) -> str:
    system = _CHITCHAT_SYSTEM_ZH if language == "zh" else _CHITCHAT_SYSTEM_EN
    try:
        return (await chat_once_async(
            system=system,
            user=query.strip(),
            task="chitchat",
            temperature=0.3,
            max_output_tokens=200,
        )).strip()
    except LLMError:
        return _CHITCHAT_FALLBACK_ZH if language == "zh" else _CHITCHAT_FALLBACK_EN


async def stream_meta(query: str, conversation: list[dict] | None, language: str):
    """Streaming variant. Yields text deltas; final result must be the full answer."""
    history = _format_history(conversation)
    if not history:
        yield _META_FALLBACK_ZH if language == "zh" else _META_FALLBACK_EN
        return
    system = _META_SYSTEM_ZH if language == "zh" else _META_SYSTEM_EN
    user = f"Conversation so far:\n{history}\n\nUser's latest question: {query.strip()}"
    try:
        async for piece in chat_stream_async(
            system=system,
            user=user,
            task="meta_query",
            temperature=0.2,
            max_output_tokens=600,
        ):
            yield piece
    except LLMError:
        yield _META_FALLBACK_ZH if language == "zh" else _META_FALLBACK_EN


async def stream_chitchat(query: str, language: str):
    system = _CHITCHAT_SYSTEM_ZH if language == "zh" else _CHITCHAT_SYSTEM_EN
    try:
        async for piece in chat_stream_async(
            system=system,
            user=query.strip(),
            task="chitchat",
            temperature=0.3,
            max_output_tokens=200,
        ):
            yield piece
    except LLMError:
        yield _CHITCHAT_FALLBACK_ZH if language == "zh" else _CHITCHAT_FALLBACK_EN
