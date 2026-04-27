"""Three-way intent classification: meta-query / chitchat / kb-query.

Solves a real RAG defect: when a user asks a question ABOUT prior turns
("刚刚翻译的是什么", "summarize what you just said"), the existing pipeline
re-runs retrieval against the rewritten query, which fetches arbitrary
KB chunks and the LLM hallucinates a "summary" of content it never wrote.

This module decides the route BEFORE retrieval:

    meta     -> answer from conversation history, no KB hit
    chitchat -> short LLM reply, no KB hit
    kb       -> existing retrieve+rerank+answer path (default)

We use a pure-rule classifier (regex on keyword lists) for two reasons:
1. Latency. A pre-retrieval LLM call would add ~300ms to every query.
2. Determinism. Rules are testable and explainable; an LLM that flips
   "kb" to "chitchat" on a real KB question would be a regression
   nobody could reproduce.

The patterns below were chosen from concrete user turns we observed in
testing; meta keywords trigger only when the query is short and clearly
about prior turns (so a long question that happens to contain "上面"
in another sense doesn't get misrouted).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class IntentVerdict:
    intent: str  # "meta" | "chitchat" | "kb"
    matched: str | None  # the pattern fragment that fired (for trace/debug)
    via: str  # "rule" | "fallback"


# Meta-query patterns. We require that the query is SHORT (<= 40 chars) and
# the keyword appears as a clear reference to prior turns. A long technical
# question that mentions "above" in passing should still go to KB.
_META_PATTERNS_ZH = [
    r"刚刚(?:说|讲|提到|翻译|回答)",
    r"刚才(?:说|讲|提到|翻译|回答)",
    r"上(?:面|述|一句|一条|一段|一轮)(?:的|是|说)",
    r"上(?:面|述)(?:翻译|内容|说的|提到)",
    r"前(?:面|一(?:句|条|段|轮))(?:说|提到|的)",
    # "之前" — covers "我之前问的", "之前提到", "之前说的"
    r"(?:我)?之前(?:问|提到|说|讲|回答|的)",
    # 你刚 / 你刚才 followed by 说/讲/回答, OR 的+说/回答/话/内容
    r"你刚(?:才|刚)?(?:说|讲|提到|回答|的(?:回答|话|内容|那(?:句|段|条)))",
    # translate/summarize/explain/repeat + 上面/刚才/前面/你刚才/你说的 etc.
    r"(?:翻译|总结|解释|重(?:复|说))(?:一下|下)?(?:上面|上述|刚刚|刚才|前面|你说的|你刚(?:才)?|之前)",
    r"^(?:翻译|总结)(?:一下|下)?(?:上面|刚才|刚刚|前面|之前)",
]

_META_PATTERNS_EN = [
    r"\bwhat (?:did|do) you (?:just )?say\b",
    r"\bwhat was (?:the )?(?:above|previous|last)\b",
    r"\bwhat (?:did )?you (?:just )?(?:translate|answer|mention)\b",
    r"\b(?:summari[sz]e|translate|repeat|explain) (?:the )?(?:above|previous|last|that)\b",
    r"\byour (?:previous|last) (?:answer|reply|response|message)\b",
]

# Chitchat: greetings + identity + capability questions. These should NOT
# trigger KB retrieval because the answer doesn't come from any KB.
_CHITCHAT_PATTERNS_ZH = [
    r"^你好[啊呀\s!.。？?]*$",
    r"^(?:你好|您好|hi|hey|hello)[，,\s]*(?:你是(?:谁|什么)|请问你是)",
    r"^你是(?:谁|什么|哪个|什么人)",
    r"^你能(?:做|帮我做|帮)(?:什么|啥|哪些)",
    r"^(?:在吗|您在吗|in吗)",
    r"^(?:谢谢|多谢|感谢)(?:你|您)?[\s!.。]*$",
    r"^(?:再见|拜拜|bye)[\s!.。]*$",
]

_CHITCHAT_PATTERNS_EN = [
    r"^(?:hi|hey|hello|yo)[\s!.,]*$",
    r"^(?:hi|hey|hello)[\s,]+who are you\b",
    r"^who are you\b",
    r"^what can you do\b",
    r"^(?:thanks|thank you|thx|ty)[\s!.]*$",
    r"^(?:bye|goodbye)[\s!.]*$",
]


_META_RE = re.compile("|".join(_META_PATTERNS_ZH + _META_PATTERNS_EN), re.IGNORECASE)
_CHITCHAT_RE = re.compile("|".join(_CHITCHAT_PATTERNS_ZH + _CHITCHAT_PATTERNS_EN), re.IGNORECASE)

# Length cap for meta detection. Anything longer is almost certainly a real
# question that happens to use the word "above" or "previous" — we'd rather
# pay one retrieval than misroute.
_META_MAX_LEN = 40


def classify_intent(query: str, *, has_conversation: bool) -> IntentVerdict:
    """Classify a query into meta / chitchat / kb.

    ``has_conversation`` is True when the caller has at least one prior
    user+assistant pair in scope. Without history, a "meta" query has
    nothing to refer to, so we degrade to ``kb`` (existing behavior).
    """
    q = (query or "").strip()
    if not q:
        return IntentVerdict(intent="kb", matched=None, via="rule")

    # Chitchat first: greetings shouldn't be kb-ified even with a long
    # conversation in flight.
    m = _CHITCHAT_RE.search(q)
    if m:
        return IntentVerdict(intent="chitchat", matched=m.group(0)[:60], via="rule")

    # Meta requires both: keyword match AND prior turns AND short length.
    if has_conversation and len(q) <= _META_MAX_LEN:
        m = _META_RE.search(q)
        if m:
            return IntentVerdict(intent="meta", matched=m.group(0)[:60], via="rule")

    return IntentVerdict(intent="kb", matched=None, via="rule")
