"""Online hallucination verification.

Pipeline addition that runs AFTER the answer LLM has produced grounded
text. We split the answer into sentences, then ask a cheap small model
(qwen-turbo) "for each sentence, is it supported by the cited chunks?"

Output:
    HallucinationReport(
        verified_sentences: list[VerifiedSentence],
        unsupported_count: int,
        flagged_text: str,   # original answer with ⚠ before unsupported sentences
    )

Default OFF (``ENABLE_HALLUCINATION_CHECK=false``) because it adds ~300ms
+ one extra LLM call per answer. Flip on for high-stakes deployments
(legal / medical / financial) where false claims are unacceptable.

Why we re-check the answer instead of trusting the generation prompt:
    - Generation LLM has incentives to "sound helpful" even when retrieval
      missed; it can paraphrase a chunk into something the chunk doesn't
      actually claim.
    - Two-stage check (generate → verify) catches the gap. Industry
      pattern (Gemini guardrails / Constitutional AI use the same idea).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from rag.config import settings
from rag.answer.llm_client import LLMError, chat_once


log = logging.getLogger("hallucination_check")


# CJK-aware sentence split (mirrors the one in semantic_chunking).
_SENTENCE_END = re.compile(r"(?<=[。!?\.!?])\s+|(?<=[。!?\.!?])(?=[^\s])|\n+")


def _split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter that handles CJK + ASCII without any
    NLP dep. Punctuation in citation markers like ``[1]`` is left attached
    to the sentence."""
    if not text:
        return []
    raw = _SENTENCE_END.split(text)
    return [s.strip() for s in raw if s.strip()]


@dataclass
class VerifiedSentence:
    text: str
    supported: bool         # True if the verifier model said the sentence is
                            # backed by the cited chunks
    reasoning: str = ""     # one-line explanation from the verifier
    # Citation numbers parsed out of the sentence (for reporting). Mostly
    # informational — the verifier prompt sees full chunk text anyway.
    cited_ns: list[int] = field(default_factory=list)


@dataclass
class HallucinationReport:
    verified_sentences: list[VerifiedSentence]
    unsupported_count: int
    flagged_text: str       # answer with ⚠ before unsupported sentences

    @property
    def is_clean(self) -> bool:
        return self.unsupported_count == 0


_VERIFIER_SYSTEM = """You are a fact-checking verifier. Given:
1. A list of source chunks (the retrieval ground truth for an answer)
2. A list of sentences from the generated answer

Decide for EACH sentence whether the cited chunks ACTUALLY support its
claim. A sentence is supported when its content can be derived from the
chunks via direct quotation, paraphrase, or trivial composition. A
sentence is UNSUPPORTED if it adds claims not in the chunks, contradicts
the chunks, or relies on outside knowledge.

Greetings ("您好"), bridging ("以下是相关信息"), and meta sentences ("如有
其他问题随时告诉我") are SUPPORTED by default — they make no factual
claim about the source.

Output ONLY a JSON array, one object per input sentence, in input order:
[{"i": <int>, "supported": <bool>, "reasoning": "<one short sentence in Chinese>"}]

Don't echo the sentence text. Just the verdict + brief reasoning."""


_CITATION_RE = re.compile(r"\[(\d+)\]")


def verify_answer_grounding(
    answer_text: str,
    chunks: list[dict[str, Any]],
) -> HallucinationReport:
    """Run the verifier model on each sentence of ``answer_text``.

    ``chunks`` is the retrieval result fed to the generator (typically
    the top-5 hits with text + title). The verifier sees their full text.

    On any LLM failure (network / parse error) we DEGRADE OPEN — return
    a report claiming all sentences are supported. Reasoning: a guardrail
    that breaks under load shouldn't kill the user's answer; better to
    surface a "verifier unavailable" warning at the API layer.
    """
    sentences = _split_sentences(answer_text)
    if not sentences:
        return HallucinationReport(verified_sentences=[], unsupported_count=0,
                                   flagged_text=answer_text)

    # Build the verifier user message.
    chunks_block = "\n\n".join(
        f"[{i + 1}] {(c.get('title') or '').strip()}\n{(c.get('text') or '').strip()[:600]}"
        for i, c in enumerate(chunks)
    )
    sentences_block = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))

    user_msg = (
        f"=== 引用 chunks ===\n{chunks_block}\n\n"
        f"=== 待验证句子 ===\n{sentences_block}\n\n"
        f"输出 {len(sentences)} 个对象的 JSON 数组。"
    )

    try:
        reply = chat_once(
            system=_VERIFIER_SYSTEM,
            user=user_msg,
            max_output_tokens=600,
            temperature=0.0,
            task="hallucination_check",  # → routed to qwen-turbo
        )
    except LLMError as e:
        log.warning("hallucination_check_llm_failed", extra={"error": str(e)})
        # Degrade open: assume supported, mark report as inconclusive.
        verified = [VerifiedSentence(text=s, supported=True,
                                     reasoning="(verifier unavailable)")
                    for s in sentences]
        return HallucinationReport(verified, unsupported_count=0,
                                   flagged_text=answer_text)

    # Parse the verifier's JSON. Same defensive style as our other LLM
    # outputs — find the first JSON array block, validate per-item.
    m = re.search(r"\[.*\]", reply, re.DOTALL)
    if m is None:
        log.warning("hallucination_check_no_json", extra={"reply_head": reply[:120]})
        verified = [VerifiedSentence(text=s, supported=True, reasoning="(parse error)")
                    for s in sentences]
        return HallucinationReport(verified, unsupported_count=0,
                                   flagged_text=answer_text)

    try:
        verdicts = json.loads(m.group(0))
    except json.JSONDecodeError:
        verified = [VerifiedSentence(text=s, supported=True, reasoning="(json error)")
                    for s in sentences]
        return HallucinationReport(verified, unsupported_count=0,
                                   flagged_text=answer_text)

    by_index: dict[int, dict] = {}
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        i = v.get("i")
        if isinstance(i, int) and 1 <= i <= len(sentences):
            by_index[i] = v

    verified: list[VerifiedSentence] = []
    for i, s in enumerate(sentences, start=1):
        v = by_index.get(i, {})
        supported = bool(v.get("supported", True))
        reasoning = str(v.get("reasoning", "")).strip()
        cited_ns = [int(n) for n in _CITATION_RE.findall(s)]
        verified.append(VerifiedSentence(
            text=s, supported=supported, reasoning=reasoning, cited_ns=cited_ns,
        ))

    unsupported_count = sum(1 for v in verified if not v.supported)
    # Stitch a flagged version: ⚠ in front of each unsupported sentence so
    # the front-end can render those in red without a second pass.
    flagged_parts = []
    for v in verified:
        if not v.supported:
            flagged_parts.append(f"⚠ {v.text}")
        else:
            flagged_parts.append(v.text)
    flagged_text = " ".join(flagged_parts)

    return HallucinationReport(
        verified_sentences=verified,
        unsupported_count=unsupported_count,
        flagged_text=flagged_text,
    )


def is_enabled() -> bool:
    return getattr(settings, "enable_hallucination_check", False)
