"""Abstain policy + citation parsing.

Abstain logic:
- If no hits: abstain.
- If the best rerank score is below the threshold: abstain.
- If the LLM returns the abstain sentence: propagate it.

Citation parsing:
- Find all [n] markers in the answer and map them to hits by position.
- Deduplicate while preserving order so the answer card shows each source once.
"""
from __future__ import annotations

import re

from ..config import settings
from ..types import Citation, Hit


ABSTAIN_EN = "I don't have enough information in the knowledge base to answer this."
ABSTAIN_ZH = "当前知识库内信息不足,无法回答这个问题。"

_CITATION_RE = re.compile(r"\[(\d+)\]")


def should_abstain(hits: list[Hit]) -> tuple[bool, str]:
    if not hits:
        return True, "no_retrieval_hits"
    top = hits[0].score
    if top < settings.abstain_score_threshold:
        return True, f"top_score_below_threshold ({top:.3f})"
    return False, ""


def is_abstain_answer(text: str) -> bool:
    t = text.strip()
    return t.startswith(ABSTAIN_EN[:20]) or t.startswith(ABSTAIN_ZH[:15])


def parse_citations(answer_text: str, hits: list[Hit]) -> list[Citation]:
    """Map [n] markers in the answer to Citation entries."""
    indices: list[int] = []
    for m in _CITATION_RE.finditer(answer_text):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(hits) and idx not in indices:
            indices.append(idx)

    citations: list[Citation] = []
    for i in indices:
        hit = hits[i]
        snippet = hit.text.strip()
        if len(snippet) > 240:
            snippet = snippet[:240].rstrip() + " ..."
        citations.append(
            Citation(
                n=i + 1,
                source_id=hit.source_id,
                source_path=hit.source_path,
                title=hit.title,
                section_path=hit.section_path,
                snippet=snippet,
            )
        )
    return citations
