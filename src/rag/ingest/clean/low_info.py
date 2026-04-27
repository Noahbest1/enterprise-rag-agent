"""Low-information chunk heuristics.

A "low-info" chunk is one that would pollute retrieval if indexed:
- Empty or near-empty (<20 substantive chars)
- All symbols / separators / numbers (e.g. a table rule, an HR, an all-bullet TOC)
- Single repeated token ("ok ok ok ok ...")
- Boilerplate-only (copyright, "next page", "see also")
- Less than some minimum unique-token ratio

``low_info_score(text) → float in [0,1]`` is continuous so downstream can
set a threshold. ``is_low_info_chunk`` uses 0.7 as the practical cutoff.

These heuristics are cheap (no LLM, no model) and conservative: we'd
rather keep a borderline chunk than drop real content.
"""
from __future__ import annotations

import re


_SUBSTANTIVE_CHAR_RE = re.compile(r"[A-Za-z0-9一-鿿]")
_NON_TEXT_RE = re.compile(r"[\s\-_=+*#~|·—–_\.,:;()\[\]{}<>!?/\\]+")

# Stock phrases that strongly indicate boilerplate. Some we require as the
# whole line (navigation), others we allow anywhere (copyright footers).
_BOILERPLATE_ANCHORED = [
    re.compile(r"^\s*(click\s+here|点击(这里|此处))\s*$", re.IGNORECASE),
    re.compile(r"^\s*(next\s+page|previous\s+page|上一页|下一页)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(table of contents|目录)\s*$", re.IGNORECASE),
]
_BOILERPLATE_ANYWHERE = [
    re.compile(r"\ball\s+rights\s+reserved\b", re.IGNORECASE),
    re.compile(r"版权所有"),
    re.compile(r"\bcookie\s*(policy|settings|consent)\b", re.IGNORECASE),
]


def low_info_score(text: str) -> float:
    """Higher = less informative. 1.0 is "definitely boilerplate"."""
    if not text:
        return 1.0

    stripped = text.strip()
    if len(stripped) < 20:
        return 1.0
    if len(_SUBSTANTIVE_CHAR_RE.findall(stripped)) < 10:
        return 1.0

    # Boilerplate stock phrase -> immediate high score
    for p in _BOILERPLATE_ANCHORED:
        if p.search(stripped):
            return 0.95
    for p in _BOILERPLATE_ANYWHERE:
        if p.search(stripped):
            return 0.9

    # Unique-token ratio: "ok ok ok" -> tiny vocab -> clearly low-info. We
    # gate on this aggressively so repetitive text can't hide behind a
    # high substantive-char ratio.
    tokens = [t for t in re.split(r"\s+", stripped) if t]
    if tokens:
        unique_ratio = len(set(tokens)) / max(len(tokens), 1)
        if len(tokens) >= 8 and unique_ratio < 0.20:
            return 0.9
    else:
        unique_ratio = 1.0

    # Substantive-char ratio: "----- ||||" -> lots of separators, low substance
    subs_ratio = len(_SUBSTANTIVE_CHAR_RE.findall(stripped)) / max(len(stripped), 1)

    # Combine: both low → high low_info score.
    score = 1.0 - min(1.0, (unique_ratio + subs_ratio) / 1.5)
    return round(score, 3)


def is_low_info_chunk(text: str, threshold: float = 0.7) -> bool:
    return low_info_score(text) >= threshold
