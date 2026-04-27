"""One canonical normalizer. Replaces three separate paths in the old code."""
from __future__ import annotations

import re
import unicodedata

from ..index.bm25 import CJK_RE


_WHITESPACE_RE = re.compile(r"\s+")


def detect_language(text: str) -> str:
    """Return 'zh' if any CJK char present, else 'en'. Simple but enough for routing."""
    return "zh" if CJK_RE.search(text) else "en"


def normalize_query(query: str) -> str:
    """Apply the same normalization used at index time so matches don't drift."""
    q = unicodedata.normalize("NFKC", query).strip()
    q = _WHITESPACE_RE.sub(" ", q)
    return q
