"""HTML boilerplate stripping.

Goal: when we ingest a web page, the nav bars, sidebars, cookie banners
and footer are NOT indexed as retrievable text. readability-lxml does the
heavy lifting (it's the same algorithm Firefox Reader View uses). We then
do a light BeautifulSoup pass to drop `<script>`, `<style>`, and empty
lines, since readability keeps them when they're inside the "main" block.

Non-HTML input passes through unchanged -- cleaner is a no-op for MD / TXT
where there's no boilerplate concept.
"""
from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def strip_html_boilerplate(raw_html: str) -> tuple[str, str]:
    """Return (title, main_text) from a messy HTML blob.

    If readability can't extract a main block (e.g. input isn't HTML or is
    too short), falls back to a naive tag-strip so caller always gets text.
    """
    if not raw_html or "<" not in raw_html:
        return "", _collapse_whitespace(raw_html or "")

    try:
        from readability import Document
    except ImportError:
        return _naive_strip(raw_html)

    try:
        doc = Document(raw_html)
        title = (doc.short_title() or "").strip()
        main_html = doc.summary(html_partial=True)
    except Exception:
        return _naive_strip(raw_html)

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(main_html, "lxml")
        # Drop anything that isn't substantive content.
        for tag in soup(["script", "style", "noscript", "svg", "iframe", "form", "nav", "aside", "footer"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
    except Exception:
        text = re.sub(r"<[^>]+>", " ", main_html)

    return title, _collapse_whitespace(text)


def _naive_strip(raw_html: str) -> tuple[str, str]:
    import re as _re
    text = _re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw_html, flags=_re.S | _re.I)
    text = _re.sub(r"<[^>]+>", " ", text)
    return "", _collapse_whitespace(text)


def _collapse_whitespace(text: str) -> str:
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip()
