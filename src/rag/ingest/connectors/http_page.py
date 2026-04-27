"""HTTP page connector.

Given a fixed list of URLs, fetch + hash each one. Suitable for an ops
operator who curates a list of pages to sync -- enterprise wikis,
release notes pages, vendor docs.

Deliberately NOT a crawler: we don't follow links. Crawling needs
robots.txt, rate limits, politeness windows, and a frontier queue, all
of which belong in a separate service.
"""
from __future__ import annotations

from typing import Iterable, Sequence
from urllib.parse import urlparse

import httpx

from .base import Connector, SourceDoc, stable_source_id


class HttpPageConnector(Connector):
    name = "http"

    def __init__(
        self,
        kb_id: str,
        urls: Sequence[str],
        *,
        user_agent: str = "rag-connector/0.1 (+https://example.invalid)",
        timeout_s: int = 30,
    ):
        self.kb_id = kb_id
        self.urls = list(urls)
        self.user_agent = user_agent
        self.timeout_s = timeout_s

    def describe(self) -> dict:
        return {"connector": self.name, "url_count": len(self.urls)}

    def list_documents(self) -> Iterable[SourceDoc]:
        for url in self.urls:
            parsed = urlparse(url)
            # Title is a last-leg hint; real title comes from the HTML loader later.
            title_hint = parsed.path.rstrip("/").rsplit("/", 1)[-1] or parsed.netloc
            yield SourceDoc(
                source_id=stable_source_id(self.kb_id, url),
                uri=url,
                suffix=".html",
                title=title_hint,
                cheap_hash=None,  # HTTP HEAD could give ETag/Last-Modified, TBD
                extra={"netloc": parsed.netloc},
            )

    def fetch_bytes(self, doc: SourceDoc) -> bytes:
        with httpx.Client(headers={"User-Agent": self.user_agent}, timeout=self.timeout_s, follow_redirects=True) as c:
            r = c.get(doc.uri)
            r.raise_for_status()
            return r.content
