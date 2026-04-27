"""Notion connector.

Talks to the Notion REST API directly via ``httpx`` -- no SDK dependency.
Uses a Notion internal integration token (start with ``secret_...`` or
``ntn_...``). Share the target database or pages with the integration in
the Notion UI first; the token alone doesn't grant access.

Flow:
    1. ``list_documents()`` searches every page the integration can see
       (or enumerates a fixed database if ``database_id`` is set) and
       emits one ``SourceDoc`` per Notion page.
    2. ``fetch_bytes(doc)`` pulls that page's blocks, renders to markdown,
       and returns UTF-8 bytes. The ``.md`` suffix routes it through the
       existing markdown loader downstream, so all PH2 cleaning passes
       apply unchanged.

Notes on coverage:
    - Supports the common block types (headings, paragraphs, lists, code,
      quotes, to-do, callouts, dividers). Unknown block types render to an
      HTML comment so nothing is silently dropped but nothing poisons the
      chunker either.
    - Pagination is handled for the top-level ``blocks/{page}/children``
      call. Deeply-nested children (lists of lists) are walked up to 3
      levels; deeper nesting is flattened.
    - ``cheap_hash`` uses Notion's ``last_edited_time`` so unchanged pages
      short-circuit fetch in the incremental ingest diff.

Config:
    NotionConnector(
        token="secret_...",
        database_id="abc123..." | None,   # if None, uses /search (all shared pages)
        kb_id="my_kb",
    )
"""
from __future__ import annotations

from typing import Iterable, Iterator

import httpx

from ...logging import get_logger
from .base import Connector, SourceDoc, stable_source_id


log = get_logger(__name__)

NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"  # pin so Notion can't break us silently


class NotionConnector(Connector):
    name = "notion"

    def __init__(
        self,
        *,
        token: str,
        kb_id: str,
        database_id: str | None = None,
        page_size: int = 50,
        timeout: float = 20.0,
    ):
        if not token:
            raise ValueError("NotionConnector requires a non-empty token")
        self._token = token
        self._kb_id = kb_id
        self._database_id = database_id
        self._page_size = page_size
        self._timeout = timeout

    # ---------- http helpers ----------

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }

    def _post(self, path: str, body: dict) -> dict:
        with httpx.Client(timeout=self._timeout) as c:
            r = c.post(f"{NOTION_API}{path}", json=body, headers=self._headers())
            r.raise_for_status()
            return r.json()

    def _get(self, path: str, params: dict | None = None) -> dict:
        with httpx.Client(timeout=self._timeout) as c:
            r = c.get(f"{NOTION_API}{path}", params=params or {}, headers=self._headers())
            r.raise_for_status()
            return r.json()

    # ---------- list ----------

    def list_documents(self) -> Iterable[SourceDoc]:
        if self._database_id:
            yield from self._list_from_database()
        else:
            yield from self._list_from_search()

    def _list_from_database(self) -> Iterator[SourceDoc]:
        cursor: str | None = None
        while True:
            body: dict = {"page_size": self._page_size}
            if cursor:
                body["start_cursor"] = cursor
            data = self._post(f"/databases/{self._database_id}/query", body)
            for page in data.get("results") or []:
                yield self._page_to_source_doc(page)
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
            if not cursor:
                break

    def _list_from_search(self) -> Iterator[SourceDoc]:
        cursor: str | None = None
        while True:
            body: dict = {
                "filter": {"value": "page", "property": "object"},
                "page_size": self._page_size,
            }
            if cursor:
                body["start_cursor"] = cursor
            data = self._post("/search", body)
            for page in data.get("results") or []:
                yield self._page_to_source_doc(page)
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
            if not cursor:
                break

    def _page_to_source_doc(self, page: dict) -> SourceDoc:
        page_id = page.get("id", "")
        last_edited = page.get("last_edited_time", "") or ""
        title = _extract_page_title(page)
        source_id = stable_source_id(self._kb_id, f"notion::{page_id}")
        return SourceDoc(
            source_id=source_id,
            uri=f"notion://page/{page_id}",
            suffix=".md",
            title=title,
            # last_edited_time is Notion's per-page version token -- perfect
            # cheap_hash: unchanged edit time means no need to refetch.
            cheap_hash=last_edited,
            extra={"notion_page_id": page_id, "last_edited_time": last_edited},
        )

    # ---------- fetch ----------

    def fetch_bytes(self, doc: SourceDoc) -> bytes:
        page_id = doc.extra.get("notion_page_id") or doc.uri.rsplit("/", 1)[-1]
        md = self._render_page_markdown(page_id, doc.title)
        return md.encode("utf-8")

    def _render_page_markdown(self, page_id: str, title: str | None) -> str:
        blocks = list(self._list_blocks(page_id))
        body = _blocks_to_markdown(blocks, fetch_children=self._list_blocks, depth=0)
        header = f"# {title}\n\n" if title else ""
        return header + body

    def _list_blocks(self, block_id: str) -> Iterator[dict]:
        cursor: str | None = None
        while True:
            params = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            data = self._get(f"/blocks/{block_id}/children", params=params)
            for b in data.get("results") or []:
                yield b
            if not data.get("has_more"):
                return
            cursor = data.get("next_cursor")
            if not cursor:
                return


# ---------- rendering (pure functions, easy to unit-test) ----------

def _rich_text(rts: list[dict] | None) -> str:
    """Notion rich_text array -> plain markdown (bold / italic / code inline)."""
    out: list[str] = []
    for rt in rts or []:
        plain = rt.get("plain_text") or ""
        if not plain:
            continue
        ann = rt.get("annotations") or {}
        if ann.get("code"):
            plain = f"`{plain}`"
        if ann.get("bold"):
            plain = f"**{plain}**"
        if ann.get("italic"):
            plain = f"*{plain}*"
        if ann.get("strikethrough"):
            plain = f"~~{plain}~~"
        link = (rt.get("text") or {}).get("link")
        if link and link.get("url"):
            plain = f"[{plain}]({link['url']})"
        out.append(plain)
    return "".join(out)


def _extract_page_title(page: dict) -> str | None:
    # Database-backed pages put title in ``properties``; child pages use ``properties.title``.
    props = page.get("properties") or {}
    for key, val in props.items():
        if (val or {}).get("type") == "title":
            return _rich_text(val.get("title")) or None
    # Search result fallback
    title_field = (page.get("icon") or {}).get("name")
    return title_field


def _blocks_to_markdown(blocks: list[dict], *, fetch_children=None, depth: int = 0) -> str:
    """Walk blocks top-to-bottom, rendering markdown. Recurses up to depth 3."""
    out: list[str] = []
    indent = "  " * depth
    for b in blocks:
        btype = b.get("type") or ""
        data = b.get(btype) or {}
        text = _rich_text(data.get("rich_text"))

        if btype == "heading_1":
            out.append(f"# {text}")
        elif btype == "heading_2":
            out.append(f"## {text}")
        elif btype == "heading_3":
            out.append(f"### {text}")
        elif btype == "paragraph":
            if text:
                out.append(text)
        elif btype == "bulleted_list_item":
            out.append(f"{indent}- {text}")
        elif btype == "numbered_list_item":
            out.append(f"{indent}1. {text}")
        elif btype == "to_do":
            mark = "x" if data.get("checked") else " "
            out.append(f"{indent}- [{mark}] {text}")
        elif btype == "quote":
            out.append(f"> {text}")
        elif btype == "callout":
            out.append(f"> {text}")
        elif btype == "code":
            lang = data.get("language") or ""
            out.append(f"```{lang}\n{text}\n```")
        elif btype == "divider":
            out.append("---")
        elif btype == "toggle":
            if text:
                out.append(text)
        elif btype == "image":
            url = _notion_file_url(data)
            caption = _rich_text(data.get("caption"))
            out.append(f"![{caption}]({url})" if url else f"*[Image]* {caption}")
        elif btype == "bookmark":
            url = data.get("url") or ""
            caption = _rich_text(data.get("caption"))
            out.append(f"[{caption or url}]({url})")
        elif btype == "child_page":
            out.append(f"### {data.get('title') or 'Untitled page'}")
        else:
            # Unknown block type -- leave a breadcrumb in markdown so downstream
            # tooling can investigate, but keep the body renderable.
            out.append(f"<!-- unsupported block: {btype} -->")

        if b.get("has_children") and fetch_children is not None and depth < 3:
            try:
                child_blocks = list(fetch_children(b.get("id") or ""))
                child_md = _blocks_to_markdown(child_blocks, fetch_children=fetch_children, depth=depth + 1)
                if child_md:
                    out.append(child_md)
            except Exception as e:  # noqa: BLE001 -- don't fail the whole doc on one sub-fetch
                log.warning("notion_children_fetch_failed", block_id=b.get("id"), error=str(e))

    return "\n\n".join(o for o in out if o)


def _notion_file_url(data: dict) -> str | None:
    f = data.get("external") or data.get("file")
    if not f:
        return None
    return f.get("url")
