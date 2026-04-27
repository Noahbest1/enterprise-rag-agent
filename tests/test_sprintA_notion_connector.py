"""Sprint A — Notion connector tests.

No real Notion API calls. We monkeypatch ``_post`` / ``_get`` so the
connector works against in-memory fake responses. Rendering helpers
(``_rich_text``, ``_blocks_to_markdown``) are pure and tested directly.
"""
from __future__ import annotations

import pytest

from rag.ingest.connectors.notion import (
    NotionConnector,
    _blocks_to_markdown,
    _extract_page_title,
    _rich_text,
)


# ---------- rich_text ----------

def test_rich_text_plain():
    rts = [{"plain_text": "hello"}]
    assert _rich_text(rts) == "hello"


def test_rich_text_bold_italic_code():
    rts = [
        {"plain_text": "bold", "annotations": {"bold": True}},
        {"plain_text": " and "},
        {"plain_text": "italic", "annotations": {"italic": True}},
        {"plain_text": " plus "},
        {"plain_text": "code", "annotations": {"code": True}},
    ]
    out = _rich_text(rts)
    assert "**bold**" in out
    assert "*italic*" in out
    assert "`code`" in out


def test_rich_text_link():
    rts = [{"plain_text": "click", "text": {"link": {"url": "https://example.com"}}}]
    assert _rich_text(rts) == "[click](https://example.com)"


def test_rich_text_empty():
    assert _rich_text(None) == ""
    assert _rich_text([]) == ""


# ---------- title extraction ----------

def test_extract_title_from_properties():
    page = {
        "properties": {
            "Name": {"type": "title", "title": [{"plain_text": "My Doc"}]}
        }
    }
    assert _extract_page_title(page) == "My Doc"


def test_extract_title_absent():
    assert _extract_page_title({"properties": {}}) is None


# ---------- blocks -> markdown ----------

def _txt(t: str) -> list[dict]:
    return [{"plain_text": t}]


def test_headings_and_paragraph():
    blocks = [
        {"type": "heading_1", "heading_1": {"rich_text": _txt("Top")}},
        {"type": "heading_2", "heading_2": {"rich_text": _txt("Sub")}},
        {"type": "paragraph", "paragraph": {"rich_text": _txt("Body text here.")}},
    ]
    md = _blocks_to_markdown(blocks)
    assert "# Top" in md
    assert "## Sub" in md
    assert "Body text here." in md


def test_lists_and_todo():
    blocks = [
        {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": _txt("a")}},
        {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": _txt("b")}},
        {"type": "numbered_list_item", "numbered_list_item": {"rich_text": _txt("x")}},
        {"type": "to_do", "to_do": {"rich_text": _txt("ship it"), "checked": True}},
    ]
    md = _blocks_to_markdown(blocks)
    assert "- a" in md
    assert "- b" in md
    assert "1. x" in md
    assert "- [x] ship it" in md


def test_code_block():
    blocks = [{"type": "code", "code": {"rich_text": _txt("print('hi')"), "language": "python"}}]
    md = _blocks_to_markdown(blocks)
    assert "```python" in md
    assert "print('hi')" in md


def test_quote_divider_callout():
    blocks = [
        {"type": "quote", "quote": {"rich_text": _txt("be careful")}},
        {"type": "divider", "divider": {}},
        {"type": "callout", "callout": {"rich_text": _txt("note")}},
    ]
    md = _blocks_to_markdown(blocks)
    assert "> be careful" in md
    assert "---" in md
    assert "> note" in md


def test_image_block():
    blocks = [{
        "type": "image",
        "image": {"external": {"url": "https://img.example/x.png"}, "caption": _txt("a fig")},
    }]
    md = _blocks_to_markdown(blocks)
    assert "https://img.example/x.png" in md
    assert "a fig" in md


def test_unknown_block_leaves_breadcrumb():
    blocks = [{"type": "audio", "audio": {}}]
    md = _blocks_to_markdown(blocks)
    assert "<!-- unsupported block: audio -->" in md


def test_children_recursion(monkeypatch):
    parent = {
        "id": "p1",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": _txt("outer")},
        "has_children": True,
    }
    def fake_children(_bid: str):
        return [{"type": "bulleted_list_item",
                 "bulleted_list_item": {"rich_text": _txt("inner")}}]
    md = _blocks_to_markdown([parent], fetch_children=fake_children)
    assert "- outer" in md
    # nested child indented
    assert "  - inner" in md


# ---------- connector high-level ----------

def test_connector_requires_token():
    with pytest.raises(ValueError):
        NotionConnector(token="", kb_id="kb")


def test_list_documents_from_database(monkeypatch):
    conn = NotionConnector(token="secret_x", kb_id="kb_notion", database_id="db-123")
    responses = [
        {
            "results": [
                {"id": "page-a", "last_edited_time": "2026-04-23T00:00:00Z",
                 "properties": {"Name": {"type": "title", "title": [{"plain_text": "Page A"}]}}},
            ],
            "has_more": True, "next_cursor": "c1",
        },
        {
            "results": [
                {"id": "page-b", "last_edited_time": "2026-04-23T00:01:00Z",
                 "properties": {"Name": {"type": "title", "title": [{"plain_text": "Page B"}]}}},
            ],
            "has_more": False,
        },
    ]
    calls: list[tuple] = []
    def fake_post(path, body):
        calls.append((path, body))
        return responses.pop(0)
    monkeypatch.setattr(conn, "_post", fake_post)

    docs = list(conn.list_documents())
    assert len(docs) == 2
    assert docs[0].title == "Page A"
    assert docs[1].title == "Page B"
    assert docs[0].cheap_hash == "2026-04-23T00:00:00Z"
    assert docs[0].suffix == ".md"
    assert docs[0].uri.startswith("notion://page/")
    assert docs[0].extra["notion_page_id"] == "page-a"
    # Pagination wired through
    assert len(calls) == 2
    assert calls[1][1].get("start_cursor") == "c1"


def test_list_documents_from_search(monkeypatch):
    conn = NotionConnector(token="secret_x", kb_id="kb")
    monkeypatch.setattr(conn, "_post", lambda path, body: {
        "results": [
            {"id": "p1", "last_edited_time": "t",
             "properties": {"Name": {"type": "title", "title": [{"plain_text": "Only"}]}}},
        ],
        "has_more": False,
    })
    docs = list(conn.list_documents())
    assert len(docs) == 1
    assert docs[0].title == "Only"


def test_fetch_bytes_renders_markdown(monkeypatch):
    conn = NotionConnector(token="secret_x", kb_id="kb")
    # Fake _list_blocks to return two blocks
    def fake_list_blocks(_block_id):
        yield {"type": "heading_1", "heading_1": {"rich_text": _txt("Title")}}
        yield {"type": "paragraph", "paragraph": {"rich_text": _txt("Body.")}}
    monkeypatch.setattr(conn, "_list_blocks", fake_list_blocks)

    doc = type("D", (), {})()
    doc.extra = {"notion_page_id": "p-xyz"}
    doc.uri = "notion://page/p-xyz"
    doc.title = "External Title"

    raw = conn.fetch_bytes(doc)
    md = raw.decode("utf-8")
    assert md.startswith("# External Title")
    assert "# Title" in md
    assert "Body." in md
