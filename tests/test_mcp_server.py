"""MCP server surface tests. We exercise the list/call handlers directly
(bypassing stdio) to avoid needing a full client connection.
"""
from __future__ import annotations

import asyncio
import json

import pytest


@pytest.fixture()
def seeded(seeded_db):
    # recommend tool needs catalogue; refresh after seed.
    from agent.tools.recommend import refresh_catalogue
    refresh_catalogue()
    return seeded_db


def test_tool_list_has_expected_names(seeded):
    from mcp_server.server import _list_tools
    tools = asyncio.run(_list_tools())
    names = {t.name for t in tools}
    assert {
        "rag_search", "get_order", "list_user_orders", "track_package",
        "check_return_eligibility", "create_return_request", "similar_products",
    } <= names


def test_get_order_tool(seeded):
    from mcp_server.server import _call_tool
    out = asyncio.run(_call_tool("get_order", {"order_id": "JD20260418123"}))
    data = json.loads(out[0].text)
    assert data is not None
    assert data["id"] == "JD20260418123"
    assert data["tenant"] == "jd"


def test_list_user_orders_tool(seeded):
    from mcp_server.server import _call_tool
    out = asyncio.run(_call_tool("list_user_orders", {
        "user_id": "jd-demo-user", "tenant": "jd", "limit": 5,
    }))
    data = json.loads(out[0].text)
    assert isinstance(data, list)
    assert len(data) == 3


def test_check_return_eligibility_tool(seeded):
    from mcp_server.server import _call_tool
    out = asyncio.run(_call_tool("check_return_eligibility", {"order_id": "JD20260418123"}))
    data = json.loads(out[0].text)
    assert data["ok"] is True
    assert data["refund_cents"] > 0


def test_similar_products_tool(seeded):
    from mcp_server.server import _call_tool
    out = asyncio.run(_call_tool("similar_products", {"query_text": "MacBook Pro", "top_k": 3}))
    data = json.loads(out[0].text)
    assert isinstance(data, list)
    assert len(data) > 0


def test_unknown_tool_returns_error(seeded):
    from mcp_server.server import _call_tool
    out = asyncio.run(_call_tool("banana_tool", {}))
    data = json.loads(out[0].text)
    assert "error" in data


def test_list_resources_includes_kbs(seeded):
    from mcp_server.server import _list_resources
    resources = asyncio.run(_list_resources())
    uris = {str(r.uri) for r in resources}
    # At least one kb:// resource should show up if there's a built KB.
    assert any(u.startswith("kb://") for u in uris), f"got URIs: {uris}"


def test_read_order_resource(seeded):
    from mcp_server.server import _read_resource
    raw = asyncio.run(_read_resource("order://JD20260418123"))
    data = json.loads(raw)
    assert data.get("id") == "JD20260418123"


def test_list_prompts(seeded):
    from mcp_server.server import _list_prompts
    prompts = asyncio.run(_list_prompts())
    names = {p.name for p in prompts}
    assert {"customer_service_jd", "customer_service_taobao", "refund_eligibility_check"} <= names


def test_get_prompt_persona(seeded):
    from mcp_server.server import _get_prompt
    r = asyncio.run(_get_prompt("customer_service_jd", None))
    assert r.description
    assert r.messages
    assert "京东" in r.messages[0].content.text
