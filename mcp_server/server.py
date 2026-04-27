"""MCP server: exposes this project's RAG + order + return tools, its
knowledge bases as resources, and its canonical system prompts.

Transport
---------
stdio -- the standard transport for Claude Desktop / Cursor. Run via:

    python -m mcp_server.server

To wire into Claude Desktop, add to its config:

    {
      "mcpServers": {
        "enterprise-rag": {
          "command": "python",
          "args": ["-m", "mcp_server.server"],
          "cwd": "/Users/neo/enterprise-multimodal-copilot",
          "env": {
            "PYTHONPATH": "src:.",
            "DASHSCOPE_API_KEY": "..."
          }
        }
      }
    }

What MCP primitives we expose
-----------------------------
Tools (LLM-callable functions):
    - rag_search        : query any KB, get grounded answer + citations
    - get_order         : DB lookup by order_id
    - list_user_orders  : most-recent N for a user
    - track_package     : mock shipping timeline
    - check_return_eligibility
    - create_return_request
    - similar_products  : BGE-M3 product similarity

Resources (read-only data the model can reference):
    - kb://{kb_id}                     -- manifest + chunk count for a KB
    - order://{order_id}               -- a single order as JSON

Prompts (reusable templates, used via mcp-client's prompt picker):
    - customer_service_jd              -- JD persona
    - customer_service_taobao          -- Taobao persona
    - refund_eligibility_check         -- structured refund eligibility prompt

Design choices
--------------
- Tool implementations are thin wrappers around the same Python functions
  the agent uses, so MCP clients get the same behaviour as our LangGraph
  specialists.
- All tool I/O is JSON-typed via jsonschema in the ``inputSchema`` -- Claude
  Desktop uses this to validate calls before dispatching.
- Read-only resources for KB metadata + order details so a Claude
  Desktop user can ``@order://JD20260420456`` in a conversation.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any


# Make `rag` and `agent` importable when run as `python -m mcp_server.server`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


import mcp.server.stdio
import mcp.types as mcp_types
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions


# ---- imports from our own packages ----
from agent.persona import PERSONAS
from agent.tools.orders import get_order, list_user_orders, track_package
from agent.tools.recommend import similar_products
from agent.tools.returns import check_eligibility, create_return_request
from rag import knowledge_base as kb_mod
from rag.pipeline import answer_query


SERVER_NAME = "enterprise-rag"
SERVER_VERSION = "0.1.0"


server: Server = Server(SERVER_NAME)


# =================== Tools ===================

_TOOL_SCHEMAS: list[mcp_types.Tool] = [
    mcp_types.Tool(
        name="rag_search",
        description=(
            "Retrieve from a knowledge base and return a grounded answer with citations. "
            "Use for product info, policies (returns / shipping / membership), release notes, "
            "or any free-form question that should be answered from documents."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "natural-language question"},
                "kb_id": {
                    "type": "string",
                    "description": "knowledge base id",
                    "enum": ["jd_demo", "taobao_demo", "airbyte_demo", "cs4303"],
                },
                "conversation": {
                    "type": "array",
                    "description": "optional prior turns: [{role:'user'|'assistant', content:string}]",
                    "items": {"type": "object"},
                },
            },
            "required": ["query", "kb_id"],
        },
    ),
    mcp_types.Tool(
        name="get_order",
        description="Fetch an order by its ID. Returns null if not found.",
        inputSchema={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    ),
    mcp_types.Tool(
        name="list_user_orders",
        description="List most recent N orders for a user in a tenant.",
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "tenant": {"type": "string", "enum": ["jd", "taobao"]},
                "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
            },
            "required": ["user_id", "tenant"],
        },
    ),
    mcp_types.Tool(
        name="track_package",
        description="Look up a shipment timeline by tracking number.",
        inputSchema={
            "type": "object",
            "properties": {"tracking_no": {"type": "string"}},
            "required": ["tracking_no"],
        },
    ),
    mcp_types.Tool(
        name="check_return_eligibility",
        description="Return eligibility + refund_cents or a reason code if not eligible.",
        inputSchema={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    ),
    mcp_types.Tool(
        name="create_return_request",
        description="Persist a pending return/refund/exchange request against an order.",
        inputSchema={
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "kind": {"type": "string", "enum": ["refund", "return", "exchange", "price_protect"]},
                "reason": {"type": "string"},
            },
            "required": ["order_id", "kind", "reason"],
        },
    ),
    mcp_types.Tool(
        name="similar_products",
        description="Find products similar to a given text or SKU using BGE-M3 embeddings.",
        inputSchema={
            "type": "object",
            "properties": {
                "query_text": {"type": "string"},
                "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                "exclude_sku": {"type": "string"},
            },
            "required": ["query_text"],
        },
    ),
]


@server.list_tools()  # type: ignore[no-redef]
async def _list_tools() -> list[mcp_types.Tool]:
    return _TOOL_SCHEMAS


def _tool_result(payload: Any) -> list[mcp_types.TextContent]:
    return [mcp_types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, default=str))]


@server.call_tool()  # type: ignore[no-redef]
async def _call_tool(name: str, arguments: dict[str, Any]) -> list[mcp_types.TextContent]:
    if name == "rag_search":
        ans = answer_query(
            query=arguments["query"],
            kb_id=arguments["kb_id"],
            conversation=arguments.get("conversation"),
        )
        return _tool_result({
            "answer": ans.text,
            "abstained": ans.abstained,
            "citations": [
                {"n": c.n, "title": c.title, "source_id": c.source_id, "snippet": c.snippet}
                for c in ans.citations
            ],
            "latency_ms": ans.latency_ms,
        })

    if name == "get_order":
        return _tool_result(get_order(arguments["order_id"]))

    if name == "list_user_orders":
        return _tool_result(
            list_user_orders(
                user_id=arguments["user_id"],
                tenant=arguments["tenant"],
                limit=arguments.get("limit", 10),
            )
        )

    if name == "track_package":
        return _tool_result(track_package(arguments["tracking_no"]))

    if name == "check_return_eligibility":
        return _tool_result(check_eligibility(arguments["order_id"]))

    if name == "create_return_request":
        return _tool_result(
            create_return_request(
                order_id=arguments["order_id"],
                kind=arguments["kind"],
                reason=arguments["reason"],
            )
        )

    if name == "similar_products":
        return _tool_result(
            similar_products(
                query_text=arguments["query_text"],
                top_k=arguments.get("top_k", 5),
                exclude_sku=arguments.get("exclude_sku"),
            )
        )

    return [mcp_types.TextContent(type="text", text=json.dumps({"error": f"unknown tool: {name}"}))]


# =================== Resources ===================

@server.list_resources()  # type: ignore[no-redef]
async def _list_resources() -> list[mcp_types.Resource]:
    resources: list[mcp_types.Resource] = []
    for kb in kb_mod.list_kbs():
        resources.append(
            mcp_types.Resource(
                uri=f"kb://{kb.kb_id}",
                name=f"Knowledge base: {kb.kb_id}",
                description=f"Manifest + size for {kb.kb_id} ({kb.chunk_count} chunks)",
                mimeType="application/json",
            )
        )
    return resources


@server.read_resource()  # type: ignore[no-redef]
async def _read_resource(uri: str) -> str:
    if uri.startswith("kb://"):
        kb_id = uri[len("kb://"):]
        try:
            kb = kb_mod.get_kb(kb_id)
        except FileNotFoundError as e:
            return json.dumps({"error": str(e)})
        return json.dumps(
            {
                "kb_id": kb.kb_id,
                "chunk_count": kb.chunk_count,
                "description": kb.description,
                "is_built": kb.is_built(),
                "root": str(kb.root),
            },
            ensure_ascii=False,
            indent=2,
        )

    if uri.startswith("order://"):
        order_id = uri[len("order://"):]
        order = get_order(order_id)
        return json.dumps(order or {"error": "not found", "order_id": order_id}, ensure_ascii=False, indent=2)

    return json.dumps({"error": f"unknown resource: {uri}"})


# =================== Prompts ===================

_PROMPTS: dict[str, mcp_types.Prompt] = {
    "customer_service_jd": mcp_types.Prompt(
        name="customer_service_jd",
        description="JD 客服助手 人设 system prompt",
        arguments=[],
    ),
    "customer_service_taobao": mcp_types.Prompt(
        name="customer_service_taobao",
        description="淘宝/天猫客服助手 人设 system prompt",
        arguments=[],
    ),
    "refund_eligibility_check": mcp_types.Prompt(
        name="refund_eligibility_check",
        description="Given an order_id, produce a structured refund-eligibility judgement.",
        arguments=[
            mcp_types.PromptArgument(name="order_id", description="Order ID to evaluate", required=True),
        ],
    ),
}


@server.list_prompts()  # type: ignore[no-redef]
async def _list_prompts() -> list[mcp_types.Prompt]:
    return list(_PROMPTS.values())


@server.get_prompt()  # type: ignore[no-redef]
async def _get_prompt(name: str, arguments: dict[str, str] | None) -> mcp_types.GetPromptResult:
    if name == "customer_service_jd":
        return mcp_types.GetPromptResult(
            description="JD 客服 system prompt",
            messages=[
                mcp_types.PromptMessage(
                    role="assistant",
                    content=mcp_types.TextContent(type="text", text=PERSONAS["jd"]["system_prompt"]),
                )
            ],
        )
    if name == "customer_service_taobao":
        return mcp_types.GetPromptResult(
            description="淘宝小蜜 system prompt",
            messages=[
                mcp_types.PromptMessage(
                    role="assistant",
                    content=mcp_types.TextContent(type="text", text=PERSONAS["taobao"]["system_prompt"]),
                )
            ],
        )
    if name == "refund_eligibility_check":
        order_id = (arguments or {}).get("order_id", "")
        user_msg = (
            f"Call `check_return_eligibility` for order `{order_id}`. If eligible, "
            "follow up by calling `create_return_request` after confirming the reason "
            "with the user. Output a concise summary of eligibility + next steps."
        )
        return mcp_types.GetPromptResult(
            description=f"Refund eligibility flow for {order_id}",
            messages=[
                mcp_types.PromptMessage(
                    role="user",
                    content=mcp_types.TextContent(type="text", text=user_msg),
                )
            ],
        )
    raise ValueError(f"unknown prompt: {name}")


# =================== Entry point ===================

async def _main() -> None:
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=SERVER_NAME,
                server_version=SERVER_VERSION,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(_main())
