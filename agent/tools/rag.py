"""The canonical knowledge tool: HTTP-call the RAG service.

By going through HTTP (not a Python import) we:
  * keep agent + RAG independently deployable
  * can swap the RAG implementation without touching agent code
  * let the same RAG serve multiple agents (jd_demo, taobao_demo, ...)
"""
from __future__ import annotations

import asyncio
import os

import httpx


RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://127.0.0.1:8008")
RAG_TIMEOUT = int(os.getenv("RAG_TIMEOUT_SECONDS", "90"))


class RAGError(RuntimeError):
    pass


def _normalize_response(data: dict) -> dict:
    return {
        "answer": data.get("answer", ""),
        "rewritten_query": data.get("rewritten_query") or data.get("effective_query", ""),
        "abstain": bool(data.get("abstain") or data.get("abstained")),
        "abstain_reason": data.get("abstain_reason") or data.get("reason"),
        "citations": data.get("citations", []),
        "hits": data.get("hits") or data.get("retrieved", []),
        "latency_ms": data.get("latency_ms"),
    }


def _build_body(
    query: str, kb_id: str, conversation: list[dict] | None, use_multi_query: bool, filter: dict | None
) -> dict:
    body = {"query": query, "kb_id": kb_id, "use_multi_query": use_multi_query}
    if conversation:
        body["conversation"] = conversation
    if filter:
        body["filter"] = filter
    return body


async def rag_search_async(
    query: str,
    kb_id: str,
    *,
    conversation: list[dict] | None = None,
    use_multi_query: bool = False,
    filter: dict | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict:
    body = _build_body(query, kb_id, conversation, use_multi_query, filter)
    owns = client is None
    client = client or httpx.AsyncClient(timeout=RAG_TIMEOUT)
    try:
        r = await client.post(f"{RAG_BASE_URL}/answer", json=body)
        r.raise_for_status()
        return _normalize_response(r.json())
    except httpx.HTTPStatusError as e:
        raise RAGError(f"RAG HTTP {e.response.status_code}: {e.response.text[:300]}") from e
    except httpx.RequestError as e:
        raise RAGError(f"RAG network error: {e}") from e
    finally:
        if owns:
            await client.aclose()


def rag_search(
    query: str,
    kb_id: str,
    *,
    conversation: list[dict] | None = None,
    use_multi_query: bool = False,
    filter: dict | None = None,
) -> dict:
    """Sync wrapper. LangGraph specialist nodes call this; they run in a
    regular thread so a fresh event loop is fine.
    """
    body = _build_body(query, kb_id, conversation, use_multi_query, filter)
    try:
        with httpx.Client(timeout=RAG_TIMEOUT) as client:
            r = client.post(f"{RAG_BASE_URL}/answer", json=body)
            r.raise_for_status()
            return _normalize_response(r.json())
    except httpx.HTTPStatusError as e:
        raise RAGError(f"RAG HTTP {e.response.status_code}: {e.response.text[:300]}") from e
    except httpx.RequestError as e:
        raise RAGError(f"RAG network error: {e}") from e
