"""Compatibility shims for the original React frontend.

The old frontend was built against a much bigger admin surface. For the Q&A
view we only need the endpoints that chat.tsx actually calls:

    GET  /profiles
    GET  /platform/knowledge-spaces
    GET  /platform/knowledge-spaces/{id}/rebuild-plan
    GET  /platform/knowledge-spaces/{id}/documents
    POST /answer   (old request + response shape)

Everything else (connectors, sync jobs, OCR beta, feedback) is not implemented;
the 知识库 and 评测 tabs will be empty. That is intentional -- we rebuilt the
pipeline, not the whole platform.
"""
from __future__ import annotations

import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from rag import knowledge_base as kb_mod
from rag.ingest.pipeline import read_chunks_jsonl
from rag.pipeline import answer_query, answer_query_async

from .auth import AuthContext, require_api_key


router = APIRouter()


def _trace_id() -> str:
    return uuid.uuid4().hex


def _kb_to_space(kb) -> dict[str, Any]:
    """Map new KnowledgeBase -> old KnowledgeSpace shape."""
    return {
        "id": kb.kb_id,
        "display_name": kb.kb_id,
        "tenant_id": "default",
        "description": kb.description or f"Knowledge base: {kb.kb_id}",
        "storage_root": str(kb.root),
        "source_families": ["doc"],
        "supported_connector_types": [],
        "default_search_profiles": {"default": f"{kb.kb_id}-hybrid-rerank"},
        "default_answer_mode": "extractive_grounded_v1",
        "available_answer_modes": ["extractive_grounded_v1"],
    }


def _empty_platform_list(resource: str) -> dict[str, Any]:
    return {"trace_id": _trace_id(), "resource_type": resource, "count": 0, "items": []}


@router.get("/platform/connectors")
def connectors():
    return _empty_platform_list("connector")


@router.get("/platform/sources")
def sources():
    return _empty_platform_list("managed_source")


@router.get("/platform/sync-jobs")
def sync_jobs():
    return _empty_platform_list("sync_job")


@router.get("/platform/knowledge-spaces/{kb_id}/ocr-upload-entry-policy")
def ocr_entry_policy(kb_id: str):
    return {
        "trace_id": _trace_id(),
        "knowledge_space_id": kb_id,
        "allowed": False,
        "reason": "ocr_not_implemented",
        "message": "OCR upload is not enabled in this build.",
    }


@router.get("/profiles")
def profiles():
    return {
        "trace_id": _trace_id(),
        "count": 1,
        "profiles": [
            {
                "name": "hybrid-rerank",
                "retrieval_mode": "bm25+vector+rerank",
                "description": "BM25 + BGE-M3 vectors fused via RRF, cross-encoder reranked.",
                "summary": "default hybrid pipeline",
                "source_families": ["doc"],
            }
        ],
    }


@router.get("/platform/knowledge-spaces")
def list_knowledge_spaces():
    kbs = kb_mod.list_kbs()
    items = [_kb_to_space(kb) for kb in kbs]
    return {
        "trace_id": _trace_id(),
        "resource_type": "knowledge_space",
        "count": len(items),
        "items": items,
    }


@router.get("/platform/knowledge-spaces/{kb_id}/rebuild-plan")
def rebuild_plan(kb_id: str):
    try:
        kb = kb_mod.get_kb(kb_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "trace_id": _trace_id(),
        "resource_type": "knowledge_space_rebuild_plan",
        "knowledge_space_id": kb_id,
        "display_name": kb_id,
        "records_path": str(kb.chunks_path),
        "storage_root": str(kb.root),
        "artifacts": [
            {"name": "chunks.jsonl", "path": str(kb.chunks_path), "exists": kb.chunks_path.exists(), "required": True, "size_bytes": kb.chunks_path.stat().st_size if kb.chunks_path.exists() else None},
            {"name": "bm25.sqlite", "path": str(kb.root / "bm25.sqlite"), "exists": (kb.root / "bm25.sqlite").exists(), "required": True, "size_bytes": (kb.root / "bm25.sqlite").stat().st_size if (kb.root / "bm25.sqlite").exists() else None},
            {"name": "vector.faiss", "path": str(kb.root / "vector.faiss"), "exists": (kb.root / "vector.faiss").exists(), "required": True, "size_bytes": (kb.root / "vector.faiss").stat().st_size if (kb.root / "vector.faiss").exists() else None},
        ],
        "readiness": {
            "records_ready": kb.chunks_path.exists(),
            "chunks_ready": kb.chunks_path.exists(),
            "lexical_ready": (kb.root / "bm25.sqlite").exists(),
            "embedding_ready": (kb.root / "vector.faiss").exists(),
            "hybrid_ready": kb.is_built(),
            "serving_ready": kb.is_built(),
        },
        "commands": {
            "full_rebuild": f"python scripts_new/build_kb.py {kb_id}",
            "lexical_rebuild": f"python scripts_new/build_kb.py {kb_id}",
        },
        "notes": [],
    }


@router.get("/platform/knowledge-spaces/{kb_id}/documents")
def list_documents(kb_id: str):
    try:
        kb = kb_mod.get_kb(kb_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not kb.chunks_path.exists():
        return {
            "trace_id": _trace_id(),
            "resource_type": "knowledge_space_documents",
            "knowledge_space_id": kb_id,
            "records_path": str(kb.chunks_path),
            "count": 0,
            "source_type_counts": {},
            "items": [],
            "plan": rebuild_plan(kb_id),
        }

    chunks = read_chunks_jsonl(kb.chunks_path)
    by_source: dict[str, dict[str, Any]] = {}
    for c in chunks:
        doc = by_source.get(c.source_id)
        if doc is None:
            file_name = c.source_path.rsplit("/", 1)[-1] if c.source_path else c.source_id
            by_source[c.source_id] = {
                "source_id": c.source_id,
                "title": c.title,
                "file_name": file_name,
                "source_type": "doc",
                "language": None,
                "canonical_url": c.source_path,
                "section_count": 1,
                "char_count": len(c.text),
                "content_scope": None,
                "uploaded_at": None,
                "status": "ready",
            }
        else:
            doc["section_count"] += 1
            doc["char_count"] += len(c.text)

    items = list(by_source.values())
    return {
        "trace_id": _trace_id(),
        "resource_type": "knowledge_space_documents",
        "knowledge_space_id": kb_id,
        "records_path": str(kb.chunks_path),
        "count": len(items),
        "source_type_counts": dict(Counter(["doc"] * len(items))),
        "items": items,
        "plan": rebuild_plan(kb_id),
    }


class LegacyAnswerRequest(BaseModel):
    query: str
    knowledge_space_id: str | None = None
    kb_id: str | None = None
    conversation_id: str | None = None
    conversation_history: list[dict] | None = None
    conversation: list[dict] | None = None  # also accept new-shape key
    allow_fallback: bool = True
    profile: str | None = None
    answer_mode: str | None = None


def _normalize_conversation(req: "LegacyAnswerRequest") -> list[dict] | None:
    """Accept either key, also tolerate frontend's {"role","content"} shape."""
    raw = req.conversation or req.conversation_history
    if not raw:
        return None
    out: list[dict] = []
    for turn in raw:
        role = turn.get("role") or ("assistant" if turn.get("is_assistant") else "user")
        content = turn.get("content") or turn.get("text") or ""
        if not content:
            continue
        out.append({"role": role, "content": content})
    return out or None


@router.post("/answer")
async def legacy_answer(
    req: LegacyAnswerRequest,
    auth: AuthContext = Depends(require_api_key),
):
    kb_id = req.kb_id or req.knowledge_space_id
    if not kb_id:
        raise HTTPException(status_code=400, detail="kb_id or knowledge_space_id required")

    conversation = _normalize_conversation(req)

    try:
        ans = await answer_query_async(
            req.query, kb_id, conversation=conversation, use_multi_query=False
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    trace_id = _trace_id()
    served_at = datetime.now(timezone.utc).isoformat()

    citations = []
    for c in ans.citations:
        citations.append(
            {
                "rank": c.n,
                "n": c.n,
                "source_id": c.source_id,
                "source_type": "doc",
                "authority": None,
                "title": c.title,
                "section_title": " / ".join(c.section_path) if c.section_path else None,
                "canonical_url": c.source_path,
                "snippet": c.snippet,
                "score": ans.hits[c.n - 1].score if c.n - 1 < len(ans.hits) else None,
            }
        )

    retrieved = []
    for h in ans.hits:
        retrieved.append(
            {
                "chunk_id": h.chunk_id,
                "source_id": h.source_id,
                "source_type": "doc",
                "title": h.title,
                "section_title": " / ".join(h.section_path) if h.section_path else None,
                "text": h.text,
                "canonical_url": h.source_path,
                "score": h.score,
            }
        )

    top_score = ans.hits[0].score if ans.hits else None

    if ans.abstained:
        response_mode = "not_found_in_docs"
        response_mode_label = "资料不足"
        response_mode_message = "当前知识库内没有足够的证据回答这个问题。"
        grounding_status = "insufficient_evidence"
    else:
        response_mode = "docs_grounded"
        response_mode_label = "来自知识库"
        response_mode_message = "答案基于检索到的资料生成,末尾方括号为引用编号。"
        grounding_status = "grounded"

    return {
        "trace_id": trace_id,
        "conversation_id": req.conversation_id,
        "knowledge_space_id": kb_id,
        "query": req.query,
        "profile_name": f"{kb_id}-hybrid-rerank",
        "answer_mode": "extractive_grounded_v1",
        "requested_answer_mode": req.answer_mode or "extractive_grounded_v1",
        "resolved_answer_mode": "extractive_grounded_v1",
        "fallback_from_answer_mode": None,
        "fallback_reason": None,
        "route": {
            "name": "default",
            "profile_name": f"{kb_id}-hybrid-rerank",
            "rationale": "Unified hybrid pipeline: BM25 + vector + cross-encoder rerank.",
            "complexity": "moderate",
        },
        "answer": ans.text,
        "abstain": ans.abstained,
        "abstain_reason": ans.reason if ans.abstained else None,
        "response_mode": response_mode,
        "response_mode_label": response_mode_label,
        "response_mode_message": response_mode_message,
        "grounding_status": grounding_status,
        "general_knowledge_recommended": False,
        "observed_top_score": top_score,
        "threshold": 0.15,
        "citations": citations,
        "risk_flags": [],
        "retrieved": retrieved,
        "served_at": served_at,
        "effective_query": ans.rewritten_query,
        # Also expose the new-shape fields so the built-in UI still works.
        "rewritten_query": ans.rewritten_query,
        "hits": retrieved,
        "latency_ms": ans.latency_ms,
    }
