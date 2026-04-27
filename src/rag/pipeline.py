"""End-to-end query pipeline: normalize -> rewrite -> retrieve+rerank -> answer."""
from __future__ import annotations

import time

import asyncio

from .answer.compose import build_answer_prompt
from .answer.generate import generate_answer
from .answer.grounding import parse_citations, should_abstain
from .answer.fallback import chat_once_with_fallback
from .answer.llm_client import LLMError, chat_once_async, chat_stream_async
from .answer.meta_answer import (
    answer_chitchat,
    answer_chitchat_sync,
    answer_meta,
    answer_meta_sync,
    stream_chitchat,
    stream_meta,
)
from .config import settings
from .index.faiss_store import embed_query_cached
from .knowledge_base import get_kb
from .logging import get_logger
from .query.intent import classify_intent
from .query.multi_query import expand_queries
from .query.normalize import detect_language, normalize_query
from .query.rewrite import rewrite_query
from .retrieval.hybrid import retrieve, retrieve_async, rrf_fuse
from .retrieval.parent_expand import expand_to_parents
from .retrieval.rerank import rerank_hits
from .security.injection import detect_injection, sanitize_output
from .semantic_cache import get_semantic_cache
from .types import Answer, Citation, Hit

log = get_logger(__name__)


def _answer_to_dict(ans: Answer) -> dict:
    """Serialise an Answer for cache storage."""
    return {
        "query": ans.query,
        "rewritten_query": ans.rewritten_query,
        "text": ans.text,
        "citations": [
            {
                "n": c.n,
                "source_id": c.source_id,
                "source_path": c.source_path,
                "title": c.title,
                "section_path": list(c.section_path or []),
                "snippet": c.snippet,
            }
            for c in ans.citations
        ],
        "hits_preview": [
            {"chunk_id": h.chunk_id, "title": h.title, "score": h.score}
            for h in (ans.hits or [])[:5]
        ],
        "trace": dict(ans.trace or {}),
    }


def _answer_from_dict(d: dict, latency_ms: int) -> Answer:
    citations = [
        Citation(
            n=c["n"],
            source_id=c.get("source_id", ""),
            source_path=c.get("source_path", ""),
            title=c.get("title", ""),
            section_path=c.get("section_path", []),
            snippet=c.get("snippet", ""),
        )
        for c in (d.get("citations") or [])
    ]
    trace = dict(d.get("trace") or {})
    trace["from_semantic_cache"] = True
    return Answer(
        query=d.get("query", ""),
        rewritten_query=d.get("rewritten_query", ""),
        text=d.get("text", ""),
        citations=citations,
        abstained=False,
        reason="",
        hits=[],  # hits aren't reused from cache
        latency_ms=latency_ms,
        trace=trace,
    )


def _retrieve_with_multi_query(
    original_for_rerank: str,
    variants: list[str],
    kb_dir,
    *,
    filter: dict | None,
) -> list[Hit]:
    """Retrieve per variant (without rerank), fuse across variants, then rerank once."""
    per_query_hits: list[list[Hit]] = []
    for q in variants:
        hits = retrieve(
            q,
            kb_dir,
            rerank=False,
            final_top_k=settings.bm25_top_k,
            filter=filter,
        )
        if hits:
            per_query_hits.append(hits)

    if not per_query_hits:
        return []
    # Fuse across variants the same way we fuse BM25 + vector.
    fused = rrf_fuse(per_query_hits, k=settings.rrf_k)
    # One rerank pass against the ORIGINAL (normalized) query so we score
    # relevance to what the user actually asked, not a variant.
    reranked = rerank_hits(original_for_rerank, fused[: max(settings.rerank_top_k, settings.final_top_k)])
    return reranked[: settings.final_top_k]


def answer_query(
    query: str,
    kb_id: str,
    *,
    use_rewrite: bool = True,
    use_rerank: bool = True,
    use_multi_query: bool = False,
    conversation: list[dict] | None = None,
    filter: dict | None = None,
) -> Answer:
    """Run the full RAG pipeline.

    ``conversation`` is an optional list of ``{"role": "user"|"assistant", "content": str}``
    turns ordered oldest-first. When present, the rewriter resolves pronouns in
    the current query against this history so follow-ups like "这个怎么退货"
    become independent search queries.

    ``filter`` is an optional metadata filter: ``source_path_contains``,
    ``title_contains``, ``section_contains``, or ``source_type``.

    ``use_multi_query`` asks the LLM to generate 3-4 alternative search
    queries; each is retrieved independently and results are fused. This
    substantially improves recall for vague queries at the cost of one
    extra LLM call per answer.
    """
    t0 = time.perf_counter()
    kb = get_kb(kb_id)
    if not kb.is_built():
        raise RuntimeError(f"KB '{kb_id}' has no built indexes. Run build_kb.py first.")

    original = normalize_query(query)
    language = detect_language(original)

    # Intent routing: meta-questions ("what did you just translate") and
    # chitchat ("hello") must NOT trigger retrieval, otherwise the LLM
    # hallucinates a "summary" of unrelated KB chunks. Default ON.
    if getattr(settings, "enable_intent_routing", True):
        verdict = classify_intent(original, has_conversation=bool(conversation))
        if verdict.intent == "meta" and conversation:
            text = answer_meta_sync(original, conversation, language)
            return Answer(
                query=original, rewritten_query=original, text=text,
                citations=[], abstained=False, reason="", hits=[],
                latency_ms=int((time.perf_counter() - t0) * 1000),
                trace={"language": language, "intent": "meta",
                       "intent_matched": verdict.matched, "intent_via": verdict.via,
                       "conversation_turns": len(conversation or [])},
            )
        if verdict.intent == "chitchat":
            text = answer_chitchat_sync(original, language)
            return Answer(
                query=original, rewritten_query=original, text=text,
                citations=[], abstained=False, reason="", hits=[],
                latency_ms=int((time.perf_counter() - t0) * 1000),
                trace={"language": language, "intent": "chitchat",
                       "intent_matched": verdict.matched, "intent_via": verdict.via},
            )

    rewritten = rewrite_query(original, conversation=conversation) if use_rewrite else original

    variants_used: list[str] = [rewritten]
    if use_multi_query:
        variants_used = expand_queries(rewritten, conversation=conversation)
        hits = _retrieve_with_multi_query(
            original_for_rerank=original,
            variants=variants_used,
            kb_dir=kb.root,
            filter=filter,
        )
    else:
        hits = retrieve(rewritten, kb.root, rerank=use_rerank, filter=filter)

    # Small-to-big: swap each precision-matched leaf for its full parent section.
    hits = expand_to_parents(hits, kb.chunks_path)

    abstain, reason = should_abstain(hits)
    if abstain:
        fallback_text = (
            "当前知识库内信息不足,无法回答这个问题。"
            if language == "zh"
            else "I don't have enough information in the knowledge base to answer this."
        )
        return Answer(
            query=original,
            rewritten_query=rewritten,
            text=fallback_text,
            citations=[],
            abstained=True,
            reason=reason,
            hits=hits,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            trace={
                "language": language,
                "used_rerank": use_rerank,
                "used_rewrite": use_rewrite,
                "used_multi_query": use_multi_query,
                "query_variants": variants_used,
                "filter": filter,
                "conversation_turns": len(conversation or []),
            },
        )

    answer_text, _ = generate_answer(original, hits, conversation=conversation)
    citations: list[Citation] = parse_citations(answer_text, hits)

    return Answer(
        query=original,
        rewritten_query=rewritten,
        text=answer_text,
        citations=citations,
        abstained=False,
        reason="",
        hits=hits,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        trace={
            "language": language,
            "used_rerank": use_rerank,
            "used_rewrite": use_rewrite,
            "conversation_turns": len(conversation or []),
        },
    )


async def answer_query_async(
    query: str,
    kb_id: str,
    *,
    use_rewrite: bool = True,
    use_rerank: bool = True,
    use_multi_query: bool = False,
    use_semantic_cache: bool = True,
    conversation: list[dict] | None = None,
    filter: dict | None = None,
) -> Answer:
    """Async pipeline. Runs BM25 + vector in parallel, uses async LLM + cache."""
    t0 = time.perf_counter()
    kb = get_kb(kb_id)
    if not kb.is_built():
        raise RuntimeError(f"KB '{kb_id}' has no built indexes. Run build_kb.py first.")

    original = normalize_query(query)
    language = detect_language(original)

    # ---- Prompt-injection defence (opt-out via settings) ----
    if getattr(settings, "enable_injection_defense", True):
        verdict = detect_injection(original)
        if verdict.is_injection:
            log.warning("injection_blocked", rule=verdict.rule, matched=verdict.matched[:120] if verdict.matched else None)
            msg = (
                "该问题看起来像是尝试绕过系统指令,已被安全策略拦截。请直接向我询问与知识库相关的问题。"
                if language == "zh"
                else "This input appears to attempt a prompt-injection. Please ask a question about the knowledge base instead."
            )
            return Answer(
                query=original, rewritten_query=original, text=msg, citations=[],
                abstained=True, reason="injection_blocked", hits=[],
                latency_ms=int((time.perf_counter() - t0) * 1000),
                trace={"blocked": True, "rule": verdict.rule, "matched": verdict.matched},
            )

    # ---- Intent routing (skip retrieval for meta / chitchat) ----
    if getattr(settings, "enable_intent_routing", True):
        intent_v = classify_intent(original, has_conversation=bool(conversation))
        if intent_v.intent == "meta" and conversation:
            text = await answer_meta(original, conversation, language)
            return Answer(
                query=original, rewritten_query=original, text=text,
                citations=[], abstained=False, reason="", hits=[],
                latency_ms=int((time.perf_counter() - t0) * 1000),
                trace={"language": language, "async": True, "intent": "meta",
                       "intent_matched": intent_v.matched, "intent_via": intent_v.via,
                       "conversation_turns": len(conversation or [])},
            )
        if intent_v.intent == "chitchat":
            text = await answer_chitchat(original, language)
            return Answer(
                query=original, rewritten_query=original, text=text,
                citations=[], abstained=False, reason="", hits=[],
                latency_ms=int((time.perf_counter() - t0) * 1000),
                trace={"language": language, "async": True, "intent": "chitchat",
                       "intent_matched": intent_v.matched, "intent_via": intent_v.via},
            )

    # ---- Semantic cache peek ----
    # Skip if there's a conversation (multi-turn rewrites change meaning) or
    # an active metadata filter (answer would differ under constraints).
    sem_cache = get_semantic_cache()
    cache_probe_embedding = None
    if use_semantic_cache and not conversation and not filter and not use_multi_query:
        try:
            cache_probe_embedding = embed_query_cached(original)
            cached, sim, best_q = await sem_cache.lookup(kb_id, cache_probe_embedding)
            if cached is not None:
                log.info("semantic_cache_hit", kb_id=kb_id, similarity=round(sim, 4), matched_query=best_q[:60])
                return _answer_from_dict(cached, latency_ms=int((time.perf_counter() - t0) * 1000))
            elif best_q:
                log.info("semantic_cache_miss", kb_id=kb_id, best_similarity=round(sim, 4))
        except Exception as e:
            log.warning("semantic_cache_skip", error=str(e))

    if use_rewrite:
        rewritten = await asyncio.to_thread(rewrite_query, original, conversation)
    else:
        rewritten = original

    variants_used: list[str] = [rewritten]
    if use_multi_query:
        variants_used = await asyncio.to_thread(expand_queries, rewritten, conversation)
        per_variant = await asyncio.gather(
            *(retrieve_async(v, kb.root, rerank=False, final_top_k=settings.bm25_top_k, filter=filter)
              for v in variants_used)
        )
        per_variant = [h for h in per_variant if h]
        if not per_variant:
            hits = []
        else:
            fused = rrf_fuse(per_variant, k=settings.rrf_k)
            hits = await asyncio.to_thread(
                rerank_hits, original, fused[:max(settings.rerank_top_k, settings.final_top_k)]
            )
            hits = hits[: settings.final_top_k]
    else:
        hits = await retrieve_async(rewritten, kb.root, rerank=use_rerank, filter=filter)

    hits = expand_to_parents(hits, kb.chunks_path)

    abstain, reason = should_abstain(hits)
    if abstain:
        fallback_text = (
            "当前知识库内信息不足,无法回答这个问题。"
            if language == "zh"
            else "I don't have enough information in the knowledge base to answer this."
        )
        return Answer(
            query=original,
            rewritten_query=rewritten,
            text=fallback_text,
            citations=[],
            abstained=True,
            reason=reason,
            hits=hits,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            trace={"language": language, "async": True, "used_rewrite": use_rewrite,
                   "used_multi_query": use_multi_query, "query_variants": variants_used,
                   "conversation_turns": len(conversation or [])},
        )

    system, user = build_answer_prompt(original, hits, language, conversation=conversation)
    provider_used = "qwen"
    try:
        answer_text, provider_used = await chat_once_with_fallback(
            system=system,
            user=user,
            task="generate",
            hits=hits,
            user_query_for_extractive=original,
        )
        answer_text = answer_text.strip()
    except LLMError as e:
        # Fallback chain either disabled or disabled-and-all-rungs-failed:
        # degrade gracefully like before so the API still returns a well-formed Answer.
        fallback_text = (
            "当前知识库内信息不足,无法回答这个问题。"
            if language == "zh"
            else "I don't have enough information in the knowledge base to answer this."
        )
        answer_text = f"{fallback_text} (LLM error: {e})"
        provider_used = "error"

    # Strip any secret-leak patterns from generated output.
    if getattr(settings, "enable_injection_defense", True):
        answer_text, leaked = sanitize_output(answer_text)
        if leaked:
            log.warning("output_sanitized", redactions=len(leaked))

    citations = parse_citations(answer_text, hits)

    answer = Answer(
        query=original,
        rewritten_query=rewritten,
        text=answer_text,
        citations=citations,
        abstained=False,
        reason="",
        hits=hits,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        trace={
            "language": language,
            "async": True,
            "used_rewrite": use_rewrite,
            "used_rerank": use_rerank,
            "used_multi_query": use_multi_query,
            "query_variants": variants_used,
            "filter": filter,
            "conversation_turns": len(conversation or []),
            "llm_provider": provider_used,
        },
    )

    # Persist to semantic cache for future rephrased lookups.
    if use_semantic_cache and cache_probe_embedding is not None and not answer.abstained:
        try:
            await sem_cache.store(
                kb_id=kb_id,
                query=original,
                query_embedding=cache_probe_embedding,
                answer=_answer_to_dict(answer),
            )
        except Exception as e:
            log.warning("semantic_cache_store_failed", error=str(e))

    return answer


# --------------------------------------------------------------------------
# Sprint A.4 -- streaming pipeline
# --------------------------------------------------------------------------

async def answer_query_stream(
    query: str,
    kb_id: str,
    *,
    use_rewrite: bool = True,
    use_rerank: bool = True,
    conversation: list[dict] | None = None,
    filter: dict | None = None,
):
    """Yield ``(event_type, payload)`` tuples for SSE streaming.

    Event sequence:
        - ``meta``     -- {"rewritten_query", "language"}
        - ``hits``     -- {"citations_preview": [...]} (after retrieval, before LLM)
        - ``abstain``  -- {"text", "reason"} if evidence too weak (then ``done``)
        - ``delta``    -- {"text": "..."} streamed from LLM, 0..N times
        - ``done``     -- {"citations": [...], "latency_ms": ..., "abstained": bool}
        - ``error``    -- {"detail": "..."} on LLM or internal failure

    Streaming here applies to the GENERATION step only. Retrieval/rerank is
    synchronous because it's only 200-500ms and splitting it into events
    fragments the frontend for no gain. Users feel the first token arriving
    within ~1s of submission, which is the point.
    """
    t0 = time.perf_counter()
    kb = get_kb(kb_id)
    if not kb.is_built():
        yield ("error", {"detail": f"KB '{kb_id}' has no built indexes."})
        return

    original = normalize_query(query)
    language = detect_language(original)

    # Injection guard -- same rule as non-streaming path.
    if getattr(settings, "enable_injection_defense", True):
        verdict = detect_injection(original)
        if verdict.is_injection:
            msg = (
                "该问题看起来像是尝试绕过系统指令,已被安全策略拦截。"
                if language == "zh"
                else "This input appears to attempt a prompt-injection. Please ask a knowledge-base question instead."
            )
            yield ("abstain", {"text": msg, "reason": "injection_blocked"})
            yield ("done", {"abstained": True, "citations": [],
                            "latency_ms": int((time.perf_counter() - t0) * 1000)})
            return

    # Intent routing (SSE variant). Meta/chitchat skip retrieval entirely:
    # we emit `meta` with intent=..., a synthetic empty `hits`, then stream
    # the LLM reply as `delta` events, then `done` with no citations.
    if getattr(settings, "enable_intent_routing", True):
        intent_v = classify_intent(original, has_conversation=bool(conversation))
        if intent_v.intent in ("meta", "chitchat") and not (
            intent_v.intent == "meta" and not conversation
        ):
            yield ("meta", {
                "rewritten_query": original, "language": language,
                "intent": intent_v.intent, "intent_matched": intent_v.matched,
            })
            yield ("hits", {"citations_preview": []})
            full_parts: list[str] = []
            stream_iter = (
                stream_meta(original, conversation, language)
                if intent_v.intent == "meta"
                else stream_chitchat(original, language)
            )
            try:
                async for piece in stream_iter:
                    full_parts.append(piece)
                    yield ("delta", {"text": piece})
            except LLMError as e:
                yield ("error", {"detail": f"LLM error: {e}"})
            yield ("done", {
                "abstained": False, "citations": [],
                "intent": intent_v.intent,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
            })
            return

    rewritten = (
        await asyncio.to_thread(rewrite_query, original, conversation)
        if use_rewrite else original
    )
    yield ("meta", {"rewritten_query": rewritten, "language": language})

    hits = await retrieve_async(rewritten, kb.root, rerank=use_rerank, filter=filter)
    hits = expand_to_parents(hits, kb.chunks_path)

    yield ("hits", {"citations_preview": [
        {"n": i + 1, "title": h.title, "source_id": h.source_id, "score": h.score}
        for i, h in enumerate(hits[:5])
    ]})

    abstain, reason = should_abstain(hits)
    if abstain:
        fallback_text = (
            "当前知识库内信息不足,无法回答这个问题。"
            if language == "zh"
            else "I don't have enough information in the knowledge base to answer this."
        )
        yield ("abstain", {"text": fallback_text, "reason": reason})
        yield ("done", {"abstained": True, "citations": [],
                        "latency_ms": int((time.perf_counter() - t0) * 1000)})
        return

    system, user = build_answer_prompt(original, hits, language, conversation=conversation)
    full_parts: list[str] = []
    try:
        async for piece in chat_stream_async(system=system, user=user, task="generate"):
            full_parts.append(piece)
            yield ("delta", {"text": piece})
    except LLMError as e:
        yield ("error", {"detail": f"LLM error: {e}"})
        yield ("done", {"abstained": True, "citations": [],
                        "latency_ms": int((time.perf_counter() - t0) * 1000)})
        return

    answer_text = "".join(full_parts).strip()
    if getattr(settings, "enable_injection_defense", True):
        answer_text, leaked = sanitize_output(answer_text)
        if leaked:
            log.warning("stream_output_sanitized", redactions=len(leaked))

    citations = parse_citations(answer_text, hits)
    yield ("done", {
        "abstained": False,
        "citations": [
            {"n": c.n, "source_id": c.source_id, "title": c.title,
             "source_path": c.source_path, "section_path": list(c.section_path or []),
             "snippet": c.snippet}
            for c in citations
        ],
        "latency_ms": int((time.perf_counter() - t0) * 1000),
    })
