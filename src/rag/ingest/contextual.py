"""Contextual Retrieval (Anthropic, 2024).

Before embedding or indexing each leaf chunk, we ask the LLM to generate a
short (1-2 sentence) context paragraph that situates the chunk within its
parent document. The paragraph is prepended to the chunk text; the
prepended text is what gets embedded and BM25-indexed. The chunk's
*original* text is preserved in ``text`` for display / LLM context.

Anthropic reports ~35% recall improvement (contextual embeddings +
contextual BM25 + reranker). We only wire the embedding+BM25 side here;
reranker is unchanged.

Cost model
----------
1 LLM call per leaf. For a 450-leaf KB this is ~450 Qwen calls at
max_output_tokens=80, so ~30¢ per full rebuild at current Qwen prices.
Rebuilds are infrequent so this is fine; for truly large KBs you'd use
prompt caching + batching, not in scope today.

Opt-in
------
Controlled by ``settings.enable_contextual_retrieval``. Default off so
eval reproducibility is preserved. Turn on per KB via the CLI flag.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..answer.llm_client import LLMError, chat_once
from ..config import settings
from ..logging import get_logger
from ..types import Chunk


log = get_logger(__name__)


SYSTEM = """You generate short contextual summaries that situate a specific chunk of text within its source document.

Your output is prepended to the chunk before embedding, so it must:
- Be 1-2 sentences, ≤ 60 words
- Identify the subject, parent section, and what this chunk is about
- Use the language of the chunk (Chinese or English)
- NOT repeat or paraphrase the chunk's content
- NOT invent facts; if unsure, stick to document-level framing

Output ONLY the contextual summary. No labels, no quotes, no markdown.
"""


def _build_prompt_user(title: str, section_path: list[str], chunk_text: str, parent_text: str | None) -> str:
    breadcrumb = " / ".join(section_path) if section_path else "(top level)"
    parent_block = ""
    if parent_text and parent_text.strip() != chunk_text.strip():
        # Clip parent so prompt stays short.
        pt = parent_text if len(parent_text) <= 1500 else parent_text[:1500] + " ..."
        parent_block = f"\nParent section text:\n{pt}\n"
    return (
        f"Document title: {title}\n"
        f"Section: {breadcrumb}\n"
        f"{parent_block}\n"
        f"Chunk to situate:\n{chunk_text}\n\n"
        f"Write the 1-2 sentence contextual summary."
    )


def _cache_path(kb_dir: Path) -> Path:
    return kb_dir / "contextual_cache.jsonl"


def _load_cache(kb_dir: Path) -> dict[str, str]:
    path = _cache_path(kb_dir)
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out[d["key"]] = d["context"]
            except Exception:
                continue
    return out


def _append_cache(kb_dir: Path, key: str, context: str) -> None:
    with _cache_path(kb_dir).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "context": context}, ensure_ascii=False) + "\n")


def _key_for(chunk: Chunk) -> str:
    # Keyed by content hash so re-building doesn't regenerate unchanged chunks.
    payload = f"{chunk.title}|{'/'.join(chunk.section_path)}|{chunk.text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def annotate_chunks_with_context(
    chunks: list[Chunk],
    kb_dir: Path,
) -> list[Chunk]:
    """Return new Chunk list with contextual prefix baked into ``text``.

    Only leaves are annotated (parents are not indexed). Parents are used
    to look up the "surrounding context" passed to the LLM, so the
    generated summary knows what neighbors exist.
    """
    parents_by_id = {c.chunk_id: c for c in chunks if c.chunk_role == "parent"}
    cache = _load_cache(kb_dir)

    annotated: list[Chunk] = []
    total = sum(1 for c in chunks if c.chunk_role == "leaf")
    done = 0

    for c in chunks:
        if c.chunk_role != "leaf":
            annotated.append(c)
            continue

        key = _key_for(c)
        context = cache.get(key)
        if not context:
            parent = parents_by_id.get(c.parent_id) if c.parent_id else None
            user = _build_prompt_user(
                title=c.title,
                section_path=c.section_path,
                chunk_text=c.text,
                parent_text=parent.text if parent else None,
            )
            try:
                reply = chat_once(system=SYSTEM, user=user, max_output_tokens=120, temperature=0.0)
                context = reply.strip().splitlines()[0].strip() if reply else ""
            except LLMError as e:
                log.warning("contextual_llm_failed", chunk_id=c.chunk_id, error=str(e))
                context = ""
            if context:
                _append_cache(kb_dir, key, context)
                cache[key] = context

        done += 1
        if done % 20 == 0:
            log.info("contextual_progress", done=done, total=total)

        # Prepend context to text. Keep a copy of the original in metadata
        # so display paths can show the un-doctored text.
        new_text = (context + "\n\n" + c.text) if context else c.text
        new_meta = {**(c.metadata or {}), "contextual_prefix": context, "original_text": c.text}
        annotated.append(
            Chunk(
                chunk_id=c.chunk_id,
                kb_id=c.kb_id,
                source_id=c.source_id,
                source_path=c.source_path,
                title=c.title,
                section_path=c.section_path,
                text=new_text,
                token_count=c.token_count + len((context or "").split()),
                order=c.order,
                metadata=new_meta,
                parent_id=c.parent_id,
                chunk_role=c.chunk_role,
            )
        )
    return annotated
