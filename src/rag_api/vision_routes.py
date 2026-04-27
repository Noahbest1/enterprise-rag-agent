"""Multimodal endpoints (PH3 upgrade).

Endpoints
---------
  POST /vision/describe    (multipart)
    image file(s) + optional question
    → [{description, width, height, pixel_hash, cached}]
    Multi-image: send `images` multiple times or a single `image` field.

  POST /vision/layout      (multipart)
    image file
    → {regions:[{type,text,markdown,...}], pixel_hash}

  POST /vision/ask         (multipart)
    image file(s) + question + kb_id
    → RAG-grounded answer using the images' descriptions as context

  POST /vision/ocr         (multipart)
    image file
    → {ocr_text, pixel_hash}

Caching
-------
All endpoints go through ``rag.vision.pipeline.*_cached`` helpers: same
pixel hash returns the same result for 24h, so repeated uploads of the
same screenshot don't re-bill the VLM.
"""
from __future__ import annotations

import asyncio
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from rag.knowledge_base import get_kb
from rag.logging import get_logger
from rag.pipeline import answer_query_async
from rag.vision import (
    ImagePreprocessError,
    VLMError,
    analyse_layout_cached,
    describe_image_cached,
    ocr_image_cached,
)
from rag.vision.image_embed import embed_image_bytes, embed_text_for_image
from rag.vision.image_index import ImageIndex


router = APIRouter()
log = get_logger("vision_routes")

MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB per file
MAX_IMAGES_PER_REQUEST = 5


async def _read_and_validate(image: UploadFile) -> tuple[bytes, str]:
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image payload")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"image too large: {len(data)} bytes (limit {MAX_IMAGE_BYTES})",
        )
    return data, image.filename or "unnamed"


async def _read_many(image: UploadFile | None, images: List[UploadFile] | None) -> list[tuple[bytes, str]]:
    files: list[UploadFile] = []
    if images:
        files.extend(images)
    if image is not None:
        files.append(image)
    if not files:
        raise HTTPException(status_code=400, detail="no image provided")
    if len(files) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"too many images: {len(files)} (limit {MAX_IMAGES_PER_REQUEST})",
        )
    return [await _read_and_validate(f) for f in files]


@router.post("/vision/describe")
async def vision_describe(
    image: UploadFile | None = File(None),
    images: List[UploadFile] = File(default_factory=list),
    question: str = Form(""),
):
    payloads = await _read_many(image, images)
    try:
        results = await asyncio.to_thread(
            lambda: [describe_image_cached(b, question) for b, _ in payloads]
        )
    except (VLMError, ImagePreprocessError) as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "count": len(results),
        "items": [
            {**r, "filename": fn}
            for r, (_, fn) in zip(results, payloads)
        ],
    }


@router.post("/vision/layout")
async def vision_layout(
    image: UploadFile = File(...),
):
    data, fn = await _read_and_validate(image)
    try:
        result = await asyncio.to_thread(analyse_layout_cached, data)
    except (VLMError, ImagePreprocessError) as e:
        raise HTTPException(status_code=502, detail=str(e))
    result["filename"] = fn
    return result


@router.post("/vision/ocr")
async def vision_ocr(
    image: UploadFile = File(...),
):
    data, fn = await _read_and_validate(image)
    try:
        result = await asyncio.to_thread(ocr_image_cached, data)
    except (VLMError, ImagePreprocessError) as e:
        raise HTTPException(status_code=502, detail=str(e))
    result["filename"] = fn
    return result


@router.post("/vision/ask")
async def vision_ask(
    question: str = Form(...),
    kb_id: str = Form(...),
    image: UploadFile | None = File(None),
    images: List[UploadFile] = File(default_factory=list),
):
    """Describe images → concat into the user's question → RAG answer.

    Multi-image: if the user sends 3 screenshots, each is described
    individually and all descriptions go into the composed query. The
    LLM then decides what to match against the KB.
    """
    payloads = await _read_many(image, images)
    try:
        descriptions = await asyncio.to_thread(
            lambda: [
                describe_image_cached(b, f"简要描述图中与'{question}'相关的信息。")
                for b, _ in payloads
            ]
        )
    except (VLMError, ImagePreprocessError) as e:
        raise HTTPException(status_code=502, detail=f"image understanding failed: {e}")

    image_block = "\n\n".join(
        f"[图片 {i+1}: {fn}]\n{d['description']}"
        for i, ((_, fn), d) in enumerate(zip(payloads, descriptions))
    )
    composed_query = (
        f"用户提问:{question}\n\n"
        f"用户上传了 {len(payloads)} 张图片,内容如下:\n{image_block}\n\n"
        f"如果上述图片内容与知识库的主题明显无关(例如上传了和本店铺 / 本产品 / 本政策完全不相关的图片),"
        f"请不要引用任何知识库条目,直接告知用户'这张图片与当前知识库的内容无关',然后只基于图片本身回答。"
        f"否则,请基于以上信息和知识库回答用户。"
    )
    log.info("vision_ask", kb_id=kb_id, n_images=len(payloads), total_desc_len=sum(len(d["description"]) for d in descriptions))

    try:
        ans = await answer_query_async(composed_query, kb_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Guard against off-domain "hallucinated citations": if the best retrieval
    # score is weak, we strip the citations from the response and fall back
    # to pure image description. The LLM can still mention what's in the
    # image, but the UI won't imply the KB actually supports anything.
    # Only kicks in when there ARE hits but they're weak -- an empty hits
    # list is the existing abstain path, handled by answer_query_async.
    top_score = ans.hits[0].score if ans.hits else None
    off_domain_threshold = 0.35
    stripped_citations = False
    answer_text = ans.text
    citations_out = [
        {"n": c.n, "title": c.title, "source_id": c.source_id, "snippet": c.snippet}
        for c in ans.citations
    ]
    if top_score is not None and top_score < off_domain_threshold and not ans.abstained:
        stripped_citations = True
        citations_out = []
        # Remove [1][2]... markers from the answer so the UI isn't misleading.
        import re
        answer_text = re.sub(r"\s*\[\d+\]", "", ans.text)
        # Prepend a clear notice.
        joined_desc = "\n".join(d["description"] for d in descriptions)
        answer_text = (
            f"(这张图的内容与 {kb_id} 知识库无关,以下只基于图片本身,不引用 KB。)\n\n"
            f"{answer_text or joined_desc}"
        )
        log.info("vision_ask_off_domain", kb_id=kb_id, top_score=round(top_score, 3))

    return {
        "image_count": len(payloads),
        "image_descriptions": [
            {"filename": fn, "description": d["description"], "pixel_hash": d.get("pixel_hash")}
            for (_, fn), d in zip(payloads, descriptions)
        ],
        "composed_query_preview": composed_query[:250],
        "answer": answer_text,
        "abstained": ans.abstained,
        "off_domain": stripped_citations,
        "top_retrieval_score": round(top_score, 4) if top_score is not None else None,
        "citations": citations_out,
        "latency_ms": ans.latency_ms,
    }


# ---------- sprint A.multimodal: CLIP image embed + image search ----------

@router.post("/vision/search")
async def vision_search(
    kb_id: str = Form(...),
    image: UploadFile | None = File(None),
    text: str = Form(""),
    top_k: int = Form(5),
):
    """Search a KB's image index by image (image->image) or text (text->image).

    Pass EITHER ``image`` OR ``text`` (if both are present, image wins). The
    index must have been built previously via ``ImageIndex.add`` + ``save``
    during KB sync; if no index file exists we return an empty hit list.
    """
    try:
        kb = get_kb(kb_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if image is None and not text.strip():
        raise HTTPException(status_code=400, detail="provide either `image` or `text`")

    if image is not None:
        data, _fn = await _read_and_validate(image)
        try:
            query_vec = await asyncio.to_thread(embed_image_bytes, data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        mode = "image"
    else:
        try:
            query_vec = await asyncio.to_thread(embed_text_for_image, text.strip())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        mode = "text"

    idx = ImageIndex.load(kb.root)
    hits = await asyncio.to_thread(idx.search, query_vec, top_k=max(1, min(int(top_k), 50)))
    return {
        "kb_id": kb_id,
        "mode": mode,
        "index_size": len(idx),
        "hits": [
            {
                "source_id": h.source_id,
                "title": h.title,
                "source_path": h.source_path,
                "similarity": round(h.similarity, 4),
            }
            for h in hits
        ],
    }
