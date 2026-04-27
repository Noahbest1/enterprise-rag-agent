"""Vision-language model client.

We call Qwen-VL via the same OpenAI-compatible DashScope endpoint used by
``answer.llm_client``; the only shape differences are:

  * We POST to the ``multimodal-generation`` / chat completions endpoint.
    DashScope's OpenAI-compat endpoint accepts image_url content blocks.
  * Model is ``qwen-vl-max`` (or the configured ``qwen_vision_model``).
  * Input images are provided as data URLs (base64-encoded) so callers can
    push arbitrary bytes without hosting them.

Two canonical tasks we expose:

  ``describe_image(image_bytes, question)`` -- free-form Q&A over an image.
      Use for: "does this product look defective?", "what error is shown
      in this screenshot?", "summarise this receipt".

  ``extract_text_from_image(image_bytes)`` -- OCR-style extraction.
      Use for: invoice / screenshot text pulled out for the RAG to search.

Both return plain strings so downstream code can feed them into the RAG
or agent pipelines without special plumbing.
"""
from __future__ import annotations

import base64
import json
from typing import Literal

import httpx

from ..answer.llm_client import _headers
from ..config import settings
from ..logging import get_logger


log = get_logger(__name__)


class VLMError(RuntimeError):
    pass


DEFAULT_VISION_MODEL = "qwen-vl-max"


def _detect_mime(data: bytes) -> str:
    """Minimal magic-number sniff; fall back to jpeg."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"GIF8"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _data_url(image_bytes: bytes) -> str:
    mime = _detect_mime(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _call_vlm(messages: list[dict], *, max_tokens: int, temperature: float, timeout: int) -> str:
    if not settings.qwen_api_key:
        raise VLMError("DASHSCOPE_API_KEY not set")
    model = getattr(settings, "qwen_vision_model", DEFAULT_VISION_MODEL)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(settings.qwen_chat_url, json=payload, headers=_headers())
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        raise VLMError(f"VLM HTTP {e.response.status_code}: {e.response.text[:300]}") from e
    except httpx.RequestError as e:
        raise VLMError(f"VLM network error: {e}") from e

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise VLMError(f"Unexpected VLM response: {json.dumps(data)[:300]}") from e


def describe_image(
    image_bytes: bytes,
    question: str = "",
    *,
    language: Literal["zh", "en", "auto"] = "auto",
    max_tokens: int = 400,
    temperature: float = 0.2,
    timeout: int = 60,
) -> str:
    """Return an answer to ``question`` grounded in the image.

    If question is empty, we default to "describe this image in detail"
    (matches the semantics most customer-service scenarios need).
    """
    q = question.strip() or "请仔细描述这张图片,包含能看到的文字、商品、缺陷或问题。"
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant for an e-commerce customer-service agent. "
                "When shown an image, describe what is relevant to a shopping / delivery / "
                "product-defect / error-screen context. Be concrete. Do not invent text that "
                "is not visible in the image. Reply in the language of the user's question."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _data_url(image_bytes)}},
                {"type": "text", "text": q},
            ],
        },
    ]
    out = _call_vlm(messages, max_tokens=max_tokens, temperature=temperature, timeout=timeout)
    log.info("vlm_describe_ok", q_len=len(q), out_len=len(out), image_bytes=len(image_bytes))
    return out


def extract_text_from_image(
    image_bytes: bytes,
    *,
    max_tokens: int = 600,
    temperature: float = 0.0,
    timeout: int = 60,
) -> str:
    """Transcribe visible text from the image (no commentary, no translation).

    Useful for invoices, screenshots of error dialogs, receipts that later
    need full-text search via the RAG pipeline.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an OCR engine. Transcribe the visible text in the image, "
                "preserving line breaks. Do not add commentary, explanations, or translations. "
                "If the image contains no text, reply exactly: (no visible text)"
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _data_url(image_bytes)}},
                {"type": "text", "text": "Transcribe."},
            ],
        },
    ]
    return _call_vlm(messages, max_tokens=max_tokens, temperature=temperature, timeout=timeout)
