"""Layout-aware region extraction.

Instead of shipping a PP-StructureV3 / LayoutLMv3 dependency (hundreds of
MB of model), we lean on Qwen-VL with a structured prompt asking it to
return regions as JSON:

    {
      "regions": [
        {"type": "title", "text": "..."},
        {"type": "text",  "text": "..."},
        {"type": "table", "markdown": "| a | b |\\n|--|--|\\n| 1 | 2 |"},
        {"type": "figure", "caption": "A bar chart showing ..."},
        {"type": "code",  "language": "py", "text": "..."}
      ]
    }

This gives us layout-typed output for chunking without a layout model.
Trade-off: Qwen-VL is less deterministic than a dedicated detector, so
we parse defensively and fall back to "one big text region" when the
response doesn't match schema.

The returned ``Region`` list plugs directly into the chunker in
``pipeline.py`` -- text regions become prose chunks, tables stay atomic,
figures contribute their caption as an indexable chunk with
``metadata.region_type="figure"``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from ..answer.llm_client import LLMError
from ..logging import get_logger
from .vlm import _call_vlm, _data_url


log = get_logger(__name__)


RegionType = Literal["title", "text", "table", "figure", "code", "unknown"]


@dataclass
class Region:
    type: RegionType
    text: str = ""                  # for title / text / figure_caption / code
    markdown: str = ""              # for table (table markdown output)
    language: str | None = None     # for code
    extra: dict = field(default_factory=dict)


SYSTEM_LAYOUT = """You are a document layout analyser. Given an image, return structured JSON describing the regions you see, in reading order.

Return EXACTLY this JSON shape (no prose, no markdown fences):

{
  "regions": [
    {"type": "title",  "text": "..."},
    {"type": "text",   "text": "..."},
    {"type": "table",  "markdown": "| h1 | h2 |\\n|---|---|\\n| a | b |"},
    {"type": "figure", "caption": "A bar chart showing Q3 revenue by region..."},
    {"type": "code",   "language": "python", "text": "def foo(): ..."}
  ]
}

Rules:
- "type" must be one of: title, text, table, figure, code.
- For table: output GitHub-flavoured markdown (pipes + dashes). Do NOT output HTML.
- For figure: write a 1-2 sentence caption describing WHAT the figure conveys.
- For code: include language if you can infer it.
- Preserve reading order. Split long regions into multiple text regions.
- If the image has no discernible structure, return one region of type "text" with the transcribed content.
- Do not invent content that isn't visibly in the image.
"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_regions(reply: str) -> list[Region]:
    match = _JSON_RE.search(reply or "")
    if not match:
        return [Region(type="text", text=(reply or "").strip())]
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return [Region(type="text", text=match.group(0).strip())]

    raw_regions = obj.get("regions") if isinstance(obj, dict) else None
    if not isinstance(raw_regions, list) or not raw_regions:
        return [Region(type="text", text=str(obj)[:2000])]

    out: list[Region] = []
    for item in raw_regions:
        if not isinstance(item, dict):
            continue
        rtype = str(item.get("type") or "unknown").lower()
        if rtype not in {"title", "text", "table", "figure", "code"}:
            rtype = "unknown"
        out.append(
            Region(
                type=rtype,  # type: ignore[arg-type]
                text=str(item.get("text") or item.get("caption") or ""),
                markdown=str(item.get("markdown") or ""),
                language=item.get("language") or None,
                extra={k: v for k, v in item.items() if k not in {"type", "text", "caption", "markdown", "language"}},
            )
        )
    return out or [Region(type="text", text="")]


def analyse_layout(image_bytes: bytes, *, timeout: int = 60) -> list[Region]:
    """Call the VLM to produce typed regions. Never raises -- on upstream
    error returns a single unknown region so callers don't crash."""
    messages = [
        {"role": "system", "content": SYSTEM_LAYOUT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _data_url(image_bytes)}},
                {"type": "text", "text": "Analyse the layout and return the JSON."},
            ],
        },
    ]
    try:
        reply = _call_vlm(messages, max_tokens=1200, temperature=0.0, timeout=timeout)
    except LLMError as e:
        log.warning("layout_vlm_error", error=str(e))
        return [Region(type="unknown", text="")]

    regions = _parse_regions(reply)
    log.info("layout_parsed", n_regions=len(regions), types=[r.type for r in regions[:10]])
    return regions
