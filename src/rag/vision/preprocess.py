"""Image preprocessing for OCR + VLM ingestion.

What this does (pipeline order):

  1. **EXIF orientation normalisation** -- iPhone/Android photos are
     typically stored as landscape with an EXIF tag telling the viewer to
     rotate. Raw OCR/VLM engines don't always honour the tag, so rotated
     text becomes unreadable. ``ImageOps.exif_transpose`` bakes the
     rotation into the pixel data.

  2. **Format conversion to RGB JPEG** -- PNG / WEBP / GIF / HEIC go to
     RGB JPEG so downstream encoders all see one format. Also strips EXIF
     metadata (PII — GPS in photos) AFTER rotation.

  3. **Resolution clamp** -- too-small images get upscaled to a minimum
     long-edge (default 1000px) so OCR isn't starved; too-large get
     downscaled (default 4096px max long edge) so VLM latency / cost
     stays sane and we don't blow DashScope size limits.

  4. **Pixel hash** -- SHA256 over the final RGB bytes lets the cache
     layer recognise "same picture again" even if the user renamed the
     file or re-compressed it.

Returns a ``ProcessedImage`` with bytes + metadata so callers can decide
what to do (send to VLM, stash in cache, log, etc.).
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import Literal

from PIL import Image, ImageOps, UnidentifiedImageError


# Keep generous: a modern phone photo is 4032×3024, so 4096 covers native.
DEFAULT_MAX_LONG_EDGE = 4096
DEFAULT_MIN_LONG_EDGE = 1000
JPEG_QUALITY = 88


class ImagePreprocessError(RuntimeError):
    pass


@dataclass
class ProcessedImage:
    """Preprocessing output ready for VLM / OCR."""
    bytes_: bytes
    width: int
    height: int
    original_size: int
    pixel_hash: str
    format: Literal["JPEG"] = "JPEG"
    # Trail so tests + logs can see what we did
    applied: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.bytes_)


def preprocess_image(
    raw: bytes,
    *,
    max_long_edge: int = DEFAULT_MAX_LONG_EDGE,
    min_long_edge: int = DEFAULT_MIN_LONG_EDGE,
) -> ProcessedImage:
    """Normalise a raw image blob. Returns ``ProcessedImage`` or raises."""
    if not raw:
        raise ImagePreprocessError("empty image bytes")

    try:
        im = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError as e:
        raise ImagePreprocessError(f"unrecognised image format: {e}") from e

    applied: list[str] = []
    original_size = len(raw)

    # 1. EXIF rotation
    try:
        im2 = ImageOps.exif_transpose(im)
        if im2 is not None and im2 is not im:
            im = im2
            applied.append("exif_rotated")
    except Exception:  # noqa: BLE001 -- any EXIF parse error is non-fatal
        pass

    # 2. Flatten to RGB (drops alpha / palette / EXIF PII side-channels)
    if im.mode not in ("RGB",):
        if im.mode == "RGBA":
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
            applied.append("alpha_flattened")
        else:
            im = im.convert("RGB")
            applied.append(f"converted_from_{im.mode}")

    # 3. Resolution clamp
    long_edge = max(im.size)
    if long_edge > max_long_edge:
        im = _resize_preserve_aspect(im, max_long_edge)
        applied.append(f"downscaled_to_{max_long_edge}")
    elif long_edge < min_long_edge:
        im = _resize_preserve_aspect(im, min_long_edge)
        applied.append(f"upscaled_to_{min_long_edge}")

    # 4. Encode & hash
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    jpeg_bytes = buf.getvalue()
    pixel_hash = hashlib.sha256(jpeg_bytes).hexdigest()

    return ProcessedImage(
        bytes_=jpeg_bytes,
        width=im.width,
        height=im.height,
        original_size=original_size,
        pixel_hash=pixel_hash,
        format="JPEG",
        applied=applied,
    )


def _resize_preserve_aspect(im: Image.Image, target_long_edge: int) -> Image.Image:
    w, h = im.size
    long_edge = max(w, h)
    scale = target_long_edge / long_edge
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return im.resize(new_size, Image.LANCZOS)


def pixel_hash(raw: bytes) -> str:
    """Convenience: hash WITHOUT preprocessing (useful for rare pass-through paths)."""
    return hashlib.sha256(raw).hexdigest()
