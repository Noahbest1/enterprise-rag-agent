"""CLIP-based image + text embedding in a shared 512-d space.

Uses sentence-transformers' ``clip-ViT-B-32`` (OpenAI CLIP). Same encoder
gives 512-d vectors for both images (via the vision tower) and text (via
the text tower), so cosine similarity is meaningful across modalities:

    similarity(image_of_iphone, text="iPhone 16 Pro red") > 0.25
    similarity(image_of_iphone, text="a rainy street")    ≈ 0.10

Why this model:
- Ships with sentence-transformers (already a direct dep via BGE-M3).
- ~180 MB, CPU-runnable.
- Text tower accepts both zh and en (mediocre on zh; fine for demo).

Vectors are always L2-normalised so dot product == cosine similarity.

This module is OPTIONAL: if sentence-transformers' CLIP variant isn't
downloadable (offline env), any caller can catch the raise and fall back
to the text-only pipeline. The first call lazy-loads + caches the model.
"""
from __future__ import annotations

import io
from functools import lru_cache

import numpy as np


_CLIP_MODEL_NAME = "clip-ViT-B-32"
_EMBED_DIM = 512


@lru_cache(maxsize=1)
def _get_clip():
    """Load CLIP once per process. ~180 MB download first time."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_CLIP_MODEL_NAME)


def embed_image_bytes(raw: bytes) -> np.ndarray:
    """Encode image bytes to a 512-d unit vector (cosine-ready)."""
    if not raw:
        raise ValueError("empty image bytes")
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError(f"cannot decode image: {e}") from e
    vec = _get_clip().encode(img, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vec, dtype=np.float32).reshape(-1)


def embed_text_for_image(text: str) -> np.ndarray:
    """Encode text into CLIP's shared space. Use for text->image queries."""
    if not text:
        raise ValueError("empty text")
    vec = _get_clip().encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
    return np.asarray(vec, dtype=np.float32).reshape(-1)


def embed_dim() -> int:
    return _EMBED_DIM


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity on already-normalised vectors. Guards against non-unit inputs."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
