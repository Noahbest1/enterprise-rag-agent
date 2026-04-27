"""Product similarity recommender.

Build-time is free: we reuse the BGE-M3 encoder from the vector store, and
the product catalogue lives in the DB (``order_items.title`` × ``sku``
uniques act as our product catalogue in this mock). Real platforms would
use a dedicated ``products`` table, but the shape here is identical --
embed a canonical product text, store the vector, nearest-neighbours at
query time.

Cached in-process because the catalogue is small (~30 items) and the
encoder call dominates latency.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from rag.db.base import SessionLocal
from rag.db.models import OrderItem


_lock = threading.Lock()


@dataclass
class Product:
    sku: str
    title: str
    avg_price_yuan: float
    embedding: np.ndarray


@lru_cache(maxsize=1)
def _load_catalogue() -> list[Product]:
    from rag.index.faiss_store import embed_texts

    with SessionLocal() as s:
        rows = s.query(OrderItem).all()
        # dedupe by sku, take first title + average price
        by_sku: dict[str, dict] = {}
        for it in rows:
            d = by_sku.setdefault(it.sku, {"titles": [], "prices": []})
            d["titles"].append(it.title)
            d["prices"].append(it.unit_price_cents / 100.0)

    if not by_sku:
        return []

    titles_for_embed = [sorted(d["titles"], key=len, reverse=True)[0] for d in by_sku.values()]
    with _lock:
        vecs = embed_texts(titles_for_embed, batch_size=16)

    products: list[Product] = []
    for (sku, d), title, vec in zip(by_sku.items(), titles_for_embed, vecs):
        avg = sum(d["prices"]) / len(d["prices"])
        products.append(Product(sku=sku, title=title, avg_price_yuan=round(avg, 2), embedding=vec))
    return products


def refresh_catalogue() -> int:
    """Drop the cached catalogue so the next call reloads. Call after seeding."""
    _load_catalogue.cache_clear()
    return len(_load_catalogue())


def similar_products(query_text: str, top_k: int = 5, exclude_sku: str | None = None) -> list[dict]:
    """Return up to ``top_k`` products ranked by cosine similarity to ``query_text``."""
    from rag.index.faiss_store import embed_query_cached

    products = _load_catalogue()
    if not products:
        return []

    qvec = embed_query_cached(query_text)[0]
    qn = float(np.linalg.norm(qvec)) or 1.0

    scored: list[tuple[float, Product]] = []
    for p in products:
        if exclude_sku and p.sku == exclude_sku:
            continue
        pn = float(np.linalg.norm(p.embedding)) or 1.0
        sim = float(np.dot(qvec, p.embedding) / (qn * pn))
        scored.append((sim, p))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [
        {"sku": p.sku, "title": p.title, "price_yuan": p.avg_price_yuan, "similarity": round(s, 4)}
        for s, p in scored[:top_k]
    ]
