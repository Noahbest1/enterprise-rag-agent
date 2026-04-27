"""Prometheus metrics.

Optional dependency: if ``prometheus-client`` isn't installed, every helper
becomes a no-op and ``/metrics`` returns an empty body. We keep it optional
so pytest on a lean venv still works.

Metrics exposed (when enabled):

- ``rag_requests_total{method,path,status}``       -- HTTP request count
- ``rag_request_duration_seconds{method,path}``    -- HTTP latency histogram
- ``rag_llm_calls_total{task,status}``             -- LLM call count (ok|error)
- ``rag_llm_tokens_total{task,kind}``              -- prompt|completion tokens
- ``rag_cache_events_total{layer,result}``         -- cache hit/miss by layer

Labels are kept low-cardinality: ``path`` uses the FastAPI route template
(``/kbs/{kb_id}``), never the full URL, so we don't blow out the series count
in a production with many KBs or order ids.
"""
from __future__ import annotations

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    _ENABLED = True
except ImportError:  # pragma: no cover -- exercised by lean-venv users
    _ENABLED = False


_BUCKETS_SECONDS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)


if _ENABLED:
    REGISTRY = CollectorRegistry()

    _REQUESTS_TOTAL = Counter(
        "rag_requests_total",
        "HTTP requests received, labelled by method/path/status",
        ["method", "path", "status"],
        registry=REGISTRY,
    )
    _REQUEST_DURATION = Histogram(
        "rag_request_duration_seconds",
        "HTTP request duration in seconds",
        ["method", "path"],
        buckets=_BUCKETS_SECONDS,
        registry=REGISTRY,
    )
    _LLM_CALLS = Counter(
        "rag_llm_calls_total",
        "LLM chat completions issued",
        ["task", "status"],
        registry=REGISTRY,
    )
    _LLM_TOKENS = Counter(
        "rag_llm_tokens_total",
        "LLM tokens consumed, split prompt vs completion",
        ["task", "kind"],
        registry=REGISTRY,
    )
    _CACHE_EVENTS = Counter(
        "rag_cache_events_total",
        "Cache get() outcomes, by layer",
        ["layer", "result"],
        registry=REGISTRY,
    )
else:  # pragma: no cover
    REGISTRY = None  # type: ignore[assignment]


def is_enabled() -> bool:
    return _ENABLED


def record_request(method: str, path: str, status: int, latency_s: float) -> None:
    if not _ENABLED:
        return
    _REQUESTS_TOTAL.labels(method=method, path=path, status=str(status)).inc()
    _REQUEST_DURATION.labels(method=method, path=path).observe(max(0.0, latency_s))


def record_llm(task: str, input_tokens: int, output_tokens: int, status: str = "ok") -> None:
    """Called once per LLM API call. Pass 0 tokens if the provider didn't return usage."""
    if not _ENABLED:
        return
    _LLM_CALLS.labels(task=task, status=status).inc()
    if status == "ok":
        if input_tokens:
            _LLM_TOKENS.labels(task=task, kind="input").inc(input_tokens)
        if output_tokens:
            _LLM_TOKENS.labels(task=task, kind="output").inc(output_tokens)


def record_cache(layer: str, hit: bool) -> None:
    if not _ENABLED:
        return
    _CACHE_EVENTS.labels(layer=layer, result="hit" if hit else "miss").inc()


def metrics_payload() -> tuple[bytes, str]:
    """Return (body, content_type) for the `/metrics` endpoint."""
    if not _ENABLED:
        return b"# prometheus_client not installed\n", "text/plain; charset=utf-8"
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def reset_for_tests() -> None:
    """Wipe counter state. ONLY call from tests."""
    if not _ENABLED:
        return
    for m in (_REQUESTS_TOTAL, _REQUEST_DURATION, _LLM_CALLS, _LLM_TOKENS, _CACHE_EVENTS):
        # Rebuilds the internal samples dict; safer than iterating labels.
        m._metrics.clear()  # type: ignore[attr-defined]
