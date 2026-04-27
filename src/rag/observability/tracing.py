"""OpenTelemetry tracing helpers.

Opt-in via env: set ``OTEL_EXPORTER_OTLP_ENDPOINT`` (e.g. ``http://tempo:4317``
for Grafana Tempo, ``http://jaeger:4317`` for Jaeger-Collector, etc).
If the env var is unset we install a no-op tracer provider so ``get_tracer``
still works and callers don't need ``if _enabled`` guards.

What we wrap:
- every HTTP request (root span, in the FastAPI middleware)
- LLM calls (``chat_once`` / ``chat_once_async``) -- done at their call sites
- retrieval fan-out -- done at its call site

Exporter choice:
- Default protocol is OTLP/HTTP (``OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf``).
  gRPC works too but needs ``grpcio`` on the runner.
"""
from __future__ import annotations

import os
from typing import Any

_INITIALISED = False
_ENABLED = False

try:
    from opentelemetry import trace as _trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    _OTEL_IMPORTED = True
except ImportError:  # pragma: no cover
    _trace = None  # type: ignore[assignment]
    _OTEL_IMPORTED = False


def init_tracing(service_name: str = "rag-api", force: bool = False) -> bool:
    """Configure the tracer provider. Returns True if tracing is active.

    Idempotent; safe to call from app startup + tests.
    """
    global _INITIALISED, _ENABLED
    if _INITIALISED and not force:
        return _ENABLED

    if not _OTEL_IMPORTED:
        _INITIALISED = True
        _ENABLED = False
        return False

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        # No endpoint -> no-op. Still install a default TracerProvider so
        # get_tracer() returns a real tracer; spans just don't export anywhere.
        _trace.set_tracer_provider(TracerProvider(resource=Resource.create({"service.name": service_name})))
        _INITIALISED = True
        _ENABLED = False
        return False

    # Pick exporter protocol based on env (default http/protobuf; avoids grpc dep).
    protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    _trace.set_tracer_provider(provider)

    _INITIALISED = True
    _ENABLED = True
    return True


def is_enabled() -> bool:
    return _ENABLED


def get_tracer(name: str) -> Any:
    """Return a tracer. Always works -- returns a no-op tracer if OTel unavailable."""
    if not _OTEL_IMPORTED:
        return _NoopTracer()
    if not _INITIALISED:
        init_tracing()
    return _trace.get_tracer(name)


class _NoopSpan:
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def set_attribute(self, *_a, **_kw): pass
    def set_status(self, *_a, **_kw): pass
    def record_exception(self, *_a, **_kw): pass


class _NoopTracer:
    def start_as_current_span(self, *_a, **_kw): return _NoopSpan()
    def start_span(self, *_a, **_kw): return _NoopSpan()


def reset_for_tests() -> None:
    global _INITIALISED, _ENABLED
    _INITIALISED = False
    _ENABLED = False
