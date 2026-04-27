"""PH5.2 OpenTelemetry tracing tests.

Covers:
- init_tracing() is idempotent and respects OTEL_EXPORTER_OTLP_ENDPOINT env
- get_tracer() always returns a usable tracer (no-op when OTel unavailable)
- Span context manager works without raising even when tracing is disabled
- set_attribute / record_exception are safe on no-op spans
"""
from __future__ import annotations

import os

import pytest

from rag.observability import tracing


@pytest.fixture(autouse=True)
def _reset():
    tracing.reset_for_tests()
    prev = os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
    yield
    if prev is not None:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = prev
    tracing.reset_for_tests()


def test_init_without_endpoint_is_noop():
    enabled = tracing.init_tracing()
    assert enabled is False
    assert tracing.is_enabled() is False


def test_init_is_idempotent():
    tracing.init_tracing()
    tracing.init_tracing()
    assert tracing.is_enabled() is False


def test_init_with_endpoint_enables(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    enabled = tracing.init_tracing(force=True)
    assert enabled is True
    assert tracing.is_enabled() is True


def test_get_tracer_when_disabled():
    tracing.init_tracing()  # no endpoint -> disabled
    tracer = tracing.get_tracer("test")
    # With SDK imported but no endpoint, we install a default TracerProvider,
    # so the returned tracer is a real (but non-exporting) one.
    with tracer.start_as_current_span("test-span") as span:
        span.set_attribute("foo", "bar")
        span.set_attribute("n", 1)


def test_span_survives_exception_path():
    tracer = tracing.get_tracer("test")
    with tracer.start_as_current_span("may-fail") as span:
        try:
            raise ValueError("boom")
        except ValueError as e:
            span.record_exception(e)
            # swallow -- test asserts no crash


def test_get_tracer_without_init_auto_inits():
    tracing.reset_for_tests()
    tracer = tracing.get_tracer("auto")
    assert tracer is not None
    with tracer.start_as_current_span("auto-span"):
        pass
