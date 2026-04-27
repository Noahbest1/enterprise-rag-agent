"""PH5.1 Prometheus metrics tests.

Covers:
- /metrics endpoint returns text exposition format
- HTTP middleware increments rag_requests_total and observes duration
- record_llm() increments calls + tokens counters
- cache_get() triggers record_cache() with the correct layer label
- key-prefix → layer mapping
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from rag.cache import _layer_from_key, cache_get, cache_set
from rag.observability import (
    is_enabled,
    metrics_payload,
    record_cache,
    record_llm,
    record_request,
    reset_for_tests,
)


@pytest.fixture(autouse=True)
def _wipe_metrics():
    reset_for_tests()
    yield
    reset_for_tests()


@pytest.fixture
def client():
    # Import inside the fixture so the app picks up a fresh metrics state
    # after reset_for_tests().
    from rag_api.main import app
    return TestClient(app)


def _metrics_text() -> str:
    body, _ = metrics_payload()
    return body.decode("utf-8")


def test_metrics_library_available():
    assert is_enabled(), "prometheus-client must be installed (see requirements-api.txt)"


def test_metrics_endpoint_exposition_format(client: TestClient):
    r = client.get("/metrics")
    assert r.status_code == 200
    # Prometheus text format content-type (includes "version=0.0.4")
    assert "text/plain" in r.headers["content-type"]
    # Metric descriptors are always present even before any sample.
    assert "rag_requests_total" in r.text
    assert "rag_request_duration_seconds" in r.text
    assert "rag_llm_tokens_total" in r.text
    assert "rag_cache_events_total" in r.text


def test_http_middleware_increments_request_counter(client: TestClient):
    # Trigger an HTTP call; middleware should record it.
    r = client.get("/health")
    assert r.status_code == 200

    text = _metrics_text()
    # The path label is the route template; /health is a fixed path.
    assert 'rag_requests_total{method="GET",path="/health",status="200"} 1.0' in text
    assert 'rag_request_duration_seconds_count{method="GET",path="/health"}' in text


def test_record_request_direct():
    record_request("POST", "/answer", 200, 1.2)
    record_request("POST", "/answer", 500, 0.3)
    text = _metrics_text()
    assert 'rag_requests_total{method="POST",path="/answer",status="200"} 1.0' in text
    assert 'rag_requests_total{method="POST",path="/answer",status="500"} 1.0' in text


def test_record_llm_tokens():
    record_llm(task="generate", input_tokens=300, output_tokens=120, status="ok")
    record_llm(task="rewrite", input_tokens=40, output_tokens=20, status="ok")
    record_llm(task="generate", input_tokens=0, output_tokens=0, status="error")

    text = _metrics_text()
    assert 'rag_llm_tokens_total{kind="input",task="generate"} 300.0' in text
    assert 'rag_llm_tokens_total{kind="output",task="generate"} 120.0' in text
    assert 'rag_llm_tokens_total{kind="input",task="rewrite"} 40.0' in text
    assert 'rag_llm_calls_total{status="ok",task="generate"} 1.0' in text
    assert 'rag_llm_calls_total{status="error",task="generate"} 1.0' in text
    # No tokens recorded on the error call -> should not appear.
    assert 'rag_llm_tokens_total{kind="input",task="generate"} 301' not in text


def test_record_cache_direct():
    record_cache("llm", hit=True)
    record_cache("llm", hit=False)
    record_cache("embedding", hit=True)

    text = _metrics_text()
    assert 'rag_cache_events_total{layer="llm",result="hit"} 1.0' in text
    assert 'rag_cache_events_total{layer="llm",result="miss"} 1.0' in text
    assert 'rag_cache_events_total{layer="embedding",result="hit"} 1.0' in text


@pytest.mark.parametrize(
    "key,expected_layer",
    [
        ("llm:abc123", "llm"),
        ("emb:bge-m3:def456", "embedding"),
        ("semcache:jd_demo", "semantic"),
        ("sem:foo", "semantic"),
        ("vision:v1:describe:hash", "vision"),
        ("weird-no-prefix", "other"),
    ],
)
def test_layer_from_key(key: str, expected_layer: str):
    assert _layer_from_key(key) == expected_layer


def test_cache_get_records_metric_on_miss_then_hit():
    async def run():
        # Miss
        got = await cache_get("llm:ph5test:nonexistent")
        assert got is None
        # Hit
        await cache_set("llm:ph5test:set1", "value-v1", ttl_seconds=60)
        got = await cache_get("llm:ph5test:set1")
        assert got == "value-v1"

    asyncio.run(run())

    text = _metrics_text()
    assert 'rag_cache_events_total{layer="llm",result="miss"}' in text
    assert 'rag_cache_events_total{layer="llm",result="hit"}' in text
