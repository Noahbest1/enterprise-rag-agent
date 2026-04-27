"""PH5.4 rate limit tests.

Covers:
- parse_limit() accepts N/second N/minute N/hour
- _SlidingWindow allows N in window, rejects N+1, returns retry_after
- Middleware respects rate_limit_enabled=False (default; tests pass freely)
- Middleware returns 429 once limit exceeded (anonymous tier)
- Exempt paths (/health, /metrics) are never limited
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag.config import settings
from rag_api.rate_limit import _SlidingWindow, parse_limit, reset_for_tests


@pytest.fixture(autouse=True)
def _wipe():
    reset_for_tests()
    yield
    reset_for_tests()


@pytest.fixture
def client():
    from rag_api.main import app
    return TestClient(app)


# ---------- unit ----------

def test_parse_limit_seconds():
    lim = parse_limit("5/second")
    assert lim.count == 5 and lim.window_seconds == 1


def test_parse_limit_minutes_plural():
    lim = parse_limit("200/minutes")
    assert lim.count == 200 and lim.window_seconds == 60


def test_parse_limit_hours():
    lim = parse_limit("1000/hour")
    assert lim.count == 1000 and lim.window_seconds == 3600


def test_parse_limit_bad_input():
    with pytest.raises(ValueError):
        parse_limit("abc")
    with pytest.raises(ValueError):
        parse_limit("5/fortnight")


def test_sliding_window_allows_then_blocks():
    w = _SlidingWindow()
    lim = parse_limit("3/minute")
    now = 1000.0
    for _ in range(3):
        allowed, _, _ = w.check("k", lim, now=now)
        assert allowed
    blocked, remaining, retry = w.check("k", lim, now=now)
    assert blocked is False
    assert remaining == 0
    assert retry > 0


def test_sliding_window_isolates_keys():
    w = _SlidingWindow()
    lim = parse_limit("1/minute")
    a_ok, _, _ = w.check("a", lim, now=1000.0)
    b_ok, _, _ = w.check("b", lim, now=1000.0)
    assert a_ok and b_ok  # different keys -> independent buckets


def test_sliding_window_expires_after_period():
    w = _SlidingWindow()
    lim = parse_limit("1/minute")
    assert w.check("k", lim, now=1000.0)[0]
    # Next request inside the window is blocked
    assert w.check("k", lim, now=1030.0)[0] is False
    # After window slides past it, allowed again
    allowed, _, _ = w.check("k", lim, now=1100.0)
    assert allowed


# ---------- middleware ----------

def test_middleware_bypasses_when_disabled(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    # Slam /kbs well beyond the anon limit; nothing should 429.
    for _ in range(40):
        assert client.get("/kbs").status_code == 200


def test_middleware_429s_when_enabled(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_anonymous", "3/minute")
    results = [client.get("/kbs").status_code for _ in range(5)]
    # First 3 succeed, rest 429
    assert results[:3] == [200, 200, 200]
    assert 429 in results[3:]


def test_middleware_exempts_health_and_metrics(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(settings, "rate_limit_anonymous", "1/minute")
    # Fire many; /health and /metrics should never 429.
    for _ in range(5):
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200
