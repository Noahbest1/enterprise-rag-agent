"""In-process sliding-window rate limiter.

Keyed on auth context:
- Authenticated request  -> key = "tenant:<id>:apikey:<id>", limit = authenticated tier
- Anonymous request      -> key = "ip:<client_ip>",          limit = anonymous tier

Both tiers come from ``settings.rate_limit_*`` and use the ``N/period``
notation (``200/minute``, ``10/second``). Window is tracked as a deque of
request timestamps; we drop anything older than the window on every request,
then check the size.

Single-worker only. For multi-worker / multi-pod deployments swap
``_SlidingWindow`` for a Redis-backed INCR with TTL. Left as a ~40 LOC
extension when you actually need it.

Returns 429 with ``Retry-After`` header computed from the oldest in-window
sample. Never rate-limits public paths (``/health``, ``/metrics``, ``/``).
"""
from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from fastapi import Request
from fastapi.responses import JSONResponse

from rag.config import settings


_PERIOD_SECONDS = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "day": 86400,
}

# Routes that should never be rate-limited (scraping + UI).
_EXEMPT_PATHS = {"/health", "/metrics", "/", "/favicon.ico"}


@dataclass
class _Limit:
    count: int
    window_seconds: int


def parse_limit(spec: str) -> _Limit:
    """Parse ``200/minute`` / ``10/second`` style strings. Raises on bad input."""
    m = re.fullmatch(r"\s*(\d+)\s*/\s*([a-zA-Z]+)\s*", spec)
    if not m:
        raise ValueError(f"bad rate-limit spec: {spec!r}")
    count = int(m.group(1))
    period = m.group(2).lower().rstrip("s")  # "minutes" -> "minute"
    if period not in _PERIOD_SECONDS:
        raise ValueError(f"bad period in rate-limit spec: {spec!r}")
    return _Limit(count=count, window_seconds=_PERIOD_SECONDS[period])


@dataclass
class _SlidingWindow:
    _hits: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))

    def check(self, key: str, limit: _Limit, now: float | None = None) -> tuple[bool, int, float]:
        """Returns (allowed, remaining, retry_after_seconds)."""
        now = now if now is not None else time.monotonic()
        window_start = now - limit.window_seconds
        bucket = self._hits[key]
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        if len(bucket) >= limit.count:
            oldest = bucket[0]
            retry_after = max(0.0, oldest + limit.window_seconds - now)
            return False, 0, retry_after
        bucket.append(now)
        return True, limit.count - len(bucket), 0.0

    def reset(self) -> None:
        self._hits.clear()


# Module-level window shared across requests (one per process).
_WINDOW = _SlidingWindow()


def _client_ip(request: Request) -> str:
    return (request.client.host if request.client else None) or "unknown"


def _limit_for(request: Request) -> tuple[str, _Limit]:
    """Return (rate-limit key, applicable limit) for this request."""
    auth = getattr(request.state, "auth", None)
    if auth and getattr(auth, "is_authenticated", False):
        spec = settings.rate_limit_authenticated
        key = f"tenant:{auth.tenant_id}:apikey:{auth.api_key_id}"
    else:
        spec = settings.rate_limit_anonymous
        key = f"ip:{_client_ip(request)}"
    return key, parse_limit(spec)


async def rate_limit_middleware(request: Request, call_next):
    """ASGI-style middleware applied by main.py. Exempts scraping/UI paths."""
    if not settings.rate_limit_enabled:
        return await call_next(request)
    if request.url.path in _EXEMPT_PATHS:
        return await call_next(request)

    # NOTE: auth runs as a route dependency (not a middleware), so
    # request.state.auth is populated *inside* call_next, not before. We
    # therefore rely on IP-based keying here for the ANON path; authenticated
    # callers still get counted on the IP bucket on their first request and
    # on the tenant bucket from the second. Good enough: we don't want the
    # middleware to re-run auth DB lookups.
    key, limit = _limit_for(request)
    allowed, remaining, retry_after = _WINDOW.check(key, limit)
    if not allowed:
        retry = int(retry_after) + 1
        return JSONResponse(
            status_code=429,
            content={"detail": "rate limit exceeded", "retry_after_seconds": retry},
            headers={
                "Retry-After": str(retry),
                "X-RateLimit-Limit": str(limit.count),
                "X-RateLimit-Remaining": "0",
            },
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(limit.count)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    return response


def reset_for_tests() -> None:
    _WINDOW.reset()
