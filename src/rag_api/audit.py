"""HTTP audit log middleware + business event helper.

Writes one row per request to ``audit_logs`` after the response is composed
(so status_code / latency are accurate). Failures to persist are swallowed
so the audit layer can never take down the main request path.

Body content policy:
- We NEVER store raw query/answer text here. Only sha256[:16] hashes, so an
  investigator can correlate "trace_id X asked the same question as trace_id
  Y" without the audit table becoming a PII honeypot.
- Full text recovery goes through the structured application log, keyed on
  trace_id.

Paths excluded from audit (too noisy, no compliance value):
- ``/health``, ``/metrics``, UI HTML (``/``), favicon.

To record a business event (e.g. "return_request_created") from anywhere in
the codebase::

    from rag_api.audit import record_audit
    record_audit(
        event_type="return_request_created",
        tenant_id=ctx.tenant_id, user_id=user_id,
        extra={"order_id": ..., "refund_cents": ...},
    )
"""
from __future__ import annotations

import hashlib
from typing import Any

from fastapi import Request

from rag.db.base import SessionLocal
from rag.db.models import AuditLog
from rag.logging import get_logger


log = get_logger("audit")

_AUDIT_EXEMPT = {"/health", "/metrics", "/", "/favicon.ico", "/audit"}


def _hash16(text: str | None) -> str | None:
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def record_audit(
    *,
    event_type: str,
    tenant_id: str | None = None,
    user_id: str | None = None,
    trace_id: str | None = None,
    api_key_id: int | None = None,
    method: str | None = None,
    path: str | None = None,
    status_code: int | None = None,
    latency_ms: int | None = None,
    query_hash: str | None = None,
    answer_hash: str | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist one audit row. Never raises."""
    try:
        with SessionLocal() as s:
            row = AuditLog(
                trace_id=trace_id,
                tenant_id=tenant_id,
                api_key_id=api_key_id,
                user_id=user_id,
                event_type=event_type,
                method=method,
                path=path,
                status_code=status_code,
                latency_ms=latency_ms,
                query_hash=query_hash,
                answer_hash=answer_hash,
                error=(error[:500] if error else None),
                extra=extra or {},
            )
            s.add(row)
            s.commit()
    except Exception as e:  # pragma: no cover
        log.warning("audit_persist_failed", event_type=event_type, error=str(e))


async def audit_middleware(request: Request, call_next):
    """ASGI-style middleware. Runs after rate_limit, after trace_and_log."""
    if request.url.path in _AUDIT_EXEMPT:
        return await call_next(request)
    # Resolve route template early; avoids high-cardinality path storage.
    response = await call_next(request)

    auth = getattr(request.state, "auth", None)
    tenant_id = getattr(auth, "tenant_id", None) if auth else None
    api_key_id = getattr(auth, "api_key_id", None) if auth else None

    route = request.scope.get("route")
    path_template = getattr(route, "path", None) or request.url.path
    trace_id = response.headers.get("x-trace-id") or request.headers.get("x-trace-id")
    latency_ms_hdr = response.headers.get("x-latency-ms")
    latency_ms = int(latency_ms_hdr) if latency_ms_hdr and latency_ms_hdr.isdigit() else None

    record_audit(
        event_type="http_request",
        trace_id=trace_id,
        tenant_id=tenant_id,
        api_key_id=api_key_id,
        method=request.method,
        path=path_template,
        status_code=response.status_code,
        latency_ms=latency_ms,
    )
    return response


def record_http_audit_from_middleware(
    *,
    request: Request,
    status_code: int,
    latency_ms: int,
    trace_id: str,
) -> None:
    """Alternative: call from inside the existing trace_and_log middleware so
    we don't double-walk the request. Keeps state.auth access in one place."""
    path = request.url.path
    if path in _AUDIT_EXEMPT:
        return
    route = request.scope.get("route")
    path_template = getattr(route, "path", None) or path
    auth = getattr(request.state, "auth", None)
    tenant_id = getattr(auth, "tenant_id", None) if auth else None
    api_key_id = getattr(auth, "api_key_id", None) if auth else None

    record_audit(
        event_type="http_request",
        trace_id=trace_id,
        tenant_id=tenant_id,
        api_key_id=api_key_id,
        method=request.method,
        path=path_template,
        status_code=status_code,
        latency_ms=latency_ms,
    )


# Re-export the hash helper so callers can pre-hash their own bodies
# (e.g. answer pipeline wants to record answer_hash without persisting text).
hash16 = _hash16
