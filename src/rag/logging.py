"""Structured logging setup with per-request trace_id propagation.

Usage:
    from rag.logging import configure_logging, get_logger, bind_trace_id

    configure_logging()           # call once at startup
    log = get_logger(__name__)
    log.info("retrieval_done", kb=kb_id, hits=5, latency_ms=423)

Trace ID:
    The API middleware (rag_api.main) calls ``bind_trace_id(trace_id)`` for
    each inbound request. All subsequent log lines in the same request
    automatically carry that trace_id, including calls made inside sync
    code paths via contextvars. Downstream components (RAG pipeline, Agent
    nodes, LLM client) inherit it for free.
"""
from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar

import structlog

from .config import settings


_trace_id_ctx: ContextVar[str | None] = ContextVar("trace_id", default=None)
_tenant_id_ctx: ContextVar[str | None] = ContextVar("tenant_id", default=None)


def _add_trace_id(_, __, event_dict):
    tid = _trace_id_ctx.get()
    if tid:
        event_dict.setdefault("trace_id", tid)
    tn = _tenant_id_ctx.get()
    if tn:
        event_dict.setdefault("tenant_id", tn)
    return event_dict


def bind_trace_id(trace_id: str | None = None) -> str:
    """Set the current request's trace_id. Returns the value actually bound."""
    if not trace_id:
        trace_id = uuid.uuid4().hex
    _trace_id_ctx.set(trace_id)
    return trace_id


def bind_tenant_id(tenant_id: str | None) -> None:
    """Set the current request's tenant_id so downstream code can attribute cost."""
    _tenant_id_ctx.set(tenant_id)


def current_trace_id() -> str | None:
    return _trace_id_ctx.get()


def current_tenant_id() -> str | None:
    return _tenant_id_ctx.get()


def configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_trace_id,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.log_format == "json":
        renderer = structlog.processors.JSONRenderer(ensure_ascii=False)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None):
    return structlog.get_logger(name)
