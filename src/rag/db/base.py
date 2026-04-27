"""SQLAlchemy setup.

Single engine/session factory, configured from ``settings.database_url``.
Supports both ``sqlite:///...`` (local dev) and ``postgresql+psycopg://...``
(Docker compose / prod). The ``init_engine`` helper makes tests easy --
pass an in-memory SQLite URL and get a fresh engine.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from ..config import settings


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def init_engine(url: str | None = None) -> Engine:
    """Create (or replace) the engine + session factory."""
    global _engine, _SessionLocal
    db_url = url or settings.database_url
    kwargs: dict = {"future": True}
    if db_url.startswith("sqlite"):
        # check_same_thread=False is safe with per-request sessions, and lets
        # the same sqlite file work across FastAPI worker threads.
        kwargs["connect_args"] = {"check_same_thread": False}
        # StaticPool shares a single in-memory DB across sessions. Without
        # this every sqlite:///:memory: connection opens its own empty DB
        # and nothing you write in one session is visible in the next.
        if ":memory:" in db_url:
            kwargs["poolclass"] = StaticPool
    _engine = create_engine(db_url, **kwargs)
    if db_url.startswith("sqlite"):
        # Enable foreign keys on SQLite (off by default).
        @event.listens_for(_engine, "connect")
        def _on_connect(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA foreign_keys=ON")
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, autoflush=False)
    return _engine


# Eager init so ``from rag.db import engine`` works at import time.
engine: Engine = init_engine()
SessionLocal: sessionmaker[Session] = _SessionLocal  # type: ignore[assignment]


@contextmanager
def get_session() -> Iterator[Session]:
    """Contextmanager for imperative callers (scripts, agent specialists).

    FastAPI routes should use a dependency instead (see rag_api).
    """
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
