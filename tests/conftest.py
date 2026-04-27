"""Pytest fixtures.

All tests share a single in-memory SQLite engine (configured via
``DATABASE_URL`` before any rag.* import). Each test gets a fresh schema
via drop_all + create_all -- that's isolation enough without the cost of
swapping engines.

We set the env BEFORE importing rag.db so the module-level
``engine`` / ``SessionLocal`` bind to the test DB, not the dev file.
"""
from __future__ import annotations

import os

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["ENV"] = "test"

import pytest  # noqa: E402

from rag.db.base import Base, SessionLocal, engine  # noqa: E402
from rag.db import models  # noqa: E402,F401  -- registers tables with Base.metadata


@pytest.fixture()
def db_session():
    """Fresh schema each test, reusing the shared in-memory engine."""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def seeded_db(db_session):
    """DB populated via scripts.seed_db.seed()."""
    from scripts import seed_db
    seed_db.seed()
    return db_session
