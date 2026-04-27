"""API key auth + per-request auth context.

Design:
- Only sha256 hashes are stored in the ``api_keys`` table. Raw keys are
  shown once at creation time (see ``scripts/create_api_key.py``).
- FastAPI dependency ``require_api_key`` reads ``Authorization: Bearer <key>``,
  hashes it, looks up the row, and returns an ``AuthContext`` with tenant_id.
- When ``settings.require_api_key`` is False (dev default), missing/invalid
  keys fall back to the anonymous tenant so existing tests and local use
  keep working. Production flips the env var.

Why not a global middleware? A dependency is better because OpenAPI docs
show the auth requirement per-endpoint, and specific routes (``/health``,
``/metrics``) can skip it by simply not declaring the dependency.
"""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select

from rag.config import settings
from rag.db.base import SessionLocal
from rag.db.models import ApiKey
from rag.logging import bind_tenant_id


@dataclass
class AuthContext:
    tenant_id: str
    api_key_id: int | None = None
    anonymous: bool = False

    @property
    def is_authenticated(self) -> bool:
        return not self.anonymous


def hash_key(raw_key: str) -> str:
    """Hex-encoded sha256 of the raw key. Deterministic; never store the raw key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_raw_key(prefix: str = "rag") -> str:
    """Generate a fresh API key. Show this to the user once, then hash + store."""
    return f"{prefix}_{secrets.token_urlsafe(32)}"


def _extract_bearer(request: Request) -> str | None:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth:
        return None
    parts = auth.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def _lookup_key(raw: str) -> ApiKey | None:
    h = hash_key(raw)
    with SessionLocal() as s:
        row = s.execute(
            select(ApiKey).where(ApiKey.key_hash == h, ApiKey.disabled_at.is_(None))
        ).scalar_one_or_none()
        if row is None:
            return None
        # Touch last_used_at opportunistically; don't block the response if it fails.
        try:
            row.last_used_at = datetime.now(timezone.utc)
            s.commit()
        except Exception:
            s.rollback()
        return row


def require_api_key(request: Request) -> AuthContext:
    """FastAPI dependency. Returns an AuthContext on success, raises 401 otherwise.

    Respects ``settings.require_api_key``: when False, missing/invalid keys
    are treated as the anonymous tenant (no 401). This keeps dev + pytest
    frictionless. Explicit valid keys are still honoured in dev.
    """
    raw = _extract_bearer(request)
    if raw:
        row = _lookup_key(raw)
        if row is not None:
            ctx = AuthContext(tenant_id=row.tenant_id, api_key_id=row.id, anonymous=False)
            request.state.auth = ctx
            bind_tenant_id(ctx.tenant_id)
            return ctx
        if settings.require_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid or disabled API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    if settings.require_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    ctx = AuthContext(tenant_id=settings.anonymous_tenant_id, api_key_id=None, anonymous=True)
    request.state.auth = ctx
    bind_tenant_id(ctx.tenant_id)
    return ctx


# Common typing alias for endpoint signatures:
#   def handler(..., auth: AuthDep):
AuthDep = Depends(require_api_key)
