"""Long-term user preferences — the 4th layer of the agent memory stack.

Memory layers:
    1. In-turn messages (LangGraph state.messages)
    2. In-turn entities (LangGraph state.entities) — last_order_id etc.
    3. Cross-session checkpoint (AsyncSqliteSaver, keyed on thread_id)
    4. **THIS** — durable user-level facts that survive sessions / restarts

Examples this layer is for:
    - "把 JD 快递设为默认"  → key=preferred_carrier, value=jd-express
    - "我对牛奶过敏"        → key=allergen, value=lactose
    - "我只看英文邮件"      → key=lang_preference, value=en
    - "周末别打电话"        → key=do_not_disturb_window, value=sat-sun

The Planner reads a compact summary of these preferences and injects it
into its system prompt so every plan starts already knowing the user.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from rag.db.base import SessionLocal
from rag.db.models import UserPreference


def _to_dict(p: UserPreference) -> dict[str, Any]:
    return {
        "id": p.id,
        "user_id": p.user_id,
        "key": p.key,
        "value": p.value,
        "source": p.source,
        "updated_at": _iso_utc(p.updated_at),
    }


def _iso_utc(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def set_preference(
    user_id: str, key: str, value: str, *, source: str = "user"
) -> dict[str, Any]:
    """Upsert one preference. Returns the saved dict."""
    key = (key or "").strip().lower()[:64]
    value = (value or "").strip()
    if not key or not value:
        raise ValueError("key and value must both be non-empty")
    with SessionLocal() as s:
        existing = s.execute(
            select(UserPreference).where(
                UserPreference.user_id == user_id,
                UserPreference.key == key,
            )
        ).scalar_one_or_none()
        if existing is None:
            row = UserPreference(user_id=user_id, key=key, value=value, source=source)
            s.add(row)
            s.commit()
            s.refresh(row)
            return _to_dict(row)
        existing.value = value
        existing.source = source
        s.commit()
        s.refresh(existing)
        return _to_dict(existing)


def list_preferences(user_id: str) -> list[dict[str, Any]]:
    with SessionLocal() as s:
        rows = s.execute(
            select(UserPreference).where(UserPreference.user_id == user_id)
            .order_by(UserPreference.key.asc())
        ).scalars().all()
        return [_to_dict(r) for r in rows]


def get_preference(user_id: str, key: str) -> dict[str, Any] | None:
    key = (key or "").strip().lower()[:64]
    with SessionLocal() as s:
        row = s.execute(
            select(UserPreference).where(
                UserPreference.user_id == user_id, UserPreference.key == key
            )
        ).scalar_one_or_none()
        return _to_dict(row) if row else None


def delete_preference(user_id: str, key: str) -> bool:
    key = (key or "").strip().lower()[:64]
    with SessionLocal() as s:
        row = s.execute(
            select(UserPreference).where(
                UserPreference.user_id == user_id, UserPreference.key == key
            )
        ).scalar_one_or_none()
        if row is None:
            return False
        s.delete(row)
        s.commit()
        return True


def render_for_planner(user_id: str, max_items: int = 6) -> str:
    """Compact one-line-per-pref summary the Planner injects into its
    system prompt. Empty string when the user has no preferences yet —
    callers can ``if summary: ...`` cleanly.
    """
    prefs = list_preferences(user_id)
    if not prefs:
        return ""
    # Cap at max_items so the prompt doesn't bloat for power users.
    head = prefs[:max_items]
    lines = [f"  - {p['key']}: {p['value']} (来源: {p['source']})" for p in head]
    extra = len(prefs) - len(head)
    body = "\n".join(lines)
    if extra > 0:
        body += f"\n  - ... 另有 {extra} 项偏好"
    return f"用户长期偏好(跨会话持久化的):\n{body}"
