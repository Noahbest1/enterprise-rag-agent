"""Account / address tools (DB-backed).

Security posture on mutations:
- Adding a new shipping address: OK directly (low risk).
- Setting a new default address: OK directly.
- Changing the account phone: requires SMS verification in production. Here
  we DON'T actually change it -- we return a ``pending_verification`` record
  that the specialist surfaces to the user. The verification dance is out
  of scope for this demo.
- Deleting addresses: deliberately not exposed. If the user needs to remove
  an address, they go through the website, which has a clearer confirmation
  UI (and a backup of the old address).
"""
from __future__ import annotations

from sqlalchemy import select, update

from rag.db.base import SessionLocal
from rag.db.models import User, UserAddress


# ---------- profile ----------

def _user_to_dict(u: User) -> dict:
    # NOTE: phone is a PII field. Callers (specialist, UI) are expected to mask it.
    return {
        "id": u.id,
        "tenant": u.tenant,
        "display_name": u.display_name,
        "phone": u.phone,
        "created_at": u.created_at.isoformat() if u.created_at else None,
    }


def get_user_profile(user_id: str) -> dict | None:
    with SessionLocal() as s:
        u = s.get(User, user_id)
        return _user_to_dict(u) if u else None


def mask_phone(phone: str | None) -> str | None:
    """138****1111 — safe to show in UI / audit extras."""
    if not phone:
        return phone
    p = phone.strip()
    if len(p) < 7:
        return "****"
    return p[:3] + "*" * (len(p) - 7) + p[-4:]


# ---------- addresses ----------

def _address_to_dict(a: UserAddress) -> dict:
    return {
        "id": a.id,
        "user_id": a.user_id,
        "label": a.label,
        "recipient": a.recipient,
        "phone": a.phone,
        "phone_masked": mask_phone(a.phone),
        "province": a.province,
        "city": a.city,
        "district": a.district,
        "line1": a.line1,
        "is_default": bool(a.is_default),
        "created_at": a.created_at.isoformat() if a.created_at else None,
    }


def list_addresses(user_id: str) -> list[dict]:
    with SessionLocal() as s:
        rows = s.execute(
            select(UserAddress).where(UserAddress.user_id == user_id)
            .order_by(UserAddress.is_default.desc(), UserAddress.id.asc())
        ).scalars().all()
        return [_address_to_dict(a) for a in rows]


def get_default_address(user_id: str) -> dict | None:
    with SessionLocal() as s:
        a = s.execute(
            select(UserAddress).where(
                UserAddress.user_id == user_id,
                UserAddress.is_default.is_(True),
            ).limit(1)
        ).scalar_one_or_none()
        if a is not None:
            return _address_to_dict(a)
        # fall back to "any address"
        any_ = s.execute(
            select(UserAddress).where(UserAddress.user_id == user_id).limit(1)
        ).scalar_one_or_none()
        return _address_to_dict(any_) if any_ else None


def add_address(
    *,
    user_id: str,
    label: str,
    recipient: str,
    phone: str,
    line1: str,
    province: str | None = None,
    city: str | None = None,
    district: str | None = None,
    make_default: bool = False,
) -> dict:
    with SessionLocal() as s:
        if make_default:
            s.execute(
                update(UserAddress).where(UserAddress.user_id == user_id)
                .values(is_default=False)
            )
        a = UserAddress(
            user_id=user_id, label=label, recipient=recipient, phone=phone,
            province=province, city=city, district=district, line1=line1,
            is_default=make_default,
        )
        s.add(a)
        s.commit()
        s.refresh(a)
        return _address_to_dict(a)


def set_default_address(user_id: str, address_id: int) -> dict | None:
    with SessionLocal() as s:
        target = s.get(UserAddress, address_id)
        if target is None or target.user_id != user_id:
            return None
        s.execute(
            update(UserAddress).where(UserAddress.user_id == user_id)
            .values(is_default=False)
        )
        target.is_default = True
        s.commit()
        s.refresh(target)
        return _address_to_dict(target)


# ---------- phone change (mock verification) ----------

def request_phone_change(user_id: str, new_phone: str) -> dict:
    """Do NOT mutate User.phone. Return a pending record that the specialist
    surfaces as 'please enter the SMS code'. Real rollout would persist this
    to a short-TTL store (Redis) and pair with an SMS send.

    ``new_phone_raw`` IS returned so the interactive front-end can submit it
    back together with the 6-digit code. This is a demo compromise; a real
    rollout would either key the pending change by a session token that the
    server itself holds, or accept ``(user_id, code)`` and look up the
    pending new number from a server-side TTL cache -- never round-trip the
    raw phone through the client again.
    """
    with SessionLocal() as s:
        u = s.get(User, user_id)
        if u is None:
            return {"error": "user_not_found"}
        return {
            "status": "pending_verification",
            "user_id": user_id,
            "current_phone_masked": mask_phone(u.phone),
            "new_phone_masked": mask_phone(new_phone),
            "new_phone_raw": new_phone,
            "message": "已向当前手机号发送验证码(mock),请回复 6 位验证码完成变更。",
        }
