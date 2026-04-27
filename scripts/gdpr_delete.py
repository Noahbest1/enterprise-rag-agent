"""GDPR "right to erasure" — delete all data tied to a user_id.

Scope of deletion:
    users               — the row itself
    orders + order_items — via the existing FK cascade
    return_requests     — follows orders via FK cascade (already set)
    feedback            — WHERE user_id = ?
    audit_logs          — WHERE user_id = ?  (http_request rows with no user_id
                          are untouched; we don't blanket-purge an IP, just the
                          identifiable user strand)
    api_keys            — not user-scoped (tenant-scoped), left alone

What this does NOT do:
    - remove user data from already-indexed KB vectors. If the user's personal
      info was ingested into a KB (shouldn't happen because PH2 PII redacts
      at ingest), you need a separate KB-level purge.
    - remove Prometheus counters -- those are aggregated and non-identifying.

Usage:
    python scripts/gdpr_delete.py <user_id> [--dry-run]

Exits 1 if the user does not exist (no-op, but a hard signal for ops).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import delete

from rag.db.base import SessionLocal
from rag.db.models import AuditLog, Feedback, Order, OrderItem, ReturnRequest, User


def purge_user(user_id: str, dry_run: bool = False) -> dict:
    """Return a dict of per-table delete counts. Nothing is removed when dry_run=True."""
    counts = {"users": 0, "orders": 0, "feedback": 0, "audit_logs": 0}
    with SessionLocal() as s:
        user = s.get(User, user_id)
        if user is None:
            return counts  # no-op; caller decides if that's an error

        order_ids = [o.id for o in user.orders]
        counts["orders"] = len(order_ids)
        counts["feedback"] = s.query(Feedback).filter(Feedback.user_id == user_id).count()
        counts["audit_logs"] = s.query(AuditLog).filter(AuditLog.user_id == user_id).count()

        if dry_run:
            counts["users"] = 1  # would delete
            return counts

        # Hard deletes. The ORM `relationship` lacks ``cascade="all, delete"``,
        # so we can't rely on `s.delete(user)` to cascade -- it would SET NULL
        # on orders.user_id and hit the NOT NULL constraint. Delete children
        # explicitly, bottom-up.
        if order_ids:
            s.execute(delete(OrderItem).where(OrderItem.order_id.in_(order_ids)))
            s.execute(delete(ReturnRequest).where(ReturnRequest.order_id.in_(order_ids)))
            s.execute(delete(Order).where(Order.user_id == user_id))
        s.execute(delete(Feedback).where(Feedback.user_id == user_id))
        s.execute(delete(AuditLog).where(AuditLog.user_id == user_id))
        s.delete(user)
        s.commit()
        counts["users"] = 1

    # Write a single audit record that the purge happened (no user_id, since
    # that was just deleted; record the action and a hash of the user_id).
    import hashlib
    from rag_api.audit import record_audit
    record_audit(
        event_type="gdpr_delete",
        user_id=None,  # deliberately not stored
        extra={
            "user_id_hash": hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16],
            "deleted": counts,
        },
    )
    return counts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("user_id")
    p.add_argument("--dry-run", action="store_true", help="Report what would be deleted without deleting.")
    args = p.parse_args()

    counts = purge_user(args.user_id, dry_run=args.dry_run)
    if not counts["users"]:
        print(f"[gdpr] user {args.user_id!r} not found — nothing to delete.", file=sys.stderr)
        return 1

    label = "would delete" if args.dry_run else "deleted"
    print(f"[gdpr] {label} for user_id={args.user_id}:")
    for table, n in counts.items():
        print(f"  {table:<14} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
