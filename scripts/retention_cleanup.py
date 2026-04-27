"""Retention cleanup cron.

Default policy (override via CLI flags):
    audit_logs       older than 180 days   -> delete
    feedback         older than 365 days   -> delete
    return_requests  older than 365 days AND status in (completed, rejected, cancelled)

Intended to be run daily from cron / a k8s CronJob. Always logs counts; never
fails silently (exit 0 with 0 deletes is fine; any exception propagates).

Usage:
    python scripts/retention_cleanup.py
    python scripts/retention_cleanup.py --audit-days 90 --feedback-days 180 --dry-run
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import delete

from rag.db.base import SessionLocal
from rag.db.models import AuditLog, Feedback, ReturnRequest


def cleanup(
    *,
    audit_days: int = 180,
    feedback_days: int = 365,
    return_request_days: int = 365,
    dry_run: bool = False,
) -> dict:
    now = datetime.now(timezone.utc)
    counts: dict[str, int] = {}

    with SessionLocal() as s:
        cutoff = now - timedelta(days=audit_days)
        q_audit = s.query(AuditLog).filter(AuditLog.created_at < cutoff)
        counts["audit_logs"] = q_audit.count()
        if not dry_run:
            s.execute(delete(AuditLog).where(AuditLog.created_at < cutoff))

        cutoff = now - timedelta(days=feedback_days)
        q_fb = s.query(Feedback).filter(Feedback.created_at < cutoff)
        counts["feedback"] = q_fb.count()
        if not dry_run:
            s.execute(delete(Feedback).where(Feedback.created_at < cutoff))

        cutoff = now - timedelta(days=return_request_days)
        terminal = ("completed", "rejected", "cancelled")
        q_rr = s.query(ReturnRequest).filter(
            ReturnRequest.created_at < cutoff,
            ReturnRequest.status.in_(terminal),
        )
        counts["return_requests"] = q_rr.count()
        if not dry_run:
            s.execute(
                delete(ReturnRequest).where(
                    ReturnRequest.created_at < cutoff,
                    ReturnRequest.status.in_(terminal),
                )
            )
        if not dry_run:
            s.commit()

    # Audit the cleanup itself (even if dry-run, record the attempt).
    from rag_api.audit import record_audit
    record_audit(
        event_type="retention_cleanup_dry_run" if dry_run else "retention_cleanup",
        extra={
            "audit_days": audit_days, "feedback_days": feedback_days,
            "return_request_days": return_request_days, "counts": counts,
        },
    )
    return counts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--audit-days", type=int, default=180)
    p.add_argument("--feedback-days", type=int, default=365)
    p.add_argument("--return-request-days", type=int, default=365)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    counts = cleanup(
        audit_days=args.audit_days,
        feedback_days=args.feedback_days,
        return_request_days=args.return_request_days,
        dry_run=args.dry_run,
    )
    label = "would delete" if args.dry_run else "deleted"
    print(f"[retention] {label}:")
    for table, n in counts.items():
        print(f"  {table:<20} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
