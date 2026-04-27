"""Collect thumbs-down feedback into a regression-candidate JSONL.

Pulls rows from the ``feedback`` table where verdict='down' and writes
each as a candidate for the eval set. Manual invocation only for now --
the operator reviews the output and cherry-picks rows into
``data/eval/eval_rows.jsonl`` (adding the ``relevant_source_ids`` they
believe is correct).

Usage:
    python scripts/collect_thumbs_down.py
    python scripts/collect_thumbs_down.py --since 2026-04-01
    python scripts/collect_thumbs_down.py --kb jd_demo --out data/eval/jd_down.jsonl

Output schema (one row per line):
    {"query", "kb_id", "trace_id", "created_at", "reason"}
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import select

from rag.db.base import SessionLocal
from rag.db.models import Feedback


DEFAULT_OUT = ROOT / "data" / "eval" / "regression_candidates.jsonl"


def _parse_since(s: str | None) -> datetime | None:
    if not s:
        return None
    # Accept either YYYY-MM-DD or a full ISO string.
    try:
        if len(s) == 10:
            dt = datetime.strptime(s, "%Y-%m-%d")
        else:
            dt = datetime.fromisoformat(s)
    except ValueError as e:
        raise SystemExit(f"Bad --since value: {s!r} ({e})")
    # Treat naive as UTC so comparisons with DB timestamps are unambiguous.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--since", help="ISO date/datetime lower bound (inclusive), e.g. 2026-04-01")
    p.add_argument("--kb", help="Filter to a single kb_id")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output JSONL path")
    p.add_argument("--limit", type=int, default=1000, help="Max candidates to write")
    args = p.parse_args()

    since = _parse_since(args.since)

    stmt = select(Feedback).where(Feedback.verdict == "down")
    if args.kb:
        stmt = stmt.where(Feedback.kb_id == args.kb)
    if since is not None:
        stmt = stmt.where(Feedback.created_at >= since)
    # Group by kb_id, newest first inside each kb.
    stmt = stmt.order_by(Feedback.kb_id.asc(), Feedback.created_at.desc()).limit(args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as session:
        rows = session.execute(stmt).scalars().all()
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                rec = {
                    "query": r.query,
                    "kb_id": r.kb_id,
                    "trace_id": r.trace_id,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "reason": r.reason,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Per-kb counts for quick operator visibility.
    by_kb: dict[str, int] = {}
    for r in rows:
        by_kb[r.kb_id] = by_kb.get(r.kb_id, 0) + 1

    print(f"Wrote {len(rows)} candidates to {out_path}")
    if by_kb:
        print("by kb:")
        for kb, n in sorted(by_kb.items()):
            print(f"  {kb:<20} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
