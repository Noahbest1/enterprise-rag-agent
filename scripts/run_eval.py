"""Run retrieval + answer eval on data/eval/eval_rows.jsonl.

Usage:
    python scripts/run_eval.py                 # both
    python scripts/run_eval.py --only retrieval
    python scripts/run_eval.py --only answer
    python scripts/run_eval.py --rows path/to/rows.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.eval.answer_eval import evaluate_answers
from rag.eval.retrieval_eval import evaluate_retrieval, load_eval_jsonl


def print_retrieval(m):
    skipped_note = f" (+{m.n_skipped} out_of_domain skipped)" if m.n_skipped else ""
    print(f"=== Retrieval (n={m.n}{skipped_note}) ===")
    print(f"  Hit@1:   {m.hit_at_1:.3f}")
    print(f"  Hit@3:   {m.hit_at_3:.3f}")
    print(f"  Hit@5:   {m.hit_at_5:.3f}")
    print(f"  MRR@10:  {m.mrr_at_10:.3f}")
    if m.by_category:
        print(f"  by category:")
        for cat, cm in m.by_category.items():
            marker = " [skipped]" if cat == "out_of_domain" else ""
            print(f"    {cat:<15} n={cm.n:<3} Hit@1={cm.hit_at_1:.3f} Hit@3={cm.hit_at_3:.3f} Hit@5={cm.hit_at_5:.3f} MRR={cm.mrr_at_10:.3f}{marker}")
    misses = [r for r in m.per_row if r.scored and not r.hit_at_5]
    if misses:
        print(f"  misses ({len(misses)}):")
        for r in misses:
            print(f"    - [{r.kb_id}] ({r.category}) {r.query}")
            print(f"         expected: {r.relevant_source_ids[:2]}")
            print(f"         got:      {r.retrieved_source_ids[:3]}")


def print_answers(m):
    print(f"=== Answer (n={m.n}) ===")
    print(f"  faithful:          {m.faithful:.3f}")
    print(f"  cites_correctly:   {m.cites_correctly:.3f}")
    print(f"  answers_question:  {m.answers_question:.3f}")
    print(f"  abstain_correct:   {m.abstain_correct:.3f}")
    print(f"  abstain_rate:      {m.abstain_rate:.3f}")
    print(f"  avg latency:       {m.avg_latency_ms:.0f} ms")
    bad = [r for r in m.per_row if not (r.faithful and r.cites_correctly and r.answers_question)]
    if bad:
        print(f"  failures ({len(bad)}):")
        for r in bad:
            print(f"    - [{r.kb_id}] {r.query}")
            print(f"         faithful={r.faithful} cites={r.cites_correctly} answers={r.answers_question}")
            print(f"         answer: {r.answer[:150]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default=str(ROOT / "data" / "eval" / "eval_rows.jsonl"))
    parser.add_argument("--only", choices=["retrieval", "answer"], default=None)
    parser.add_argument("--out", default=str(ROOT / "data" / "eval" / "results"))
    args = parser.parse_args()

    rows = load_eval_jsonl(Path(args.rows))
    print(f"Loaded {len(rows)} eval rows from {args.rows}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.only != "answer":
        rm = evaluate_retrieval(rows)
        print()
        print_retrieval(rm)
        (out_dir / f"retrieval_{stamp}.json").write_text(
            json.dumps(rm.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if args.only != "retrieval":
        am = evaluate_answers(rows)
        print()
        print_answers(am)
        (out_dir / f"answer_{stamp}.json").write_text(
            json.dumps(am.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
