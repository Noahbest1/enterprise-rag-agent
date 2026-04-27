"""Regression gate for CI.

Runs retrieval eval on the full dataset and answer eval on a 20-row subset,
compares against data/eval/baseline.json, and exits 1 if any tracked metric
dropped by more than the baseline's tolerance (default 0.02 = 2pp).

Usage:
    python scripts/regression_gate.py                # retrieval + answer
    python scripts/regression_gate.py --offline      # retrieval only (no LLM calls)
    python scripts/regression_gate.py --tolerance 0.03

Automatically goes offline when DASHSCOPE_API_KEY is unset, so CI without
secrets still catches retrieval regressions.

Exit codes:
    0 -- all metrics within tolerance (or above baseline)
    1 -- one or more metrics dropped beyond tolerance
    2 -- bad config (missing baseline, eval file, etc.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.eval.retrieval_eval import evaluate_retrieval, load_eval_jsonl


DEFAULT_ROWS = ROOT / "data" / "eval" / "eval_rows.jsonl"
DEFAULT_BASELINE = ROOT / "data" / "eval" / "baseline.json"
DEFAULT_RESULTS = ROOT / "data" / "eval" / "results" / "latest.json"

ANSWER_SUBSET_SIZE = 20


def _pick_answer_subset(rows: list[dict], size: int) -> list[dict]:
    """Direct + multi_hop + adversarial, skip out_of_domain. First `size`."""
    scored = [r for r in rows if r.get("category", "direct") != "out_of_domain"]
    return scored[:size]


def _compare(name: str, current: float, baseline: float, tolerance: float) -> tuple[bool, str]:
    delta = current - baseline
    if delta >= -tolerance:
        status = "ok" if delta >= 0 else "within_tol"
        return True, f"  [{status:<10}] {name:<20} {current:.3f}  (baseline {baseline:.3f}, {delta:+.3f})"
    return False, f"  [REGRESSED ] {name:<20} {current:.3f}  (baseline {baseline:.3f}, {delta:+.3f})"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default=str(DEFAULT_ROWS))
    p.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    p.add_argument("--out", default=str(DEFAULT_RESULTS))
    p.add_argument("--tolerance", type=float, default=None,
                   help="Max allowed drop per metric (overrides baseline's _tolerance)")
    p.add_argument("--offline", action="store_true",
                   help="Skip answer eval (no LLM calls). Auto-enabled when DASHSCOPE_API_KEY is unset.")
    args = p.parse_args()

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"baseline not found: {baseline_path}", file=sys.stderr)
        return 2
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    tolerance = args.tolerance if args.tolerance is not None else baseline.get("_tolerance", 0.02)

    rows_path = Path(args.rows)
    if not rows_path.exists():
        print(f"eval rows not found: {rows_path}", file=sys.stderr)
        return 2
    rows = load_eval_jsonl(rows_path)
    print(f"Loaded {len(rows)} eval rows from {rows_path}")
    print(f"Baseline: {baseline_path} (tolerance ±{tolerance:.3f})")
    print()

    offline = args.offline or not os.environ.get("DASHSCOPE_API_KEY")
    if offline and not args.offline:
        print("No DASHSCOPE_API_KEY in env -> offline mode (retrieval only)")
        print()

    all_ok = True
    current: dict = {"tolerance": tolerance}

    # --- retrieval ---
    # Keep the gate deterministic: turn off LLM-based query rewrite. rewrite
    # runs at temperature>0 so its output (and therefore BM25/vector matches)
    # shifts slightly between runs, which would make the gate flaky.
    # Baseline numbers are measured under the same flags.
    rm = evaluate_retrieval(rows, use_rewrite=False)
    print(f"=== Retrieval (n={rm.n}, skipped={rm.n_skipped}, rewrite=off) ===")
    retrieval_current = {
        "hit_at_1": rm.hit_at_1,
        "hit_at_3": rm.hit_at_3,
        "hit_at_5": rm.hit_at_5,
        "mrr_at_10": rm.mrr_at_10,
    }
    retrieval_baseline = baseline.get("retrieval", {})
    for k, v in retrieval_current.items():
        if k in retrieval_baseline:
            ok, line = _compare(f"retrieval.{k}", v, retrieval_baseline[k], tolerance)
            print(line)
            all_ok = all_ok and ok
        else:
            print(f"  [no-baseline] retrieval.{k:<12} {v:.3f}")
    current["retrieval"] = retrieval_current
    current["retrieval_by_category"] = {
        cat: {"n": cm.n, "hit_at_1": cm.hit_at_1, "hit_at_3": cm.hit_at_3,
              "hit_at_5": cm.hit_at_5, "mrr_at_10": cm.mrr_at_10}
        for cat, cm in rm.by_category.items()
    }

    # --- answer (optional) ---
    if not offline:
        print()
        print(f"=== Answer (subset of {ANSWER_SUBSET_SIZE}) ===")
        # Import here so offline runs don't pay the cost of pulling in the
        # LLM client / pipeline (it imports requests, llm_client, etc.).
        from rag.eval.answer_eval import evaluate_answers
        subset = _pick_answer_subset(rows, ANSWER_SUBSET_SIZE)
        am = evaluate_answers(subset)
        answer_current = {
            "faithful": am.faithful,
            "cites_correctly": am.cites_correctly,
            "answers_question": am.answers_question,
            "abstain_correct": am.abstain_correct,
        }
        answer_baseline = baseline.get("answer", {})
        for k, v in answer_current.items():
            if k in answer_baseline:
                ok, line = _compare(f"answer.{k}", v, answer_baseline[k], tolerance)
                print(line)
                all_ok = all_ok and ok
            else:
                print(f"  [no-baseline] answer.{k:<12} {v:.3f}")
        current["answer"] = answer_current
    else:
        current["answer"] = {"_skipped": "offline"}

    # --- write latest.json ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    current["_timestamp"] = datetime.now(timezone.utc).isoformat()
    current["_offline"] = offline
    out_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {out_path}")

    print()
    if all_ok:
        print("REGRESSION GATE: PASS")
        return 0
    print("REGRESSION GATE: FAIL (see REGRESSED lines above)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
