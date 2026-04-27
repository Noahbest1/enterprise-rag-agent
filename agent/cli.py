"""CLI smoke test for the agent.

Usage:
    python -m agent.cli jd "iPhone 16 Pro 多少钱"
    python -m agent.cli taobao "88VIP 怎么开通"
    python -m agent.cli jd --json "7 天无理由退货有哪些不支持的商品"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Make ``rag`` importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent.graph import build_graph
from agent.state import AgentState


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tenant", choices=["jd", "taobao"])
    parser.add_argument("query")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--user-id", default=None,
                        help="Defaults to the tenant's demo user (jd-demo-user / tb-demo-user)")
    args = parser.parse_args()

    graph = build_graph()

    default_user = {"jd": "jd-demo-user", "taobao": "tb-demo-user"}[args.tenant]
    initial_state: AgentState = {
        "tenant": args.tenant,
        "user_id": args.user_id or default_user,
        "messages": [{"role": "user", "content": args.query}],
        "entities": {},
        "step_results": {},
        "trace": [],
    }

    result = graph.invoke(initial_state)

    if args.json:
        safe = {
            "tenant": result.get("tenant"),
            "query": args.query,
            "plan": result.get("plan"),
            "step_results": {
                str(k): {
                    "agent": v.get("agent") if isinstance(v, dict) else None,
                    "answer_preview": (v.get("answer") or "")[:180] if isinstance(v, dict) else None,
                    "citations_count": len(v.get("citations") or []) if isinstance(v, dict) else 0,
                    "abstain": v.get("abstain") if isinstance(v, dict) else None,
                }
                for k, v in (result.get("step_results") or {}).items()
            },
            "final_answer": result.get("final_answer"),
            "trace": result.get("trace"),
        }
        print(json.dumps(safe, ensure_ascii=False, indent=2))
        return

    plan = result.get("plan") or []
    print(f"[tenant] {args.tenant}")
    print(f"[query] {args.query}")
    print()
    print(f"=== Plan ({len(plan)} step) ===")
    for s in plan:
        deps = f" depends_on={s['depends_on']}" if s["depends_on"] else ""
        print(f"  [{s['step_id']}] {s['agent']}: {s['query']}{deps}  ({s['status']})")
    print()
    print("=== Step results ===")
    for step_id in sorted(result.get("step_results") or {}):
        r = result["step_results"][step_id]
        agent = r.get("agent") if isinstance(r, dict) else "?"
        abstain = r.get("abstain") if isinstance(r, dict) else None
        cites = len(r.get("citations") or []) if isinstance(r, dict) else 0
        print(f"  [{step_id}] {agent}  cites={cites}  abstain={abstain}")
    print()
    print("=== Final answer ===")
    print(result.get("final_answer", "(no answer)"))


if __name__ == "__main__":
    main()
