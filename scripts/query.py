"""Run a single query against a KB. Useful for smoke-testing without the API."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.pipeline import answer_query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kb_id")
    parser.add_argument("query")
    parser.add_argument("--no-rewrite", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    ans = answer_query(
        args.query,
        args.kb_id,
        use_rewrite=not args.no_rewrite,
        use_rerank=not args.no_rerank,
    )

    if args.json:
        payload = {
            "query": ans.query,
            "rewritten_query": ans.rewritten_query,
            "answer": ans.text,
            "abstained": ans.abstained,
            "reason": ans.reason,
            "latency_ms": ans.latency_ms,
            "citations": [
                {"n": c.n, "title": c.title, "section_path": c.section_path, "source_id": c.source_id}
                for c in ans.citations
            ],
            "hits": [
                {
                    "chunk_id": h.chunk_id,
                    "score": h.score,
                    "title": h.title,
                    "section_path": h.section_path,
                    "retrieval_source": h.retrieval_source,
                }
                for h in ans.hits
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(f"Q: {ans.query}")
    print(f"[rewritten] {ans.rewritten_query}")
    print(f"[latency] {ans.latency_ms} ms")
    print(f"[abstained] {ans.abstained}  reason={ans.reason or '-'}")
    print()
    print("Answer:")
    print(ans.text)
    print()
    if ans.citations:
        print("Citations:")
        for c in ans.citations:
            print(f"  [{c.n}] {c.title} — {' / '.join(c.section_path)}")
            print(f"       {c.source_path}")
    print()
    print("Top hits:")
    for i, h in enumerate(ans.hits, 1):
        print(f"  {i}. [{h.score:.3f}] ({h.retrieval_source}) {h.title} — {' / '.join(h.section_path)}")


if __name__ == "__main__":
    main()
