"""Incrementally sync a KB with an external source.

Usage:
    # Watch a local directory; add new / update changed / drop removed
    python scripts/sync_connector.py <kb_id> local /path/to/docs

    # Fetch a fixed list of URLs (one per line in a file)
    python scripts/sync_connector.py <kb_id> http /path/to/urls.txt

    # Force backend override (rare)
    python scripts/sync_connector.py my_kb local ./docs --backend qdrant

Re-running this is the mental model for "pull changes". Unchanged docs are
skipped (no LLM, no embedder). Re-embedding cost is proportional to the
changed set, not the corpus size.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kb_id")
    parser.add_argument("connector", choices=["local", "http"])
    parser.add_argument("source", help="directory (for local) or urls-file (for http)")
    parser.add_argument("--backend", choices=["faiss", "qdrant"], default=None)
    parser.add_argument("--non-recursive", action="store_true", help="local connector only")
    # PH2 cleaning flags -- all off by default for back-compat.
    parser.add_argument("--clean-html", action="store_true", help="strip boilerplate from .html docs")
    parser.add_argument("--redact-pii", action="store_true", help="redact CN phone/ID/bank/email/ip")
    parser.add_argument("--filter-low-info", action="store_true", help="drop low-information chunks")
    parser.add_argument("--low-info-threshold", type=float, default=0.7)
    parser.add_argument("--dedup", action="store_true", help="MinHash near-duplicate drop within this ingest")
    parser.add_argument("--dedup-threshold", type=float, default=0.85)
    args = parser.parse_args()

    if args.backend:
        os.environ["VECTOR_BACKEND"] = args.backend
    from rag.config import get_settings
    get_settings.cache_clear()

    from rag.ingest.connectors import HttpPageConnector, LocalDirConnector
    from rag.ingest.incremental import CleanConfig, sync_kb
    from rag.knowledge_base import create_kb

    kb = create_kb(args.kb_id)

    if args.connector == "local":
        connector = LocalDirConnector(args.kb_id, args.source, recursive=not args.non_recursive)
    elif args.connector == "http":
        path = Path(args.source)
        if not path.exists():
            raise SystemExit(f"urls file not found: {path}")
        urls = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
        connector = HttpPageConnector(args.kb_id, urls)
    else:
        raise SystemExit(f"unsupported connector: {args.connector}")

    clean_cfg = CleanConfig(
        html_boilerplate=args.clean_html,
        low_info_filter=args.filter_low_info,
        low_info_threshold=args.low_info_threshold,
        pii_redact=args.redact_pii,
        dedup=args.dedup,
        dedup_threshold=args.dedup_threshold,
    )
    print(f"[sync] kb={args.kb_id} via {connector.describe()}")
    print(f"[sync] clean={clean_cfg}")
    stats = sync_kb(args.kb_id, connector, kb.root, clean_cfg=clean_cfg)
    print(json.dumps(stats.summary(), ensure_ascii=False))
    if stats.errors:
        print("[errors]")
        for sid, err in stats.errors:
            print(f"  - {sid}: {err}")


if __name__ == "__main__":
    main()
