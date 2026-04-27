"""Create a new API key.

Usage:
    python scripts/create_api_key.py <tenant_id> [--desc "jd-prod readonly"]

Prints the raw key ONCE. Store it in the caller's secrets manager. Only the
sha256 hash is written to the database; the raw key can never be recovered.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.db.base import SessionLocal
from rag.db.models import ApiKey
from rag_api.auth import generate_raw_key, hash_key


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("tenant_id", help="Tenant this key belongs to (e.g. jd, taobao, acme_corp)")
    p.add_argument("--desc", default=None, help="Optional human-readable description")
    p.add_argument("--scopes", default="", help="Comma-separated scopes (empty = all)")
    args = p.parse_args()

    raw = generate_raw_key()
    key_hash = hash_key(raw)

    with SessionLocal() as s:
        row = ApiKey(
            key_hash=key_hash,
            tenant_id=args.tenant_id,
            description=args.desc,
            scopes=args.scopes,
        )
        s.add(row)
        s.commit()
        kid = row.id

    print(f"Created API key for tenant={args.tenant_id} (id={kid}).")
    print(f"Raw key (shown once — copy now, cannot be recovered):\n\n  {raw}\n")
    print("Use it as:  Authorization: Bearer " + raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
