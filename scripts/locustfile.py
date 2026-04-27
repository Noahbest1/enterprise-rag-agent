"""Locust load test for /answer.

Run:
    pip install locust
    locust -f scripts/locustfile.py --host http://127.0.0.1:8008 \
        -u 200 -r 20 -t 2m --headless \
        --csv data/eval/results/loadtest

Then read p50/p95/p99 from ``data/eval/results/loadtest_stats.csv`` (the row
with "Aggregated"). 200 VUs ramping at 20/s for 2 minutes is a reasonable
smoke for a single Python process; if you get <30 req/s something is wrong.

The query pool is seeded from the eval dataset so the cache-hit curve is
realistic (repeats → semantic cache should fire).

Notes:
- Point ``LOCUST_API_KEY`` at a valid Bearer if ``REQUIRE_API_KEY=true``.
- The test hits POST /answer (the non-streaming endpoint) because streaming
  latency metrics measure connection-open, not total response, which would
  under-count p99 on a streaming path.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

from locust import HttpUser, between, task


ROOT = Path(__file__).resolve().parents[1]
EVAL_ROWS_PATH = ROOT / "data" / "eval" / "eval_rows.jsonl"


def _load_queries() -> list[tuple[str, str]]:
    """Return a list of (query, kb_id) pairs from the eval dataset.

    Fallback to a hard-coded mini set if eval_rows.jsonl is missing.
    """
    if not EVAL_ROWS_PATH.exists():
        return [
            ("What is Airbyte?", "airbyte_demo"),
            ("京东 PLUS 会员年卡多少钱", "jd_demo"),
            ("88VIP 的开通条件是什么", "taobao_demo"),
        ]
    out: list[tuple[str, str]] = []
    with EVAL_ROWS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("category") == "out_of_domain":
                continue  # skip: these intentionally abstain, unrealistic for load
            out.append((row["query"], row["kb_id"]))
    random.shuffle(out)
    return out


_QUERIES = _load_queries()
_API_KEY = os.environ.get("LOCUST_API_KEY", "")


class AnswerUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self):
        self.session_queries = _QUERIES or [("ping", "jd_demo")]

    @task
    def ask(self):
        q, kb_id = random.choice(self.session_queries)
        headers = {"Content-Type": "application/json"}
        if _API_KEY:
            headers["Authorization"] = f"Bearer {_API_KEY}"
        with self.client.post(
            "/answer",
            json={"query": q, "kb_id": kb_id},
            headers=headers,
            name="/answer",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:100]}")
            else:
                resp.success()
