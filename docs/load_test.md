# Load test

Goal: answer "how many QPS / what p99 latency can this survive on one process"
with a reproducible command. We deliberately don't run this in CI — a real
load test needs real LLM calls (cost) and is slow (minutes).

## Run

```bash
# 1. Start the API in a throwaway mode
DASHSCOPE_API_KEY=sk-... \
ENABLE_MODEL_ROUTING=true \
./.venv/bin/python scripts/run_api.py   # :8008

# 2. In another shell, fire locust
./.venv/bin/pip install locust
./.venv/bin/locust -f scripts/locustfile.py \
    --host http://127.0.0.1:8008 \
    -u 200 -r 20 -t 2m --headless \
    --csv data/eval/results/loadtest
```

- `-u 200` = ramp up to 200 concurrent virtual users
- `-r 20` = spawn 20/s so the ramp is over in 10s
- `-t 2m` = run for 2 minutes after the ramp
- `--csv` = write `loadtest_stats.csv` + `loadtest_failures.csv` to disk

If `REQUIRE_API_KEY=true` in the API process, export
`LOCUST_API_KEY=<bearer>` before running locust.

## Read the numbers

After the run, the last line of `loadtest_stats.csv` is the aggregated row.
The columns you want (in Locust 2.x):

| column | meaning |
|---|---|
| `Request Count` | total requests |
| `Failure Count` | non-2xx, should be 0 |
| `Average Response Time` | mean in ms |
| `50%` | p50 latency (ms) |
| `95%` | p95 latency (ms) |
| `99%` | p99 latency (ms) |
| `Requests/s` | measured throughput |

A healthy single-process baseline on this codebase (with a local DashScope
response ~600ms, semantic cache warm after 30s) should look roughly:

- p50 ~300-600ms (most requests hit the semantic cache)
- p95 ~1.5-3s
- p99 ~3-5s
- Requests/s ~30-60 depending on cache warmup

**Failures > 1%** usually mean either the API was underprovisioned (CPU
saturation, Qwen upstream throttling) or a logic bug landed since the
baseline was taken.

## What the test does NOT prove

- Multi-process scale — we only run one `uvicorn` worker. Production should
  `gunicorn -w N` or equivalent.
- Cold-start latency — locust's first few requests after spawn are always
  slower; look at the steady state, not the ramp.
- Streaming first-token latency — the test hits POST `/answer`, not
  `/answer/stream`. For streaming first-byte measurements, use a dedicated
  curl + `-w '%{time_starttransfer}'` smoke.
