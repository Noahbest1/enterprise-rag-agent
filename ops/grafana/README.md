# Grafana dashboards

Pre-built dashboards for the RAG API's Prometheus metrics.

## rag-overview.json

Six panels covering the core SLO story:

1. **Requests / s (by route)** — volume by route template (`/kbs/{kb_id}`, `/answer`, …).
2. **Latency p50 / p95 / p99** — computed from the `rag_request_duration_seconds` histogram.
3. **HTTP 4xx / 5xx rate** — error rate, with a dedicated line for 429 (rate limited).
4. **LLM tokens / s (by task, kind)** — prompt vs completion token burn, split by `task` label.
5. **LLM call status (ok / error)** — LLM request success vs failure rate.
6. **Cache hit rate (by layer)** — hit / (hit + miss) for `embedding`, `llm`, `semantic`, `vision`.

### Import

In Grafana → Dashboards → Import → Upload JSON file → select `rag-overview.json`.
Pick your Prometheus datasource for the `DS_PROMETHEUS` variable when prompted.

### Metrics source

Prometheus should scrape the RAG API's `/metrics` endpoint. Minimal `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: rag-api
    metrics_path: /metrics
    static_configs:
      - targets: ['rag-api:8008']
```
