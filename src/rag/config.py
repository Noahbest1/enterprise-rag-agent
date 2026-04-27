"""Application settings.

Reads from (in priority order):
    1. Actual environment variables (Docker/CI)
    2. .env.local   (developer overrides, not committed)
    3. .env         (shared defaults, committed as .env.example)

Use `from rag.config import settings` everywhere. Settings are immutable
after process start; tests should use `settings.model_copy(update=...)`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Later files override earlier files for the same key.
        env_file=(PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"),
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Runtime environment ---
    env: Literal["dev", "staging", "prod", "test"] = "dev"
    project_root: Path = PROJECT_ROOT
    kb_root: Path = PROJECT_ROOT / "data" / "kb"
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"  # console for dev, json for prod

    # --- Models ---
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-base"

    # --- Chunking ---
    chunk_target_tokens: int = 450
    chunk_overlap_tokens: int = 80

    # --- Retrieval ---
    bm25_top_k: int = 40
    vector_top_k: int = 40
    rrf_k: int = 60
    rerank_top_k: int = 8
    final_top_k: int = 5
    abstain_score_threshold: float = 0.15

    # --- Feature flags ---
    enable_query_rewrite: bool = True
    enable_multi_query_default: bool = False
    # Contextual Retrieval (Anthropic, 2024). Default ON: Anthropic reports
    # ~35% recall improvement at the cost of one LLM call per leaf chunk at
    # build time (cached per-chunk, so only first build pays). Without a
    # DashScope key the contextual call quietly fails and the chunk is
    # indexed unchanged -- no pytest / CI breakage.
    enable_contextual_retrieval: bool = Field(default=True, alias="ENABLE_CONTEXTUAL_RETRIEVAL")
    # Route simple tasks (rewrite/intent/contextual/judge) to qwen-turbo for
    # ~2-3x lower latency. Generation stays on qwen-plus for quality. ON by
    # default after sprint A verified no Hit@k / answer-faithfulness regression
    # (gate runs with rewrite=off so this flip has zero effect on baseline).
    enable_model_routing: bool = Field(default=True, alias="ENABLE_MODEL_ROUTING")
    qwen_turbo_model: str = "qwen-turbo"
    enable_injection_defense: bool = True  # first-line prompt injection filter
    # Intent routing: classify each query as meta / chitchat / kb BEFORE
    # retrieval. Meta-queries ("just translated", "above answer") bypass
    # KB and read the conversation history; chitchat skips KB too. This is
    # a bug fix, not a feature flag — default ON. Disable only for forensic
    # debugging of the kb path.
    enable_intent_routing: bool = Field(default=True, alias="ENABLE_INTENT_ROUTING")

    # Pass the last 2 conversation turns into the answerer prompt as
    # "context for understanding, not for facts". Helps follow-up questions
    # where the rewriter dropped the implicit subject. The prompt's
    # HISTORY_RULE clause forbids treating history as evidence — facts must
    # still come from cited chunks. Default ON.
    enable_answer_with_history: bool = Field(default=True, alias="ENABLE_ANSWER_WITH_HISTORY")
    answer_history_turns: int = 2  # user+assistant pairs
    answer_history_chars_per_turn: int = 300

    # --- LLM (Qwen / DashScope) ---
    llm_provider: Literal["qwen"] = "qwen"
    qwen_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")
    qwen_model: str = Field(default="qwen-plus", alias="COPILOT_QWEN_MODEL")
    qwen_chat_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        alias="COPILOT_QWEN_CHAT_COMPLETIONS_URL",
    )
    llm_timeout_seconds: int = 60
    llm_max_output_tokens: int = 800
    llm_temperature: float = 0.2
    max_context_chars: int = 6000

    # --- Fallback chain (sprint A) ---
    # When the primary LLM (Qwen) raises, try Claude; if Claude also fails or
    # isn't configured, fall back to a pure-extractive answer built from the
    # top hits (zero LLM calls, guaranteed to return *something* grounded).
    # Default OFF -- flip ENABLE_FALLBACK_CHAIN=true in prod, and set
    # ANTHROPIC_API_KEY if you want the Claude middle rung.
    enable_fallback_chain: bool = Field(default=False, alias="ENABLE_FALLBACK_CHAIN")

    # Online hallucination verification: after generating an answer, ask a
    # small model (qwen-turbo) to check each sentence against the cited
    # chunks. Adds ~300ms + one LLM call per answer; default OFF.
    enable_hallucination_check: bool = Field(
        default=False, alias="ENABLE_HALLUCINATION_CHECK",
    )
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-5-20250929", alias="ANTHROPIC_MODEL")
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com/v1/messages",
        alias="ANTHROPIC_BASE_URL",
    )

    # --- API service ---
    api_host: str = Field(default="127.0.0.1", alias="COPILOT_API_HOST")
    api_port: int = Field(default=8008, alias="COPILOT_API_PORT")
    cors_allow_origins: str = Field(
        default="http://127.0.0.1:5714,http://localhost:5714",
        alias="COPILOT_CORS_ALLOW_ORIGINS",
    )

    # --- Data services ---
    # vector_backend: faiss (default, zero-dep) | qdrant (prod-capable) |
    #                 pgvector (Postgres + pgvector extension, shares existing DB)
    vector_backend: Literal["faiss", "qdrant", "pgvector"] = "faiss"
    # qdrant_url options:
    #   ":memory:"         -- in-process, test-friendly
    #   "local://<path>"   -- on-disk local mode
    #   "http(s)://host"   -- remote Qdrant server
    qdrant_url: str = ":memory:"
    qdrant_api_key: str = ""
    # pgvector connection string (only used when vector_backend="pgvector").
    # Example: postgresql://user:pass@localhost:5432/rag
    # Falls back to settings.database_url when that URL starts with postgres*.
    pgvector_url: str = Field(default="", alias="PGVECTOR_URL")
    redis_url: str = "redis://localhost:6379/0"
    # SQLAlchemy URL. Dev default is a local SQLite file so nothing needs
    # Docker to come up. Docker Compose overrides to a real Postgres.
    database_url: str = Field(
        default="sqlite:///./data/rag.db",
        alias="DATABASE_URL",
    )

    # --- Rate limiting (Day 7) ---
    rate_limit_per_minute: int = 60

    # --- MMR diversification (sprint A.1) ---
    use_mmr: bool = True
    mmr_lambda: float = 0.7  # 1.0 = relevance-only, 0.0 = diversity-only

    # --- Metadata enrichment at ingest (sprint A.5) ---
    # One LLM call per leaf chunk extracts entities/topics/date into chunk
    # metadata, enabling entity_contains / topics_contains / date_after filters.
    # Cached per-chunk in <kb_dir>/metadata_enrich_cache.jsonl so rebuilds are free.
    enable_metadata_enrichment: bool = Field(default=False, alias="ENABLE_METADATA_ENRICHMENT")

    # --- Semantic chunking (sprint A) ---
    # Embedding-distance-based sub-splitting for oversized leaf sections.
    # Default OFF because it embeds every sentence (2-4x slower ingest).
    use_semantic_chunking: bool = Field(default=False, alias="ENABLE_SEMANTIC_CHUNKING")
    semantic_chunk_min_tokens: int = 150
    semantic_chunk_max_tokens: int = 800
    semantic_chunk_percentile: float = 95.0

    # --- API key auth (PH5) ---
    # When True, protected endpoints require `Authorization: Bearer <key>`.
    # Keep False in dev / tests; flip to True in prod via env.
    require_api_key: bool = Field(default=False, alias="REQUIRE_API_KEY")
    # Tenant returned when auth is disabled / no key given. Useful so
    # downstream code never has to handle "no tenant".
    anonymous_tenant_id: str = "public"

    # --- Per-request token budget (PH5) ---
    # Fail a request before calling the LLM if the prompt estimate exceeds
    # this. 0 disables.
    max_prompt_tokens_per_request: int = 8192
    max_output_tokens_per_request: int = 2048
    # --- Per-tenant token budget (PH5) ---
    # Sliding-window budget over ``tenant_budget_window_seconds``. 0 disables.
    # 1_000_000 tokens/min is generous; production tune per contract.
    tenant_token_budget: int = 1_000_000
    tenant_budget_window_seconds: int = 60

    # --- Rate limit tiers (PH5) ---
    # Default OFF so pytest and local dev don't hit 429s on bursty tests.
    # Flip to True via env in production.
    rate_limit_enabled: bool = Field(default=False, alias="RATE_LIMIT_ENABLED")
    rate_limit_authenticated: str = "200/minute"
    rate_limit_anonymous: str = "30/minute"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Module-level alias so existing `from rag.config import settings` keeps working.
settings = get_settings()
