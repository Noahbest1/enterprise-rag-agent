# Multi-stage build for the RAG API service.
# Stage 1 compiles wheels into a cacheable layer so code changes don't
# re-download torch/transformers every build. Stage 2 is a slim runtime.

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build deps only needed to compile native wheels (faiss, sentence-transformers).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements-api.txt requirements-retrieval.txt ./

# All runtime deps now live in requirements-api.txt (single source of truth).
# Previously this layer also pip-installed pydantic-settings / structlog /
# qdrant-client / etc. inline — moved them into requirements so CI + Docker
# stay in sync.
RUN pip install --upgrade pip && \
    pip install -r requirements-api.txt


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Make the package layout importable
    PYTHONPATH=/app/src

# libgomp is needed by faiss at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy only what we actually need at runtime. Keeps the image small
# and avoids invalidating this layer on unrelated file changes.
COPY src /app/src
COPY scripts /app/scripts
COPY data /app/data
COPY agent /app/agent
# alembic.ini lives at the repo root; without it `alembic upgrade head` on
# boot dies with "No 'script_location' key found in configuration."
COPY alembic.ini /app/alembic.ini

# Non-root user, common prod hygiene.
RUN useradd --create-home --uid 1000 rag && chown -R rag:rag /app
USER rag

EXPOSE 8008
# Cloud platforms (Render / Fly.io / Heroku) inject $PORT — fall back to
# 8008 for local docker-compose. Healthcheck respects the same var.
ENV PORT=8008

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import os,urllib.request,sys; p=os.environ.get('PORT','8008'); r=urllib.request.urlopen(f'http://127.0.0.1:{p}/health', timeout=3); sys.exit(0 if r.status==200 else 1)"

# Start command applies alembic migrations first (idempotent), then uvicorn.
# This means a fresh deploy boots up with the schema ready to accept writes.
CMD ["sh", "-c", "alembic upgrade head && uvicorn rag_api.main:app --host 0.0.0.0 --port ${PORT:-8008}"]
