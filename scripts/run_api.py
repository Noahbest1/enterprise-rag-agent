"""Start the new RAG API on localhost:8008."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # makes `agent` (top-level pkg) importable

# Propagate to subprocesses uvicorn may spawn when reload=True.
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(ROOT / "src"), str(ROOT), os.environ.get("PYTHONPATH", "")]
).rstrip(os.pathsep)


def main():
    import uvicorn

    host = os.getenv("COPILOT_API_HOST", "127.0.0.1")
    port = int(os.getenv("COPILOT_API_PORT", "8008"))
    uvicorn.run("rag_api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
