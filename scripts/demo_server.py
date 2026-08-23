#!/usr/bin/env python
"""
scripts/demo_server.py

Runs the FastAPI demo server (quiz_ingest.api:app). Configure the
backend via environment variables BEFORE starting -- see
quiz_ingest/api.py's _server_config docstring for exactly which ones.

Instance mode:
    export QUIZ_BACKEND=instance
    export VLLM_BASE_URL=http://localhost:8000
    export EMBED_BASE_URL=http://localhost:8001

Serverless mode:
    export QUIZ_BACKEND=serverless
    export VAST_API_KEY=...
    export VAST_ENDPOINT_NAME=quiz-gen-qwen3-4b
    export VAST_EMBED_ENDPOINT_NAME=quiz-embed-qwen3

Strongly recommended before exposing this on a public IP (see
quiz_ingest/api.py's _require_api_key docstring):
    export DEMO_API_KEY=<a secret you make up, share it with whoever gets the link>

Then:
    pip install -e ".[api]"
    python scripts/demo_server.py --host 0.0.0.0 --port 8080

Swagger UI (the page in the screenshot this script is built to match):
    http://<this-machine's-ip>:8080/docs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    import uvicorn

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--reload", action="store_true", help="Auto-reload on code changes (development only)"
    )
    args = p.parse_args()

    uvicorn.run("quiz_ingest.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
