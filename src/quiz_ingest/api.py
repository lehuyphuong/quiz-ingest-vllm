"""
api.py

FastAPI wrapper around the pipeline for a "share a public link" demo --
upload a PDF, get a scored quiz back, browse/try it interactively at
/docs (Swagger UI). This is a demo server, not a hardened production API:
no queueing, no rate limiting beyond nothing, no HTTPS termination (put
this behind a reverse proxy for anything beyond a quick demo), and only
a single shared API key for auth (optional, off by default).

Backend (instance vs Vast Serverless) is configured ONCE at server
startup via environment variables (see _server_config below), never from
the request body -- the person hitting this API from a browser shouldn't
need to know or supply GPU infrastructure details, only
topic/num_questions/the PDF itself.

The response body is read back from the SAME consolidated JSON file
run_job_streaming already writes for the CLI (see output_writer.py) --
not rebuilt here -- so the API response and a CLI run's output file can
never drift apart into two different schemas.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from quiz_ingest.config import MAX_NUM_QUESTIONS, MAX_TOPIC_CHARS, InvalidRequestError, PipelineConfig
from quiz_ingest.ingest.pdf_source import MAX_RAW_BYTES
from quiz_ingest.pipeline import run_job_streaming

app = FastAPI(
    title="Quiz Ingest API",
    description=(
        "Upload a PDF and get back a scored multiple-choice quiz "
        "(RAGAS Faithfulness, Answer Relevance, Distractor Diversity). "
        "Generation runs on a self-hosted vLLM backend."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # public demo -- tighten this for anything beyond a demo
    allow_methods=["*"],
    allow_headers=["*"],
)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_api_key(key: str | None = Depends(_api_key_header)) -> None:
    """
    No-op (endpoint left open) unless DEMO_API_KEY is set in the server's
    environment. For a link you're handing to one specific person, set
    DEMO_API_KEY and share the key out of band -- leaving this open on a
    public IP means anyone who finds the URL can burn your GPU time.
    """
    expected = os.environ.get("DEMO_API_KEY")
    if not expected:
        return
    if key != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


def _server_config(num_questions: int) -> PipelineConfig:
    backend = os.environ.get("QUIZ_BACKEND", "instance")
    common = dict(
        model=os.environ.get("QUIZ_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
        embed_model=os.environ.get("QUIZ_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
        num_questions=num_questions,
    )
    if backend == "serverless":
        endpoint_name = os.environ["VAST_ENDPOINT_NAME"]  # KeyError -> caught by caller, 500
        return PipelineConfig(
            backend="serverless",
            vast_endpoint_name=endpoint_name,
            vast_embed_endpoint_name=os.environ.get("VAST_EMBED_ENDPOINT_NAME", endpoint_name),
            **common,
        )
    vllm_base_url = os.environ["VLLM_BASE_URL"]  # KeyError -> caught by caller, 500
    return PipelineConfig(
        backend="instance",
        vllm_base_url=vllm_base_url,
        embed_base_url=os.environ.get("EMBED_BASE_URL", vllm_base_url),
        **common,
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/generate-quiz", dependencies=[Depends(_require_api_key)])
async def generate_quiz(
    pdf: UploadFile = File(..., description="PDF to generate a quiz from"),
    topic: str = Form(
        ..., max_length=MAX_TOPIC_CHARS, description="Topic hint used for retrieval within the PDF"
    ),
    num_questions: int = Form(5, ge=1, le=MAX_NUM_QUESTIONS),
) -> dict:
    raw = await pdf.read()
    if len(raw) > MAX_RAW_BYTES:
        raise HTTPException(
            status_code=413, detail=f"PDF exceeds {MAX_RAW_BYTES // 1_000_000}MB cap"
        )

    try:
        config = _server_config(num_questions)
    except (KeyError, InvalidRequestError) as exc:
        raise HTTPException(status_code=500, detail=f"server misconfigured: {exc}") from exc

    tmp_pdf_path = Path(tempfile.gettempdir()) / f"quiz-upload-{uuid.uuid4().hex}.pdf"
    tmp_output_path = Path(tempfile.gettempdir()) / f"quiz-output-{uuid.uuid4().hex}.json"
    tmp_pdf_path.write_bytes(raw)
    try:
        async for _group in run_job_streaming(
            pdf_path=str(tmp_pdf_path),
            topic=topic,
            config=config,
            output_json_path=str(tmp_output_path),
        ):
            pass  # response is read back from tmp_output_path below, not accumulated here
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"generation failed: {exc}") from exc
    finally:
        tmp_pdf_path.unlink(missing_ok=True)

    if not tmp_output_path.exists():
        # run_job_streaming only writes output_json_path on a fully
        # "completed" status -- see pipeline.py. Its absence here means
        # every item was dropped (parse failures, duplicate distractors
        # never repaired, etc.) -- check logs/*.jsonl on the server for
        # parse_failure events, same diagnostic path as the CLI.
        raise HTTPException(
            status_code=502,
            detail="No questions could be generated for this PDF/topic -- check server-side logs/*.jsonl for parse_failure events.",
        )

    try:
        output = json.loads(tmp_output_path.read_text())
    finally:
        tmp_output_path.unlink(missing_ok=True)

    return output
