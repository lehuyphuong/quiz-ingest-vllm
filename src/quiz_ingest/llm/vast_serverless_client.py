"""
llm/vast_serverless_client.py

NOT imported by scripts/ by default. This exists so that when you move
from a plain rented instance (VLLMClient) to a Vast Serverless Endpoint,
the rest of the pipeline (rag/index.py, generation/, eval/) needs ZERO
changes -- both clients satisfy the same llm.base.LLMBackend Protocol.

Do not wire this in until:
  1. VLLMClient (instance mode) has been validated end to end against the
     model/config you've chosen (per context.md's staged plan).
  2. You've created a Template + Endpoint + Workergroup for the same
     Docker image/model already validated in step 1.

Confirmed against the current (2026) `vastai` package -- NOT the old
`vast-sdk` package, which is deprecated. Install with:
    pip install "quiz-ingest-vllm[serverless]"

Usage note: `serverless.request(path, body)` takes an OpenAI-compatible
path ("/v1/completions", "/v1/embeddings") and body -- the exact same
shapes VLLMClient sends. That symmetry is why this file is a thin wrapper
rather than a rewrite: it reuses VLLMClient's prompt-building and
JSON-parsing helpers, and only swaps the transport.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from quiz_ingest.llm.base import CallTelemetry, GenerateJsonBatchResult
from quiz_ingest.llm.vllm_client import _parse_json_array_batch


@dataclass
class VastServerlessConfig:
    endpoint_name: str
    model: str
    embed_model: str | None = None
    api_key: str | None = None  # falls back to VAST_API_KEY env var, same as the vastai SDK default


class VastServerlessClient:
    """
    Wraps vastai.Serverless. Constructed lazily (import inside __init__)
    so that installing this repo WITHOUT the [serverless] extra still
    works for the instance-mode path -- vastai is an optional dependency,
    not a core one.

    Known limitation, be upfront about it: `serverless.request()` is a
    single request/response call, not a streaming one (unlike
    VLLMClient._stream_completion) as of the current vastai package --
    confirm this against the SDK's own docs/changelog before relying on
    it, since it directly affects whether TTFT can be measured the same
    way as instance mode. If it isn't streaming yet, ttft_s will be None
    here and only wall_time_s / decode_tps-from-total-time (an
    approximation, flagged as such) will be available -- do not silently
    report an approximated number in the same field as a real one.
    """

    def __init__(self, config: VastServerlessConfig):
        try:
            from vastai import Serverless
        except ImportError as exc:
            raise ImportError(
                "VastServerlessClient requires the 'serverless' extra: "
                "pip install \"quiz-ingest-vllm[serverless]\""
            ) from exc
        self._cfg = config
        self._serverless = Serverless(config.api_key) if config.api_key else Serverless()
        self._endpoint = None  # resolved lazily in health_check/first call

    @property
    def model_name(self) -> str:
        return self._cfg.model

    async def _ensure_endpoint(self):
        if self._endpoint is None:
            self._endpoint = await self._serverless.get_endpoint(self._cfg.endpoint_name)
        return self._endpoint

    async def health_check(self) -> bool:
        try:
            await self._ensure_endpoint()
            return True
        except Exception:
            return False

    async def generate_json_batch(
        self,
        *,
        shared_prefix: str,
        item_prompts: list[str],
        schema_hint: str,
        max_tokens: int,
    ) -> GenerateJsonBatchResult:
        await self._ensure_endpoint()
        numbered_items = "\n".join(f"[{i}] {p}" for i, p in enumerate(item_prompts))
        full_prompt = (
            f"{shared_prefix}\n\n"
            f"Respond with a single JSON array of exactly {len(item_prompts)} objects, "
            f"one per item below, in the same order (0-indexed). Schema per object: "
            f"{schema_hint}\n\nItems:\n{numbered_items}\n\nJSON array:"
        )
        body = {"model": self._cfg.model, "prompt": full_prompt, "max_tokens": max_tokens}

        t0 = time.monotonic()
        response = await self._serverless.request("/v1/completions", body)
        wall_time = time.monotonic() - t0

        choice = response["response"]["choices"][0]
        raw_text = choice.get("text", "")
        usage = response["response"].get("usage", {})

        items, parse_failures = _parse_json_array_batch(raw_text, len(item_prompts))
        telemetry = CallTelemetry(
            call_type="generate_json",
            model=self._cfg.model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            # See class docstring -- do not fabricate a TTFT for a
            # non-streaming transport.
            ttft_s=None,
            decode_tokens_per_second=None,
            wall_time_s=wall_time,
        )
        return GenerateJsonBatchResult(
            items=items, raw_text=raw_text, telemetry=telemetry, parse_failures=parse_failures
        )

    async def embed(self, texts: list[str]) -> tuple[list[list[float]], CallTelemetry]:
        await self._ensure_endpoint()
        if not self._cfg.embed_model:
            raise ValueError("VastServerlessConfig.embed_model not set")
        t0 = time.monotonic()
        response = await self._serverless.request(
            "/v1/embeddings", {"model": self._cfg.embed_model, "input": texts}
        )
        wall_time = time.monotonic() - t0
        data = response["response"]
        vectors = [item["embedding"] for item in data["data"]]
        usage = data.get("usage", {})
        telemetry = CallTelemetry(
            call_type="embed",
            model=self._cfg.embed_model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=0,
            ttft_s=None,
            decode_tokens_per_second=None,
            wall_time_s=wall_time,
        )
        return vectors, telemetry
