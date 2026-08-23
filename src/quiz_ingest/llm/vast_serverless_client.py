"""
llm/vast_serverless_client.py

NOT imported by scripts/ by default until you explicitly pass
`--backend serverless`. This exists so that when you move from a plain
rented instance (VLLMClient) to a Vast Serverless Endpoint, the rest of
the pipeline (rag/index.py, generation/, eval/) needs ZERO changes --
both clients satisfy the same llm.base.LLMBackend Protocol.

Do not wire this in until:
  1. VLLMClient (instance mode) has been validated end to end against the
     model/config you've chosen (per context.md's staged plan).
  2. You've created a Template + Endpoint + Workergroup for the same
     Docker image/model already validated in step 1.

Confirmed against the current `vastai` package and docs.vast.ai's
Serverless Quickstart -- NOT the old `vast-sdk` package, which is
deprecated. Install with:
    pip install "quiz-ingest-vllm[serverless]"

Call shape confirmed from docs.vast.ai/guides/serverless/quickstart:
`request()` is called on the ENDPOINT object (not the client), and takes
a `cost` kwarg (the token budget for this call, used for the autoscaler's
accounting -- pass max_tokens for a generation call):

    endpoint = await client.get_endpoint(name="...")
    result = await endpoint.request("/v1/completions", payload, cost=MAX_TOKENS)

This corrects an earlier version of this file that called
`self._serverless.request(path, body)` directly on the client -- that
was written before this doc page was available and guessed wrong.

One inconsistency worth flagging: docs.vast.ai/guides/serverless/vllm's
example wraps the payload as `{"input": {...actual args...}}`, while the
Quickstart page's example passes the args flat (no "input" wrapper). This
file uses the flat form (Quickstart's, the more general-purpose page) --
if a real call 400s on payload shape, try wrapping it in `{"input": ...}`
next.
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

    Known limitation, be upfront about it: `endpoint.request()` is a
    single request/response call in every example in the current docs --
    no streaming variant is shown. Treat it as non-streaming until proven
    otherwise: ttft_s stays None here (never fabricated), only
    wall_time_s is real. If per-call TTFT matters for a decision, that
    decision needs instance-mode data (VLLMClient), not this backend.
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
        self._endpoint = None  # resolved lazily, cached after first call

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
        endpoint = await self._ensure_endpoint()
        numbered_items = "\n".join(f"[{i}] {p}" for i, p in enumerate(item_prompts))
        full_prompt = (
            f"{shared_prefix}\n\n"
            f"Respond with a single JSON array of exactly {len(item_prompts)} objects, "
            f"one per item below, in the same order (0-indexed). Schema per object: "
            f"{schema_hint}\n\nItems:\n{numbered_items}\n\nJSON array:"
        )
        payload = {"model": self._cfg.model, "prompt": full_prompt, "max_tokens": max_tokens}

        t0 = time.monotonic()
        result = await endpoint.request("/v1/completions", payload, cost=max_tokens)
        wall_time = time.monotonic() - t0

        choice = result["response"]["choices"][0]
        raw_text = choice.get("text", "")
        usage = result["response"].get("usage", {})

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
            hit_token_limit=(usage.get("completion_tokens", 0) >= max_tokens > 0),
        )
        return GenerateJsonBatchResult(
            items=items, raw_text=raw_text, telemetry=telemetry, parse_failures=parse_failures
        )

    async def embed(self, texts: list[str]) -> tuple[list[list[float]], CallTelemetry]:
        endpoint = await self._ensure_endpoint()
        if not self._cfg.embed_model:
            raise ValueError("VastServerlessConfig.embed_model not set")
        payload = {"model": self._cfg.embed_model, "input": texts}

        t0 = time.monotonic()
        # cost=0: embedding calls have no completion tokens. Unconfirmed
        # against real billing/autoscaler behavior -- if a real call
        # rejects cost=0, try a nominal estimate (e.g. total input chars
        # // 4) instead and note here what worked.
        result = await endpoint.request("/v1/embeddings", payload, cost=0)
        wall_time = time.monotonic() - t0

        data = result["response"]
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
