"""
llm/vllm_client.py

Talks directly to a vLLM OpenAI-compatible server (`vllm serve ...`) running
on a plain rented Vast.ai GPU instance -- e.g. `http://<instance-ip>:8000`.
This is the default backend for this repo (see context.md).

Two things this client does that are easy to get wrong and directly affect
the 3 things the pipeline is being optimized for (latency, quality, real
token/s per stage):

1. STREAMING, so TTFT and decode-tps are measured, not estimated. The old
   quizrag-scale repo's vLLM client only had total request latency
   (non-streaming) and called decode_tokens_per_second an approximation.
   Self-hosting removes any reason to accept that -- see _stream_completion.

2. PREFIX STABILITY. `generate_json_batch` takes `shared_prefix` and
   `item_prompts` as SEPARATE arguments (not pre-joined by the caller) and
   concatenates them in this file, in one fixed place, so the exact same
   prefix string is byte-identical across every call in a job. If a caller
   built the full prompt itself and small formatting differences crept in
   between calls (extra whitespace, reordered fields), vLLM's prefix cache
   would silently miss and every call would re-run the full prefill --
   keeping construction in one place is what protects that invariant.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx

from quiz_ingest.llm.base import CallTelemetry, GenerateJsonBatchResult


class VLLMResponseError(RuntimeError):
    """Raised when the server returns a non-2xx status or malformed SSE."""


# Generous sanity ceiling for decode_tokens_per_second -- see the
# discard-and-flag logic in _stream_completion for why this exists (a
# client-side asyncio scheduling artifact under concurrent load, not a
# real hardware limit). Set well above any legitimate single-stream
# throughput for a small-to-mid dense model on a single consumer/
# workstation GPU; tune upward only if you've confirmed a LEGITIMATE
# faster setup (e.g. speculative decoding) actually exceeds this.
MAX_PLAUSIBLE_DECODE_TOKENS_PER_SECOND = 500.0


@dataclass
class VLLMClientConfig:
    base_url: str  # e.g. "http://136.65.146.37:8000" -- no trailing slash
    model: str
    embed_model: str | None = None  # if the same server also serves embeddings; else use a second instance
    request_timeout_s: float = 300.0
    max_retries: int = 3


class VLLMClient:
    """
    Minimal async client for one vLLM instance. Deliberately does NOT
    implement process-wide semaphores/circuit breakers here -- on a single
    rented GPU instance, vLLM's own continuous batching + request queue is
    the concurrency control; layering another semaphore on top of it (the
    way the old repo's fully-serial `slm_semaphore=1` did for CPU/Ollama)
    would just under-utilize the GPU. If/when you run multiple concurrent
    ingestion jobs against ONE instance, tune vLLM's own
    `--max-num-seqs` server-side instead of client-side throttling.
    """

    def __init__(self, config: VLLMClientConfig):
        self._cfg = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url, timeout=config.request_timeout_s
        )

    @property
    def model_name(self) -> str:
        return self._cfg.model

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Core streaming call -- this is the one place TTFT/decode-tps are real
    # ------------------------------------------------------------------
    async def _stream_completion(
        self, *, prompt: str, max_tokens: int, extra_body: dict | None = None
    ) -> tuple[str, CallTelemetry]:
        """
        Streams /v1/completions with stream=True and stream_options to get
        real per-request usage. Returns (full_text, telemetry).

        TTFT  = time from request sent to the first non-empty text chunk.
        decode_tps = (completion_tokens - 1) / (t_last_chunk - t_first_chunk)
          -- the "-1" excludes the already-counted first token, matching
          how vLLM's own Prometheus histogram defines time_per_output_token.
        """
        body = {
            "model": self._cfg.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": True,
            # Ask the server to include a final usage-only SSE chunk. This
            # is what lets prompt_tokens/completion_tokens be REAL numbers
            # from the server, not counted client-side with a tokenizer
            # that might not match the server's exactly.
            "stream_options": {"include_usage": True},
        }
        if extra_body:
            body.update(extra_body)

        last_error: Exception | None = None
        for attempt in range(self._cfg.max_retries):
            t_sent = time.monotonic()
            t_first_token: float | None = None
            t_last_token: float | None = None
            text_parts: list[str] = []
            prompt_tokens = 0
            completion_tokens = 0
            prefix_hit_tokens: int | None = None

            try:
                async with self._client.stream(
                    "POST", "/v1/completions", json=body
                ) as resp:
                    if resp.status_code != 200:
                        raw = await resp.aread()
                        raise VLLMResponseError(
                            f"vLLM returned {resp.status_code}: {raw[:500]!r}"
                        )
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[len("data:"):].strip()
                        if payload == "[DONE]":
                            break
                        chunk = json.loads(payload)

                        choices = chunk.get("choices") or []
                        if choices:
                            piece = choices[0].get("text", "")
                            if piece:
                                now = time.monotonic()
                                if t_first_token is None:
                                    t_first_token = now
                                t_last_token = now
                                text_parts.append(piece)

                        # Final usage chunk (choices == [] when include_usage is set)
                        usage = chunk.get("usage")
                        if usage:
                            prompt_tokens = usage.get("prompt_tokens", 0)
                            completion_tokens = usage.get("completion_tokens", 0)
                            # Not all vLLM versions report this; guarded read.
                            details = usage.get("prompt_tokens_details") or {}
                            prefix_hit_tokens = details.get("cached_tokens")

                full_text = "".join(text_parts)
                wall_time = time.monotonic() - t_sent
                ttft = (t_first_token - t_sent) if t_first_token is not None else None
                decode_tps = None
                decode_tps_discarded = False
                if (
                    t_first_token is not None
                    and t_last_token is not None
                    and completion_tokens > 1
                    and t_last_token > t_first_token
                ):
                    candidate_tps = (completion_tokens - 1) / (t_last_token - t_first_token)
                    # Sanity ceiling, not a real hardware limit -- this
                    # client measures inter-chunk time using its OWN
                    # asyncio task's timestamps. Under high CLIENT-SIDE
                    # concurrency (many coroutines sharing one event
                    # loop, e.g. load_test.py at concurrency>=4), a task
                    # can be starved of scheduling time while tokens sit
                    # in the OS socket buffer, then drain a burst of
                    # already-arrived chunks almost instantly once
                    # scheduled -- t_last-t_first collapses toward zero
                    # while completion_tokens stays normal, producing
                    # decode_tps values in the tens of thousands (a real
                    # incident: 54,642 tok/s logged against a GPU whose
                    # single-request baseline is ~90-100 tok/s -- not
                    # possible on this hardware for a 4B model). Discard
                    # rather than report a fabricated number -- same
                    # principle as ttft_s=None for non-streaming
                    # transports: an unreliable measurement must never
                    # look like a real one. For trustworthy decode
                    # throughput UNDER concurrent client load, use
                    # vLLM's own server-side /metrics (Prometheus) --
                    # immune to this client-side scheduling artifact
                    # entirely, since it's measured at the GPU, not the
                    # asyncio event loop.
                    if candidate_tps <= MAX_PLAUSIBLE_DECODE_TOKENS_PER_SECOND:
                        decode_tps = candidate_tps
                    else:
                        decode_tps_discarded = True

                telemetry = CallTelemetry(
                    call_type="generate_json",
                    model=self._cfg.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    ttft_s=ttft,
                    decode_tokens_per_second=decode_tps,
                    wall_time_s=wall_time,
                    prefix_cache_hit_tokens=prefix_hit_tokens,
                    retry_count=attempt,
                    hit_token_limit=(completion_tokens >= max_tokens > 0),
                    decode_tps_discarded_as_implausible=decode_tps_discarded,
                )
                return full_text, telemetry

            except (httpx.HTTPError, VLLMResponseError, json.JSONDecodeError) as exc:
                last_error = exc
                continue

        raise VLLMResponseError(
            f"vLLM request failed after {self._cfg.max_retries} attempts"
        ) from last_error

    # ------------------------------------------------------------------
    # Public API (LLMBackend Protocol)
    # ------------------------------------------------------------------
    async def generate_json_batch(
        self,
        *,
        shared_prefix: str,
        item_prompts: list[str],
        schema_hint: str,
        max_tokens: int,
    ) -> GenerateJsonBatchResult:
        # Prefix is built ONCE, here, and always in this exact order --
        # this is the string vLLM's prefix cache keys on. Do not let
        # callers interpolate item-specific text into `shared_prefix`.
        numbered_items = "\n".join(
            f"[{i}] {p}" for i, p in enumerate(item_prompts)
        )
        full_prompt = (
            f"{shared_prefix}\n\n"
            f"Respond with a single JSON array of exactly {len(item_prompts)} objects, "
            f"one per item below, in the same order (0-indexed). Schema per object: "
            f"{schema_hint}\n\nItems:\n{numbered_items}\n\nJSON array:"
        )

        raw_text, telemetry = await self._stream_completion(
            prompt=full_prompt, max_tokens=max_tokens
        )

        items, parse_failures = _parse_json_array_batch(raw_text, len(item_prompts))
        return GenerateJsonBatchResult(
            items=items, raw_text=raw_text, telemetry=telemetry, parse_failures=parse_failures
        )

    async def embed(self, texts: list[str]) -> tuple[list[list[float]], CallTelemetry]:
        if not self._cfg.embed_model:
            raise ValueError(
                "VLLMClientConfig.embed_model not set -- point this client at an "
                "instance running with --runner pooling (the current vLLM flag; "
                "--task embed is deprecated), or use a second VLLMClient for "
                "embeddings (see README: generation and embedding are separate "
                "vLLM processes/instances)."
            )
        t0 = time.monotonic()
        resp = await self._client.post(
            "/v1/embeddings",
            json={"model": self._cfg.embed_model, "input": texts},
        )
        if resp.status_code != 200:
            raise VLLMResponseError(f"embed call failed: {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        wall_time = time.monotonic() - t0
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


def _parse_json_array_batch(raw_text: str, expected_len: int) -> tuple[list[dict], list[int]]:
    """
    Parses a JSON array out of `raw_text`, tolerating the model wrapping it
    in prose or markdown fences BEFORE the array, and (this is the part
    that matters here) trailing prose AFTER it too. Returns (items,
    failed_indices) -- items shorter than expected_len are padded with {}
    so callers can align by position; failed_indices records which slots
    are placeholders so downstream eval code can EXCLUDE them from
    scoring instead of silently treating a placeholder as a real 0.0
    score (see repo README's incident #13 in the earlier quizrag-scale
    project for why this distinction matters).

    Uses json.JSONDecoder().raw_decode() instead of text.find("[") +
    text.rfind("]") + json.loads() on the slice between them -- rfind was
    a real bug, confirmed from a load test's logs: when the model appends
    a trailing disclaimer sentence that happens to itself contain bracket
    characters (observed verbatim: "...] The provided reference context
    does not contain any information about [topic]. ... Output: []"),
    rfind("]") grabs the LAST bracket in the whole text -- the one from
    "Output: []" -- not the one that actually closes the real array. The
    resulting slice spans across the prose in between, which is not valid
    JSON, so json.loads() failed and the ENTIRE batch was marked a parse
    failure even though the model had produced a fully valid array right
    up until that trailing sentence. raw_decode() parses forward from the
    first "[" and stops exactly where the JSON value ends, ignoring
    whatever comes after -- immune to this class of bug regardless of
    what characters appear in any trailing text.
    """
    text = raw_text.strip()
    start = text.find("[")
    parsed: list = []
    if start != -1:
        try:
            parsed, _end_index = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            parsed = []
        if not isinstance(parsed, list):
            parsed = []

    items: list[dict] = []
    failures: list[int] = []
    for i in range(expected_len):
        if i < len(parsed) and isinstance(parsed[i], dict):
            items.append(parsed[i])
        else:
            items.append({})
            failures.append(i)
    return items, failures
