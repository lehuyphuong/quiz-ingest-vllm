"""
llm/base.py

Defines the Protocol every backend must satisfy. pipeline.py, rag/index.py,
generation/batch_quiz_gen.py, and eval/*.py all only call the methods
declared here -- they never import a concrete client. This is what lets
the same pipeline run unmodified against:

  - VLLMClient        (llm/vllm_client.py)        -- a plain rented Vast.ai
    GPU instance, called directly over HTTP via the vLLM OpenAI-compatible
    server. This is the default and the only backend exercised in this
    repo's tests.

  - VastServerlessClient (llm/vast_serverless_client.py) -- the same vLLM
    server, but reached through a Vast Serverless Endpoint instead of a
    fixed instance IP. Not wired into scripts/ by default -- see that
    file's docstring for why, and switch to it only after the instance-mode
    path has been validated end to end.

Both implementations send the exact same request bodies (OpenAI-compatible
/v1/completions, /v1/embeddings shape) -- the only thing that changes
between them is how the HTTP call is transported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class CallTelemetry:
    """
    Per-call timing, captured directly from the streaming response --
    never estimated. See vllm_client.py::_stream_completion for how
    these numbers are derived.
    """

    call_type: str  # "generate_json" | "embed"
    model: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float | None  # time to first token; None for embed calls (no decode phase)
    decode_tokens_per_second: float | None  # None for embed calls, or if completion_tokens <= 1
    wall_time_s: float
    prefix_cache_hit_tokens: int | None = None  # from vLLM's usage extension, when the server reports it
    retry_count: int = 0
    hit_token_limit: bool = False  # completion_tokens == the max_tokens sent for this call --
    # a strong signal of truncation (the response was very likely cut off
    # mid-content, not that the model naturally finished at that exact
    # length). Computed automatically in vllm_client.py -- see that
    # file's docstring for a real incident this would have caught
    # instantly instead of requiring manual completion_tokens-vs-budget
    # cross-referencing across several log entries.
    decode_tps_discarded_as_implausible: bool = False  # True when a
    # computed decode_tokens_per_second exceeded a sanity ceiling and was
    # discarded (decode_tokens_per_second left None instead) -- see
    # vllm_client.py's MAX_PLAUSIBLE_DECODE_TOKENS_PER_SECOND for why:
    # a real incident under concurrent client load produced values like
    # 54,642 tok/s on hardware whose real baseline is ~90-100 tok/s.
    contains_thinking_tags: bool = False  # True if the raw completion
    # text contains a "<think>" tag. This pipeline calls the raw
    # /v1/completions endpoint (no chat template), so it's genuinely
    # unclear whether a hybrid-thinking model (e.g. bare "Qwen3-8B", as
    # opposed to an explicit "-Instruct-2507" non-thinking checkpoint)
    # will spontaneously emit a reasoning preamble here -- this flag
    # lets a benchmark answer that empirically instead of guessing. A
    # thinking preamble is also a real risk to _parse_json_array_batch:
    # if the reasoning text contains a stray "[" before the real JSON
    # array, raw_decode() will try to parse FROM that bracket and fail
    # the whole batch (the earlier trailing-prose fix only protects
    # against text AFTER a valid array, not a bad "[" appearing before it).


@dataclass
class GenerateJsonBatchResult:
    """Result of one batched structured-generation call."""

    items: list[dict]  # one dict per item in the batch, parsed from the model's JSON array
    raw_text: str  # unparsed response, kept for debugging partial-parse failures
    telemetry: CallTelemetry
    parse_failures: list[int] = field(default_factory=list)  # indices that failed to parse


@runtime_checkable
class LLMBackend(Protocol):
    """
    The 4-method contract. Concrete clients implement all of these;
    nothing above this layer is allowed to know which backend it's using.
    """

    async def generate_json_batch(
        self,
        *,
        shared_prefix: str,
        item_prompts: list[str],
        schema_hint: str,
        max_tokens: int,
    ) -> GenerateJsonBatchResult:
        """
        Sends ONE completion call whose prompt is `shared_prefix` (the RAG
        context + system instruction -- kept byte-identical across calls in
        the same job so vLLM's prefix cache actually hits) followed by all
        of `item_prompts` joined into a single batched instruction, asking
        for a JSON array with one object per item. `schema_hint` is the
        literal schema description injected into the prompt.
        """
        ...

    async def embed(self, texts: list[str]) -> tuple[list[list[float]], CallTelemetry]:
        """Embeds a batch of texts in as few requests as possible."""
        ...

    async def health_check(self) -> bool:
        """Cheap liveness probe -- used before a job starts, not per-call."""
        ...

    @property
    def model_name(self) -> str: ...
