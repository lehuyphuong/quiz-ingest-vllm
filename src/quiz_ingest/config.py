"""
config.py

PipelineConfig.backend is currently always "instance" -- the "serverless"
value is accepted (constructs VastServerlessClient instead) but not the
default, and not exercised by scripts/ until instance mode is validated
end to end (see llm/vast_serverless_client.py's docstring).
"""
from __future__ import annotations

from dataclasses import dataclass

MAX_NUM_QUESTIONS = 50
MAX_TOPIC_CHARS = 200


class InvalidRequestError(ValueError):
    pass


@dataclass
class PipelineConfig:
    backend: str = "instance"  # "instance" | "serverless"

    # instance mode
    vllm_base_url: str | None = None  # generation model, e.g. http://<ip>:8000
    embed_base_url: str | None = None  # embedding model -- separate vLLM process/instance
    model: str = "google/gemma-3-4b-it"
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # serverless mode (only read when backend == "serverless")
    vast_endpoint_name: str | None = None
    vast_embed_endpoint_name: str | None = None

    # generation
    num_questions: int = 5
    batch_size: int = 8
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieve_top_k: int = 12

    # vLLM server-side flags this repo ASSUMES are set when the instance
    # was started (documented here so a mismatch is easy to spot, not
    # enforced in code -- see scripts/README's `vllm serve` command):
    #   --enable-prefix-caching
    #   --kv-cache-dtype fp8   (optional, trade-off -- see README)

    def __post_init__(self) -> None:
        if not (1 <= self.num_questions <= MAX_NUM_QUESTIONS):
            raise InvalidRequestError(
                f"num_questions must be 1-{MAX_NUM_QUESTIONS}, got {self.num_questions}"
            )
        if self.backend == "instance" and not self.vllm_base_url:
            raise InvalidRequestError("vllm_base_url is required when backend='instance'")
        if self.backend == "serverless" and not self.vast_endpoint_name:
            raise InvalidRequestError("vast_endpoint_name is required when backend='serverless'")
