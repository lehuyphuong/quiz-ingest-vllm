"""
logging_setup.py

Writes JSONL events to logs/quiz-ingest-events-YYYY-MM-DD.jsonl. Two event
types:

  - api_call: one per LLM/embed call, carries the FULL CallTelemetry
    (ttft_s, decode_tokens_per_second, prefix_cache_hit_tokens) --
    unlike the earlier project's `vllm` backend, these are real
    measurements here (see llm/vllm_client.py), not approximations.
  - job_summary: one per finished job, with stage_timings_s broken out
    per pipeline stage (index_build, generate_questions, generate_distractors,
    eval_faithfulness_relevance, eval_diversity) so a latency regression
    can be attributed to a specific stage instead of just one aggregate
    wall_time_s.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from quiz_ingest.llm.base import CallTelemetry

LOG_DIR = Path("logs")


def _log_path() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    return LOG_DIR / f"quiz-ingest-events-{date.today().isoformat()}.jsonl"


def log_parse_failure(
    *, stage: str, raw_text: str, failed_indices: list[int], total_count: int
) -> None:
    """
    Written whenever a batch JSON response has ANY parse failures --
    including a full-batch failure, which previously left NO trace: the
    group would just silently disappear (assemble_quiz_items returns
    empty, pipeline.py skips to the next group) with nothing in the log
    explaining why. raw_text is truncated to keep log lines a sane size --
    this is for diagnosing WHY parsing failed (malformed JSON, wrapped in
    prose, truncated at max_tokens, schema drift under a larger batch
    size), not for replaying the call.
    """
    record = {
        "event": "parse_failure",
        "stage": stage,
        "ts": time.time(),
        "failed_count": len(failed_indices),
        "total_count": total_count,
        "failed_indices": failed_indices,
        "raw_text_excerpt": raw_text[:4000],
    }
    with open(_log_path(), "a") as f:
        f.write(json.dumps(record) + "\n")


def log_api_call(telemetry: CallTelemetry, *, stage: str) -> None:
    record = {"event": "api_call", "stage": stage, "ts": time.time(), **asdict(telemetry)}
    with open(_log_path(), "a") as f:
        f.write(json.dumps(record) + "\n")


@dataclass
class JobTimer:
    """Accumulates stage_timings_s across one job's lifetime."""

    stage_timings_s: dict[str, float] = field(default_factory=dict)
    _stage_start: float | None = None
    _current_stage: str | None = None

    def start_stage(self, stage: str) -> None:
        self._current_stage = stage
        self._stage_start = time.monotonic()

    def end_stage(self) -> None:
        if self._current_stage is None or self._stage_start is None:
            return
        elapsed = time.monotonic() - self._stage_start
        self.stage_timings_s[self._current_stage] = (
            self.stage_timings_s.get(self._current_stage, 0.0) + elapsed
        )
        self._current_stage = None
        self._stage_start = None


def log_job_summary(
    *,
    status: str,
    num_delivered: int,
    num_requested: int,
    wall_time_s: float,
    stage_timings_s: dict[str, float],
    mean_scores: dict[str, float],
) -> None:
    record = {
        "event": "job_summary",
        "ts": time.time(),
        "status": status,
        "delivered": num_delivered,
        "requested": num_requested,
        "wall_time_s": wall_time_s,
        "stage_timings_s": stage_timings_s,
        "mean_scores": mean_scores,
    }
    with open(_log_path(), "a") as f:
        f.write(json.dumps(record) + "\n")
