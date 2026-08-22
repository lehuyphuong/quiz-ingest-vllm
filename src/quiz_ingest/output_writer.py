"""
output_writer.py

Writes one consolidated JSON file per job: items (with per-item 3-metric
scores), usage (token counts), and latency (wall time + per-stage
breakdown + REAL token/s throughput per stage, including decode). This
mirrors the shape of a per-job summary JSON from an earlier project
(topic/PDF -> quiz + usage + latency), adapted for this repo:

  - source_mode is always "pdf" -- this repo has no web-research/topic
    ingestion mode (see README).
  - No estimated_cost_usd. Self-hosted GPU has no per-token API cost to
    attribute to a job; the real cost is $/hr instance rental over
    however long the instance stays up, which isn't a property of any
    single job. Fabricating a per-job dollar figure here would misstate
    the actual cost model, so it's omitted rather than guessed.
  - Added stage_token_throughput (mean TTFT, mean decode tokens/s, total
    tokens PER STAGE), which the earlier project's vLLM backend couldn't
    populate honestly (its own docstring called decode tps an
    approximation, non-streaming). Every number in here comes from the
    same CallTelemetry objects also written to logs/*.jsonl -- computed
    once, in this module, so the JSONL log and this summary can never
    disagree.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING

from quiz_ingest.llm.base import CallTelemetry

if TYPE_CHECKING:
    from quiz_ingest.pipeline import ScoredQuizItem


def _stage_token_throughput(telemetries_by_stage: dict[str, list[CallTelemetry]]) -> dict:
    out = {}
    for stage, calls in telemetries_by_stage.items():
        ttfts = [c.ttft_s for c in calls if c.ttft_s is not None]
        decode_tps_vals = [
            c.decode_tokens_per_second for c in calls if c.decode_tokens_per_second is not None
        ]
        out[stage] = {
            "call_count": len(calls),
            "total_prompt_tokens": sum(c.prompt_tokens for c in calls),
            "total_completion_tokens": sum(c.completion_tokens for c in calls),
            # None (not 0.0) when a stage has no streaming calls to average
            # (e.g. eval_diversity is embed-only -- no decode phase at all)
            # -- a missing measurement must never look like a real zero.
            "mean_ttft_s": round(mean(ttfts), 4) if ttfts else None,
            "mean_decode_tokens_per_second": round(mean(decode_tps_vals), 2)
            if decode_tps_vals
            else None,
        }
    return out


def build_job_output(
    *,
    source_mode: str,
    source: str,
    scored_items: list["ScoredQuizItem"],
    wall_time_s: float,
    stage_timings_s: dict[str, float],
    telemetries_by_stage: dict[str, list[CallTelemetry]],
) -> dict:
    items_json = []
    for it in scored_items:
        options = [
            {"text": d.get("text", ""), "is_correct": False, "type": d.get("type", "distractor")}
            for d in it.distractors
        ]
        options.append({"text": it.correct_answer, "is_correct": True, "type": "correct"})
        items_json.append(
            {
                "question": it.question,
                "options": options,
                "num_correct": 1,
                "supporting_fact": it.supporting_fact,
                "faithfulness": it.faithfulness,
                "answer_relevance": it.answer_relevance,
                "diversity": it.diversity,
                "source_chunk_ids": it.source_chunk_ids,
            }
        )

    all_calls = [c for calls in telemetries_by_stage.values() for c in calls]
    total_prompt = sum(c.prompt_tokens for c in all_calls)
    total_completion = sum(c.completion_tokens for c in all_calls)
    n_delivered = len(scored_items)

    return {
        "source_mode": source_mode,
        "source": source,
        "items": items_json,
        "usage": {
            "total_calls": len(all_calls),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "avg_tokens_per_question": (
                round((total_prompt + total_completion) / n_delivered, 1) if n_delivered else None
            ),
        },
        "latency": {
            "wall_time_s": round(wall_time_s, 2),
            "stage_timings_s": {k: round(v, 2) for k, v in stage_timings_s.items()},
            "stage_token_throughput": _stage_token_throughput(telemetries_by_stage),
        },
    }


def write_job_output_json(output: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
