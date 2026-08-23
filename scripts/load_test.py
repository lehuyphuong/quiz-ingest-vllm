#!/usr/bin/env python
"""
scripts/load_test.py

Load-tests the instance-mode vLLM backend by running N concurrent
end-to-end quiz-generation jobs (pipeline.run_job) at each concurrency
level in --concurrency-levels -- this measures "how many concurrent
users can this handle" against the REAL code path (retrieval, batched
generation, eval, repair), not a synthetic raw-completion benchmark.

Per the team's requirement, each simulated user gets:
  - a randomized topic (from --topics, or a small built-in list for the
    "Attention Is All You Need" sample PDF) -- varies CONTEXT SIZE, since
    different topics retrieve different chunks via RagIndex
  - a randomized num_questions (1..MAX_NUM_QUESTIONS) and retrieve_top_k
    (--min-top-k..--max-top-k) -- varies PROMPT SIZE (batch size, context
    block length)

Deliberately NOT identical repeated requests: real users querying
different topics wouldn't share a cache-friendly prefix the way N copies
of the same request would, so identical requests would give an
unrealistically rosy result.

Usage:
    python scripts/load_test.py \
        --pdf test_docs/sample.pdf \
        --vllm-base-url http://localhost:8000 --embed-base-url http://localhost:8001 \
        --concurrency-levels 1 2 4 8 16 \
        --requests-per-level 20

Increase --concurrency-levels gradually and watch p95 latency / success
rate / mean decode tok/s per level (the last is read back from
logs/*.jsonl -- the same real per-call telemetry every other script in
this repo uses, not re-measured here). The script auto-stops once a
level shows clear degradation (see --degradation-latency-multiplier /
--degradation-success-rate) and reports an estimated sustainable
concurrency -- treat that as a starting estimate to sanity-check, not a
guaranteed production number.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quiz_ingest.config import MAX_NUM_QUESTIONS, PipelineConfig
from quiz_ingest.logging_setup import LOG_DIR, _log_path
from quiz_ingest.pipeline import run_job

DEFAULT_TOPICS = [
    "attention mechanism in transformers",
    "training efficiency of the transformer",
    "multi-head attention",
    "positional encoding",
    "future research directions for transformers",
    "comparison with recurrent and convolutional models",
]


@dataclass
class JobResult:
    ok: bool
    wall_s: float
    requested: int
    top_k: int
    delivered: int = 0
    error: str | None = None


async def run_one_job(
    *, pdf_path: str, topic: str, base_config: PipelineConfig, num_questions: int, top_k: int
) -> JobResult:
    cfg = replace(base_config, num_questions=num_questions, retrieve_top_k=top_k)
    t0 = time.monotonic()
    try:
        items = await run_job(pdf_path=pdf_path, topic=topic, config=cfg)
        return JobResult(
            ok=True, wall_s=time.monotonic() - t0,
            requested=num_questions, top_k=top_k, delivered=len(items),
        )
    except Exception as exc:  # noqa: BLE001 -- load test must survive individual job failures
        return JobResult(
            ok=False, wall_s=time.monotonic() - t0,
            requested=num_questions, top_k=top_k, error=repr(exc),
        )


async def run_level(
    *, concurrency: int, total_requests: int, pdf_path: str, topics: list[str],
    base_config: PipelineConfig, top_k_range: tuple[int, int],
) -> list[JobResult]:
    """
    Caps true in-flight concurrency at `concurrency` via a Semaphore
    acquired around the run_job call -- wall_s is measured from BEFORE
    acquiring the semaphore, so queueing time under load is counted
    (a request stuck waiting for a slot is exactly the degradation this
    script exists to detect, not something to hide from the numbers).
    """
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(i: int) -> JobResult:
        topic = random.choice(topics)
        num_q = random.randint(1, MAX_NUM_QUESTIONS)
        top_k = random.randint(*top_k_range)
        t0 = time.monotonic()
        async with sem:
            result = await run_one_job(
                pdf_path=pdf_path, topic=topic, base_config=base_config,
                num_questions=num_q, top_k=top_k,
            )
        # wall_s from run_one_job excludes queue wait; overwrite with the
        # full including-queue-wait duration measured here.
        result.wall_s = time.monotonic() - t0
        return result

    return list(await asyncio.gather(*(_bounded(i) for i in range(total_requests))))


def collect_decode_tps_since(start_ts: float) -> list[float]:
    """Reads today's JSONL log for api_call events after start_ts with a
    real decode_tokens_per_second -- reused, not re-measured, so this
    number always matches what every other script in this repo reports."""
    path = _log_path()
    if not path.exists():
        return []
    values = []
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                rec.get("event") == "api_call"
                and rec.get("ts", 0) >= start_ts
                and rec.get("decode_tokens_per_second") is not None
            ):
                values.append(rec["decode_tokens_per_second"])
    return values


def summarize_level(results: list[JobResult], decode_tps_samples: list[float]) -> dict:
    ok_results = [r for r in results if r.ok]
    latencies = [r.wall_s for r in ok_results]
    success_rate = len(ok_results) / len(results) if results else 0.0
    summary = {
        "total_requests": len(results),
        "success_count": len(ok_results),
        "success_rate": round(success_rate, 3),
        "delivered_total": sum(r.delivered for r in ok_results),
        "requested_total": sum(r.requested for r in results),
    }
    if latencies:
        sorted_lat = sorted(latencies)
        summary.update({
            "latency_mean_s": round(statistics.mean(latencies), 2),
            "latency_p50_s": round(statistics.median(sorted_lat), 2),
            "latency_p95_s": round(sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))], 2),
            "latency_max_s": round(max(latencies), 2),
        })
    else:
        summary.update({"latency_mean_s": None, "latency_p50_s": None, "latency_p95_s": None, "latency_max_s": None})
    summary["mean_decode_tokens_per_second"] = (
        round(statistics.mean(decode_tps_samples), 1) if decode_tps_samples else None
    )
    if not ok_results:
        summary["sample_errors"] = list({r.error for r in results if r.error})[:3]
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdf", required=True)
    p.add_argument("--vllm-base-url", required=True)
    p.add_argument("--embed-base-url", default=None)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--concurrency-levels", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--requests-per-level", type=int, default=20)
    p.add_argument("--min-top-k", type=int, default=4)
    p.add_argument("--max-top-k", type=int, default=20)
    p.add_argument("--topics", nargs="+", default=None, help="Defaults to a built-in list for the sample PDF")
    p.add_argument(
        "--degradation-latency-multiplier", type=float, default=3.0,
        help="Stop escalating once a level's p95 latency exceeds this multiple of level-1's p95",
    )
    p.add_argument(
        "--degradation-success-rate", type=float, default=0.8,
        help="Stop escalating once a level's success rate drops below this",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    topics = args.topics or DEFAULT_TOPICS
    base_config = PipelineConfig(
        backend="instance",
        vllm_base_url=args.vllm_base_url,
        embed_base_url=args.embed_base_url or args.vllm_base_url,
        model=args.model,
        embed_model=args.embed_model,
        num_questions=1,  # overridden per-request in run_level
    )

    baseline_p95 = None
    level_summaries = {}
    sustainable_concurrency = None

    print(f"{'concurrency':<12} {'success_rate':<13} {'p50_s':<8} {'p95_s':<8} {'max_s':<8} {'decode_tps':<11}")
    print("-" * 65)

    for concurrency in args.concurrency_levels:
        start_ts = time.time()
        results = await run_level(
            concurrency=concurrency, total_requests=args.requests_per_level,
            pdf_path=args.pdf, topics=topics, base_config=base_config,
            top_k_range=(args.min_top_k, args.max_top_k),
        )
        decode_tps_samples = collect_decode_tps_since(start_ts)
        summary = summarize_level(results, decode_tps_samples)
        level_summaries[concurrency] = summary

        print(
            f"{concurrency:<12} {summary['success_rate']:<13} "
            f"{summary['latency_p50_s'] or '-':<8} {summary['latency_p95_s'] or '-':<8} "
            f"{summary['latency_max_s'] or '-':<8} {summary['mean_decode_tokens_per_second'] or '-':<11}"
        )

        if baseline_p95 is None and summary["latency_p95_s"] is not None:
            baseline_p95 = summary["latency_p95_s"]

        degraded = False
        if summary["success_rate"] < args.degradation_success_rate:
            print(f"  -> success rate dropped below {args.degradation_success_rate}, stopping here")
            degraded = True
        elif (
            baseline_p95 and summary["latency_p95_s"]
            and summary["latency_p95_s"] > baseline_p95 * args.degradation_latency_multiplier
        ):
            print(
                f"  -> p95 latency {summary['latency_p95_s']}s exceeds "
                f"{args.degradation_latency_multiplier}x baseline ({baseline_p95}s), stopping here"
            )
            degraded = True

        if not degraded:
            sustainable_concurrency = concurrency
        else:
            break

    print()
    if sustainable_concurrency is not None:
        print(f"Estimated sustainable concurrency: {sustainable_concurrency} concurrent users")
        print("(last level tested before degradation -- verify with a longer soak test before trusting this for capacity planning)")
    else:
        print("Degradation appeared at the very first concurrency level tested -- start lower.")

    LOG_DIR.mkdir(exist_ok=True)
    out_path = LOG_DIR / f"load_test_summary_{int(time.time())}.json"
    out_path.write_text(json.dumps({str(k): v for k, v in level_summaries.items()}, indent=2))
    print(f"\nFull per-level summary: {out_path}")
    print(f"Per-request telemetry (real ttft/decode-tps for every call made during this test): logs/quiz-ingest-events-*.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
