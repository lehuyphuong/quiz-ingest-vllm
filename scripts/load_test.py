#!/usr/bin/env python
"""
scripts/load_test.py

Load-tests the instance-mode vLLM backend by running `concurrency`
parallel WORKERS at each level in --concurrency-levels, each worker
firing --requests-per-worker requests SEQUENTIALLY -- this measures
"how many concurrent users can this handle" against the REAL code path
(retrieval, batched generation, eval, repair, off-topic filtering), not
a synthetic raw-completion benchmark.

METHODOLOGY NOTE (fixed from an earlier version of this script): total
requests submitted now SCALES with concurrency (concurrency *
requests_per_worker), not a fixed pool shared across all levels. An
earlier version fired a fixed --requests-per-level total at every
concurrency level via one shared queue -- at low concurrency, most of
those requests spent most of their measured time waiting in line behind
each other (a burst-arrival artifact), not reflecting real per-request
service time, which made latency look like it improved dramatically as
concurrency increased. It didn't; the queue just got shorter. Each
worker here loops through its own requests one at a time, so at any
instant exactly `concurrency` requests are genuinely in flight --
that's the steady-state measurement this tool is meant to produce.

Per the team's requirement, each request gets a randomized topic
(varies retrieved context size) and randomized num_questions/top_k
(varies prompt size) -- deliberately not identical repeated requests,
see DEFAULT_TOPICS below.

DELIVERY RATE, not just success rate: a job that completes without
raising an exception can still deliver ZERO of the questions it was
asked for (parse failures, off-topic content dropped, distractors that
couldn't be repaired -- see README "Eval score integrity" /
"Cross-group question dedup" / detect_off_topic_indices). An earlier
version of this script only reported success_rate (did the job crash?)
which looked perfect (1.0) even in a real run where 71% of jobs
delivered nothing. delivery_rate (delivered_total / requested_total) is
now a primary, always-reported metric, not an afterthought.

Usage:
    python scripts/load_test.py \
        --pdf test_docs/sample.pdf \
        --vllm-base-url http://localhost:8000 --embed-base-url http://localhost:8001 \
        --concurrency-levels 1 2 4 8 16 \
        --requests-per-worker 5
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
    wall_s: float  # true per-request service time -- each worker is sequential, so this is never queue-wait-inflated
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
    *, concurrency: int, requests_per_worker: int, pdf_path: str, topics: list[str],
    base_config: PipelineConfig, top_k_range: tuple[int, int],
) -> list[JobResult]:
    """
    `concurrency` worker coroutines run in parallel, each firing
    `requests_per_worker` requests ONE AT A TIME. Total requests at this
    level = concurrency * requests_per_worker -- scales with concurrency,
    unlike a fixed shared pool (see module docstring for why that
    mattered). At any instant, in-flight requests == concurrency exactly
    (steady state), never more, never fewer once all workers have started.
    """

    async def worker(worker_id: int) -> list[JobResult]:
        results = []
        for _ in range(requests_per_worker):
            topic = random.choice(topics)
            num_q = random.randint(1, MAX_NUM_QUESTIONS)
            top_k = random.randint(*top_k_range)
            result = await run_one_job(
                pdf_path=pdf_path, topic=topic, base_config=base_config,
                num_questions=num_q, top_k=top_k,
            )
            results.append(result)
        return results

    per_worker_results = await asyncio.gather(*(worker(i) for i in range(concurrency)))
    return [r for worker_results in per_worker_results for r in worker_results]


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
    delivered_total = sum(r.delivered for r in ok_results)
    requested_total = sum(r.requested for r in results)
    delivery_rate = delivered_total / requested_total if requested_total else 0.0

    summary = {
        "total_requests": len(results),
        "success_count": len(ok_results),
        "success_rate": round(success_rate, 3),
        # PRIMARY quality signal -- a job that didn't crash can still
        # deliver nothing it was asked for (see module docstring).
        # Do not treat success_rate alone as "it worked".
        "delivered_total": delivered_total,
        "requested_total": requested_total,
        "delivery_rate": round(delivery_rate, 3),
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
    p.add_argument(
        "--requests-per-worker", type=int, default=5,
        help="Each of the `concurrency` workers fires this many requests SEQUENTIALLY. "
        "Total requests at a level = concurrency * requests-per-worker.",
    )
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
    p.add_argument(
        "--degradation-delivery-rate", type=float, default=0.5,
        help="Stop escalating once a level's delivery rate (delivered/requested "
        "questions, not just non-crashed jobs) drops below this",
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

    header = (
        f"{'concurrency':<12} {'total_req':<10} {'success_rate':<13} {'delivery_rate':<14} "
        f"{'p50_s':<8} {'p95_s':<8} {'max_s':<8} {'decode_tps':<11}"
    )
    print(header)
    print("-" * len(header))

    for concurrency in args.concurrency_levels:
        start_ts = time.time()
        results = await run_level(
            concurrency=concurrency, requests_per_worker=args.requests_per_worker,
            pdf_path=args.pdf, topics=topics, base_config=base_config,
            top_k_range=(args.min_top_k, args.max_top_k),
        )
        decode_tps_samples = collect_decode_tps_since(start_ts)
        summary = summarize_level(results, decode_tps_samples)
        level_summaries[concurrency] = summary

        print(
            f"{concurrency:<12} {summary['total_requests']:<10} "
            f"{summary['success_rate']:<13} {summary['delivery_rate']:<14} "
            f"{summary['latency_p50_s'] or '-':<8} {summary['latency_p95_s'] or '-':<8} "
            f"{summary['latency_max_s'] or '-':<8} {summary['mean_decode_tokens_per_second'] or '-':<11}"
        )

        if baseline_p95 is None and summary["latency_p95_s"] is not None:
            baseline_p95 = summary["latency_p95_s"]

        degraded = False
        if summary["success_rate"] < args.degradation_success_rate:
            print(f"  -> success rate dropped below {args.degradation_success_rate}, stopping here")
            degraded = True
        elif summary["delivery_rate"] < args.degradation_delivery_rate:
            print(
                f"  -> delivery rate ({summary['delivery_rate']}) dropped below "
                f"{args.degradation_delivery_rate} -- jobs are 'succeeding' but not "
                f"delivering what was asked, stopping here"
            )
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
    print("Per-request telemetry (real ttft/decode-tps for every call made during this test): logs/quiz-ingest-events-*.jsonl")
    print('If delivery_rate looks low at any level, grep logs/*.jsonl for "event": "parse_failure" to see why (raw model output included).')


if __name__ == "__main__":
    asyncio.run(main())
