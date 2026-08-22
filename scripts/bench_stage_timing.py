#!/usr/bin/env python
"""
scripts/bench_stage_timing.py

Runs the SAME (pdf, topic) job repeatedly across a grid of batch sizes
(and, optionally, models -- pass --model multiple times), then prints a
table of stage_timings_s + mean 3-metric scores per configuration, read
back from logs/ (not recomputed) so this script and generate_quiz_from_pdf.py
share one source of truth.

Prefix caching itself is a server-side vLLM flag (--enable-prefix-caching)
-- this script does NOT toggle it (that requires restarting the vLLM
server with a different flag). To compare on/off, run this script twice
against two different vLLM server processes/ports, one started with the
flag and one without, and diff the resulting tables.

Usage:
    python scripts/bench_stage_timing.py --pdf doc.pdf --topic "history of rome" \
        --vllm-base-url http://<ip>:8000 --embed-base-url http://<ip>:8001 \
        --batch-sizes 4 8 16 32 --num-questions 16 --repeats 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quiz_ingest.config import PipelineConfig
from quiz_ingest.logging_setup import LOG_DIR, _log_path
from quiz_ingest.pipeline import run_job


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdf", required=True)
    p.add_argument("--topic", required=True)
    p.add_argument("--vllm-base-url", required=True)
    p.add_argument("--embed-base-url", default=None)
    p.add_argument("--model", action="append", default=None, help="repeatable")
    p.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--num-questions", type=int, default=16)
    p.add_argument("--repeats", type=int, default=1)
    return p.parse_args()


async def run_one(args, model: str, batch_size: int, repeat_idx: int) -> None:
    config = PipelineConfig(
        backend="instance",
        vllm_base_url=args.vllm_base_url,
        embed_base_url=args.embed_base_url or args.vllm_base_url,
        model=model,
        embed_model=args.embed_model,
        num_questions=args.num_questions,
        batch_size=batch_size,
    )
    tag = f"model={model} batch={batch_size} rep={repeat_idx}"
    print(f"--- running {tag} ---")
    try:
        results = await run_job(pdf_path=args.pdf, topic=args.topic, config=config)
        print(f"    delivered {len(results)}/{args.num_questions}")
    except Exception as exc:  # noqa: BLE001 -- benchmark script, keep going on failure
        print(f"    FAILED: {exc}")


def summarize_logs() -> None:
    """Reads today's job_summary events and prints a comparison table."""
    path = _log_path()
    if not path.exists():
        print("No logs found.")
        return

    rows = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event") == "job_summary":
                rows.append(rec)

    if not rows:
        print("No job_summary events logged.")
        return

    header = (
        f"{'status':<10} {'delivered':<10} {'wall_s':<8} "
        f"{'gen_q_s':<9} {'gen_d_s':<9} {'eval_s':<9} "
        f"{'faith':<7} {'rel':<7} {'div':<7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        st = r.get("stage_timings_s", {})
        ms = r.get("mean_scores", {})
        eval_s = st.get("eval_faithfulness_relevance", 0.0) + st.get("eval_diversity", 0.0)
        print(
            f"{r['status']:<10} {r['delivered']}/{r['requested']:<8} "
            f"{r['wall_time_s']:<8.1f} "
            f"{st.get('generate_questions', 0.0):<9.1f} "
            f"{st.get('generate_distractors', 0.0):<9.1f} "
            f"{eval_s:<9.1f} "
            f"{ms.get('faithfulness', 0.0):<7.2f} "
            f"{ms.get('answer_relevance', 0.0):<7.2f} "
            f"{ms.get('diversity', 0.0):<7.2f}"
        )
    print(f"\nFull per-call telemetry (ttft_s, decode_tokens_per_second) is in {path}")


async def main() -> None:
    args = parse_args()
    models = args.model or ["Qwen/Qwen3-4B-Instruct-2507"]

    for model in models:
        for batch_size in args.batch_sizes:
            for rep in range(args.repeats):
                await run_one(args, model, batch_size, rep)

    print()
    summarize_logs()


if __name__ == "__main__":
    asyncio.run(main())
