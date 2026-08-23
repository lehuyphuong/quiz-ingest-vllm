#!/usr/bin/env python
"""
scripts/generate_quiz_from_pdf.py

python scripts/generate_quiz_from_pdf.py --pdf doc.pdf --topic "history of rome" \
    --vllm-base-url http://<instance-ip>:8000 \
    --embed-base-url http://<instance-ip>:8001 \
    --num-questions 5

--embed-base-url defaults to --vllm-base-url if omitted (single instance
serving both roles via 2 vLLM processes on different ports -- see
README's "Running 2 vLLM processes on 1 GPU" section).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quiz_ingest.config import PipelineConfig
from quiz_ingest.pipeline import run_job_streaming


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdf", required=True)
    p.add_argument("--topic", required=True, help="Used to guide retrieval/rerank within the PDF")
    p.add_argument("--num-questions", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--vllm-base-url", default=None, help="Required when --backend instance (default)")
    p.add_argument("--embed-base-url", default=None)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--chunk-overlap", type=int, default=150)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument(
        "--backend", choices=["instance", "serverless"], default="instance",
        help="'instance' talks to a plain rented GPU (--vllm-base-url). "
        "'serverless' talks to Vast Serverless Endpoints (--vast-endpoint-name).",
    )
    p.add_argument("--vast-endpoint-name", default=None, help="Required when --backend serverless")
    p.add_argument(
        "--vast-embed-endpoint-name", default=None,
        help="Defaults to --vast-endpoint-name if omitted (only valid if that "
        "one endpoint also serves the embedding model, which it normally "
        "won't -- see README, generation and embedding are separate vLLM processes)",
    )
    p.add_argument(
        "--output-json",
        default=None,
        help="Path for the consolidated job summary JSON (items + usage + "
        "latency incl. per-stage token/s). Defaults to "
        "outputs/quiz_output_<pdf-stem>_<timestamp>.json",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    if args.backend == "instance" and not args.vllm_base_url:
        raise SystemExit("--vllm-base-url is required when --backend instance (the default)")
    if args.backend == "serverless" and not args.vast_endpoint_name:
        raise SystemExit("--vast-endpoint-name is required when --backend serverless")

    config = PipelineConfig(
        backend=args.backend,
        vllm_base_url=args.vllm_base_url,
        embed_base_url=args.embed_base_url or args.vllm_base_url,
        vast_endpoint_name=args.vast_endpoint_name,
        vast_embed_endpoint_name=args.vast_embed_endpoint_name or args.vast_endpoint_name,
        model=args.model,
        embed_model=args.embed_model,
        num_questions=args.num_questions,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        retrieve_top_k=args.top_k,
    )

    n = 0
    output_json_path = args.output_json or (
        f"outputs/quiz_output_{Path(args.pdf).stem}_{int(time.time())}.json"
    )
    async for group in run_job_streaming(
        pdf_path=args.pdf, topic=args.topic, config=config, output_json_path=output_json_path
    ):
        for item in group:
            n += 1
            print("=" * 70)
            print(f"Question {n}")
            print("=" * 70)
            print(f"Q: {item.question}\n")
            print(f"Correct: {item.correct_answer}")
            for d in item.distractors:
                print(f"  [{d.get('type', '?')}] {d.get('text', '')}")
            print(
                f"\nScores: faithfulness={item.faithfulness} "
                f"answer_relevance={item.answer_relevance} "
                f"diversity={item.diversity}"
            )
            print()

    print(f"Done. Delivered {n}/{args.num_questions} questions.")
    print(f"Per-call telemetry (JSONL): logs/")
    print(f"Consolidated job output (JSON): {output_json_path}")


if __name__ == "__main__":
    asyncio.run(main())
