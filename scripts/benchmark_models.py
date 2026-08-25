#!/usr/bin/env python
"""
scripts/benchmark_models.py

Benchmarks multiple (generation model, embedding model) PAIRS against the
REAL pipeline (pipeline.run_job) -- not a synthetic completion benchmark --
by managing each pair's vLLM server lifecycle itself: start both servers,
wait for health, run a fixed workload, tear both down, move to the next
pair. One combination is loaded on the GPU at a time.

DESIGN: one-factor-at-a-time, not a full cross product. Testing every
generation model against every embedding model (5 gen x 2 embed = 10
pairs) costs roughly the same GPU time as testing 6 pairs that isolate
each factor separately:
  - baseline_embed x each generation candidate  (isolates: does the
    generation model matter?)
  - baseline_generation x each OTHER embedding candidate (isolates: does
    the embedding model matter?)
This is deliberately NOT exhaustive -- if you want the full cross
product, edit CANDIDATE PAIRS construction in main() directly.

MODEL SELECTION -- reasoning, not just names:
  - All generation candidates are confirmed TEXT-ONLY dense models (no
    multimodal/-VL variants) -- see this repo's earlier lesson with
    Gemma3 (natively multimodal, caused loader errors).
  - Sizes capped at ~9B: this GPU has 24GB total, shared with an
    embedding model running as a SEPARATE vLLM process at the same
    time -- going bigger (14B+) stops fitting comfortably without heavy
    quantization, which would confound "is this model better" with "is
    quantization hurting quality", two different questions.
  - Deliberately spans same-family-different-size (Qwen3-4B vs
    Qwen3-8B) AND different-family-same-size (Qwen3-8B vs GLM-4-9B vs
    Granite-4.1-8B) -- isolates whether SIZE or TRAINING LINEAGE matters
    more for this structured-JSON-batch-generation task, which public
    leaderboards don't tell you (see this repo's earlier model-selection
    discussion).
  - meta-llama/Meta-Llama-3.1-8B-Instruct was considered but is
    DELIBERATELY EXCLUDED here: it's gated on HuggingFace (requires
    requesting access and waiting for approval), and that approval
    wasn't granted yet at the time this candidate list was set -- add it
    back to GENERATION_CANDIDATES once access is confirmed, using
    exactly that repo id (NOT "Llama-3.3-8B", which was never released
    as open weights by Meta -- only reachable through Meta's proprietary
    Llama API).

Usage:
    python scripts/benchmark_models.py \
        --pdf test_docs/sample.pdf \
        --jobs-per-pair 8 --num-questions 5

    # Re-run just a couple of candidates (e.g. after adding a new one):
    python scripts/benchmark_models.py --pdf test_docs/sample.pdf --only qwen3-8b glm4-9b
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx

from quiz_ingest.config import PipelineConfig
from quiz_ingest.logging_setup import LOG_DIR, _log_path
from quiz_ingest.pipeline import run_job

GEN_PORT = 8000
EMBED_PORT = 8001

DEFAULT_TOPICS = [
    "attention mechanism in transformers",
    "training efficiency of the transformer",
    "multi-head attention",
    "positional encoding",
    "future research directions for transformers",
    "comparison with recurrent and convolutional models",
]

GENERATION_CANDIDATES = [
    {
        "id": "qwen3-4b",
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "vram_gb": 8,
        "notes": "Current baseline. Confirmed text-only, working, all prior "
        "load-test tuning (max_tokens budgets etc.) was calibrated against this model.",
    },
    {
        "id": "qwen3-8b",
        "model": "Qwen/Qwen3-8B",
        "vram_gb": 16,
        "notes": "Same family as baseline, 2x size -- isolates whether size "
        "alone helps. CONFIRMED RISK (not hypothetical): unlike the 4B "
        "baseline, there is no '-Instruct-2507' non-thinking release at 8B -- "
        "bare 'Qwen3-8B' is the original hybrid-thinking checkpoint, which "
        "defaults to thinking mode via chat templates. Whether that triggers "
        "through this pipeline's raw /v1/completions call (no chat template) "
        "is unverified -- check the 'thinking_tag_calls' field in this "
        "candidate's result before trusting its other numbers.",
    },
    {
        "id": "glm4-9b",
        "model": "zai-org/GLM-4-9B-0414",
        "vram_gb": 18,
        "notes": "Different training lineage, marketed strength in structured/"
        "tool-call output -- relevant to this pipeline's batched-JSON task.",
    },
    {
        "id": "granite-4.1-8b",
        "model": "ibm-granite/granite-4.1-8b",
        "vram_gb": 16,
        "notes": "Different lineage, enterprise/tool-calling/RAG-oriented, Apache 2.0.",
    },
]

EMBEDDING_CANDIDATES = [
    {
        "id": "qwen3-embed-0.6b",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "vram_gb": 2,
        "notes": "Current baseline.",
    },
    {
        "id": "qwen3-embed-4b",
        "model": "Qwen/Qwen3-Embedding-4B",
        "vram_gb": 9,
        "notes": "Larger embedding model -- isolates whether retrieval/"
        "off-topic-detection quality is currently bottlenecked by embedding size.",
    },
]


def resolve_gpu_utilization_pair(
    gen_vram_gb: float, embed_vram_gb: float, total_vram_gb: float,
    kv_cache_multiplier: float = 1.3, ceiling: float = 0.90,
) -> tuple[float, float]:
    """
    Returns (--gpu-memory-utilization for generation, for embedding) as
    fractions of TOTAL card VRAM. Each candidate's raw weight footprint
    is multiplied by kv_cache_multiplier for KV-cache/activation
    headroom, converted to a fraction of the total card, then BOTH
    fractions are scaled down proportionally if their sum would exceed
    `ceiling` -- leaves margin for CUDA context overhead rather than
    packing the card to 100%.

    This is a rough STARTING ESTIMATE, not a guarantee -- actual usage
    depends on real tokenizer/vocab size, attention implementation, and
    model-specific overhead this function has no way to know. Watch
    `nvidia-smi` on the first run of any new candidate and adjust
    --total-vram-gb or edit vram_gb in the candidate table if a server
    fails to start with an OOM error.
    """
    gen_frac = (gen_vram_gb * kv_cache_multiplier) / total_vram_gb
    embed_frac = (embed_vram_gb * kv_cache_multiplier) / total_vram_gb
    total = gen_frac + embed_frac
    if total > ceiling:
        scale = ceiling / total
        gen_frac *= scale
        embed_frac *= scale
    gen_frac = round(max(0.10, min(0.85, gen_frac)), 2)
    embed_frac = round(max(0.05, min(0.85, embed_frac)), 2)
    return gen_frac, embed_frac


class VLLMServerProcess:
    """
    Manages one `vllm serve` subprocess: start, poll /health until ready
    (or timeout), stop cleanly (SIGTERM, fall back to SIGKILL). NOT
    exercised against a real vLLM install in this environment -- the
    subprocess/health-check mechanics are implemented defensively but
    unverified on real hardware; run this on your instance and report
    back if the health-check timing needs adjusting.
    """

    def __init__(
        self, *, model: str, port: int, gpu_memory_utilization: float,
        max_model_len: int, extra_args: list[str] | None = None, log_path: Path,
    ):
        self.model = model
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.extra_args = extra_args or []
        self.log_path = log_path
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        cmd = [
            "vllm", "serve", self.model,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
            *self.extra_args,
        ]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(self.log_path, "w")
        self._proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    async def wait_until_healthy(self, timeout_s: float, poll_interval_s: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_s
        url = f"http://localhost:{self.port}/health"
        async with httpx.AsyncClient() as client:
            while time.monotonic() < deadline:
                if self._proc is not None and self._proc.poll() is not None:
                    return False  # process exited before becoming healthy -- check log_path
                try:
                    resp = await client.get(url, timeout=5.0)
                    if resp.status_code == 200:
                        return True
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(poll_interval_s)
        return False

    def stop(self, timeout_s: float = 30.0) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=10.0)
        self._proc = None


def collect_decode_tps_since(start_ts: float) -> list[float]:
    """Same approach as scripts/load_test.py -- reads today's JSONL log
    for real decode_tokens_per_second values after start_ts. Values the
    client already discarded as implausible (see vllm_client.py's
    sanity-ceiling fix) are never in this log in the first place, so no
    separate filtering is needed here."""
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


def count_thinking_tag_calls_since(start_ts: float) -> int:
    """
    Counts api_call events (since start_ts) whose raw output contained a
    "<think>" tag -- see llm/base.py's contains_thinking_tags docstring.
    A nonzero count for a candidate means it's spontaneously emitting a
    reasoning preamble even via this pipeline's raw /v1/completions call
    (no chat template) -- worth a manual look at that candidate's
    behavior before trusting its other numbers, since a thinking preamble
    is also a known risk to the JSON array parser (a stray "[" in the
    reasoning text can make the whole batch fail to parse).
    """
    path = _log_path()
    if not path.exists():
        return 0
    count = 0
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                rec.get("event") == "api_call"
                and rec.get("ts", 0) >= start_ts
                and rec.get("contains_thinking_tags")
            ):
                count += 1
    return count


async def run_benchmark_workload(
    *, pdf_path: str, topics: list[str], config: PipelineConfig,
    n_jobs: int, num_questions: int, top_k: int,
) -> dict:
    """
    Runs n_jobs SEQUENTIAL jobs (not concurrent -- this benchmark
    compares MODELS, not load capacity; concurrency effects are
    scripts/load_test.py's job, kept as a separate, orthogonal axis).
    Each job gets a randomized topic, same rationale as load_test.py:
    avoids an unrealistically easy, identical-request-repeated result.
    """
    start_ts = time.time()
    all_items = []
    wall_times = []
    job_failures = 0
    total_requested = 0

    for _ in range(n_jobs):
        topic = random.choice(topics)
        cfg = replace(config, num_questions=num_questions, retrieve_top_k=top_k)
        t0 = time.monotonic()
        total_requested += num_questions
        try:
            items = await run_job(pdf_path=pdf_path, topic=topic, config=cfg)
            wall_times.append(time.monotonic() - t0)
            all_items.extend(items)
        except Exception as exc:  # noqa: BLE001 -- benchmark must survive individual job failures
            job_failures += 1
            print(f"    job failed: {exc!r}")

    decode_tps_samples = collect_decode_tps_since(start_ts)
    thinking_tag_calls = count_thinking_tag_calls_since(start_ts)

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.mean(vals), 3) if vals else None

    return {
        "jobs_run": n_jobs,
        "job_failures": job_failures,
        "delivered_total": len(all_items),
        "requested_total": total_requested,
        "delivery_rate": round(len(all_items) / total_requested, 3) if total_requested else 0.0,
        "mean_faithfulness": _mean(it.faithfulness for it in all_items),
        "mean_answer_relevance": _mean(it.answer_relevance for it in all_items),
        "mean_diversity": _mean(it.diversity for it in all_items),
        "mean_decode_tokens_per_second": _mean(decode_tps_samples),
        "latency_mean_s": _mean(wall_times),
        "latency_p50_s": round(statistics.median(wall_times), 2) if wall_times else None,
        # Nonzero here on a raw-/v1/completions pipeline like this one is
        # worth a manual look -- see count_thinking_tag_calls_since's
        # docstring. Most relevant to bare "Qwen3-8B" (hybrid-thinking
        # default) vs. an explicit "-Instruct-2507" checkpoint.
        "thinking_tag_calls": thinking_tag_calls,
    }


async def benchmark_pair(gen: dict, embed: dict, args: argparse.Namespace) -> dict:
    gen_frac, embed_frac = resolve_gpu_utilization_pair(
        gen["vram_gb"], embed["vram_gb"], args.total_vram_gb
    )
    print(f"\n=== {gen['id']}  x  {embed['id']} ===")
    print(f"    gpu_memory_utilization: gen={gen_frac}  embed={embed_frac}")

    gen_server = VLLMServerProcess(
        model=gen["model"], port=GEN_PORT, gpu_memory_utilization=gen_frac,
        max_model_len=args.max_model_len,
        extra_args=["--enable-prefix-caching"],
        log_path=LOG_DIR / f"vllm_gen_{gen['id']}.log",
    )
    embed_server = VLLMServerProcess(
        model=embed["model"], port=EMBED_PORT, gpu_memory_utilization=embed_frac,
        max_model_len=8192,
        extra_args=["--runner", "pooling"],
        log_path=LOG_DIR / f"vllm_embed_{embed['id']}.log",
    )

    result = {
        "generation_model": gen["id"], "embedding_model": embed["id"],
        "gen_gpu_memory_utilization": gen_frac, "embed_gpu_memory_utilization": embed_frac,
        "status": "unknown",
    }
    try:
        gen_server.start()
        embed_server.start()
        print("    waiting for both servers to become healthy...")
        gen_ok, embed_ok = await asyncio.gather(
            gen_server.wait_until_healthy(args.startup_timeout_s),
            embed_server.wait_until_healthy(args.startup_timeout_s),
        )
        if not gen_ok:
            result["status"] = "generation_server_failed_to_start"
            print(f"    FAILED -- see {gen_server.log_path}")
            return result
        if not embed_ok:
            result["status"] = "embedding_server_failed_to_start"
            print(f"    FAILED -- see {embed_server.log_path}")
            return result

        config = PipelineConfig(
            backend="instance",
            vllm_base_url=f"http://localhost:{GEN_PORT}",
            embed_base_url=f"http://localhost:{EMBED_PORT}",
            model=gen["model"], embed_model=embed["model"],
            num_questions=args.num_questions,
        )
        workload = await run_benchmark_workload(
            pdf_path=args.pdf, topics=args.topics or DEFAULT_TOPICS, config=config,
            n_jobs=args.jobs_per_pair, num_questions=args.num_questions, top_k=args.top_k,
        )
        result.update(workload)
        result["status"] = "completed"
        print(
            f"    delivery_rate={result['delivery_rate']}  "
            f"faithfulness={result['mean_faithfulness']}  "
            f"answer_relevance={result['mean_answer_relevance']}  "
            f"diversity={result['mean_diversity']}  "
            f"decode_tps={result['mean_decode_tokens_per_second']}"
        )
        if result.get("thinking_tag_calls"):
            print(
                f"    ⚠ {result['thinking_tag_calls']} call(s) contained a <think> "
                f"tag -- this model is emitting reasoning preambles even via raw "
                f"/v1/completions. Worth a manual look at "
                f"logs/quiz-ingest-events-*.jsonl (parse_failure raw_text_excerpt) "
                f"before trusting this candidate's other numbers."
            )
    except Exception as exc:  # noqa: BLE001 -- one pair's crash must not kill the whole sweep
        result["status"] = "error"
        result["error"] = repr(exc)
        print(f"    ERROR: {exc!r}")
    finally:
        gen_server.stop()
        embed_server.stop()
        print(f"    cooling down {args.cooldown_s}s for GPU memory to release...")
        await asyncio.sleep(args.cooldown_s)
    return result


def build_pairs(args: argparse.Namespace) -> list[tuple[dict, dict]]:
    baseline_embed = EMBEDDING_CANDIDATES[0]
    baseline_gen = GENERATION_CANDIDATES[0]

    if args.only:
        # Restricted mode: ONLY build pairs from what was explicitly
        # named, each against the baseline on the other axis. Does NOT
        # also run the full sweep on the other axis -- e.g. `--only
        # glm4-9b` means "just benchmark glm4-9b against baseline embed",
        # not "also re-run the whole embedding sweep against baseline gen".
        wanted = set(args.only)
        gen_matches = [g for g in GENERATION_CANDIDATES if g["id"] in wanted]
        embed_matches = [e for e in EMBEDDING_CANDIDATES if e["id"] in wanted]
        pairs: list[tuple[dict, dict]] = []
        seen = set()
        for g in gen_matches:
            key = (g["id"], baseline_embed["id"])
            if key not in seen:
                pairs.append((g, baseline_embed))
                seen.add(key)
        for e in embed_matches:
            key = (baseline_gen["id"], e["id"])
            if key not in seen:
                pairs.append((baseline_gen, e))
                seen.add(key)
        return pairs

    pairs = []
    seen = set()
    if not args.skip_generation_sweep:
        for g in GENERATION_CANDIDATES:
            key = (g["id"], baseline_embed["id"])
            if key not in seen:
                pairs.append((g, baseline_embed))
                seen.add(key)
    if not args.skip_embedding_sweep:
        for e in EMBEDDING_CANDIDATES:
            key = (baseline_gen["id"], e["id"])
            if key not in seen:
                pairs.append((baseline_gen, e))
                seen.add(key)
    return pairs


def print_summary_table(results: list[dict]) -> None:
    header = (
        f"{'gen':<16} {'embed':<18} {'status':<28} {'delivery':<9} "
        f"{'faithful':<9} {'relevance':<10} {'diversity':<9} {'decode_tps':<11}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['generation_model']:<16} {r['embedding_model']:<18} {r['status']:<28} "
            f"{r.get('delivery_rate', '-'):<9} {r.get('mean_faithfulness', '-'):<9} "
            f"{r.get('mean_answer_relevance', '-'):<10} {r.get('mean_diversity', '-'):<9} "
            f"{r.get('mean_decode_tokens_per_second', '-'):<11}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdf", required=True)
    p.add_argument("--topics", nargs="+", default=None)
    p.add_argument("--num-questions", type=int, default=5)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--jobs-per-pair", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--total-vram-gb", type=float, default=24.0)
    p.add_argument("--startup-timeout-s", type=float, default=600.0)
    p.add_argument("--cooldown-s", type=float, default=15.0)
    p.add_argument(
        "--only", nargs="+", default=None,
        help="Restrict to these candidate ids (generation and/or embedding). "
        "Default: run the full one-factor-at-a-time sweep.",
    )
    p.add_argument("--skip-generation-sweep", action="store_true")
    p.add_argument("--skip-embedding-sweep", action="store_true")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    pairs = build_pairs(args)
    print(f"Benchmarking {len(pairs)} pair(s):")
    for g, e in pairs:
        print(f"  {g['id']}  x  {e['id']}")

    results = []
    for gen, embed in pairs:
        result = await benchmark_pair(gen, embed, args)
        results.append(result)

    print_summary_table(results)

    out_path = LOG_DIR / f"benchmark_models_{int(time.time())}.json"
    LOG_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nFull results: {out_path}")
    print(f"Per-server startup logs: logs/vllm_gen_*.log, logs/vllm_embed_*.log")


if __name__ == "__main__":
    asyncio.run(main())
