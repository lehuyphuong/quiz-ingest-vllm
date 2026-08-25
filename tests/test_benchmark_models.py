import importlib.util
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

_spec = importlib.util.spec_from_file_location("benchmark_models", REPO_ROOT / "scripts" / "benchmark_models.py")
benchmark_models = importlib.util.module_from_spec(_spec)
sys.modules["benchmark_models"] = benchmark_models
_spec.loader.exec_module(benchmark_models)


# ---------------------------------------------------------------------------
# resolve_gpu_utilization_pair
# ---------------------------------------------------------------------------

def test_gpu_utilization_small_pair_fits_comfortably():
    gen_frac, embed_frac = benchmark_models.resolve_gpu_utilization_pair(
        gen_vram_gb=8, embed_vram_gb=2, total_vram_gb=24
    )
    assert gen_frac + embed_frac <= 0.90
    assert gen_frac > embed_frac  # bigger model gets bigger share


def test_gpu_utilization_scales_down_when_oversubscribed():
    # 18GB gen + 9GB embed on a 24GB card, with 1.3x headroom, way exceeds
    # capacity -- must scale BOTH down proportionally, not just clip one.
    gen_frac, embed_frac = benchmark_models.resolve_gpu_utilization_pair(
        gen_vram_gb=18, embed_vram_gb=9, total_vram_gb=24
    )
    assert gen_frac + embed_frac <= 0.90
    # ratio preserved: gen should still be noticeably bigger than embed
    assert gen_frac > embed_frac


def test_gpu_utilization_never_below_floor_or_above_ceiling():
    gen_frac, embed_frac = benchmark_models.resolve_gpu_utilization_pair(
        gen_vram_gb=0.5, embed_vram_gb=0.1, total_vram_gb=24
    )
    assert 0.10 <= gen_frac <= 0.85
    assert 0.05 <= embed_frac <= 0.85


# ---------------------------------------------------------------------------
# build_pairs
# ---------------------------------------------------------------------------

def _fake_args(**overrides):
    class Args:
        only = None
        skip_generation_sweep = False
        skip_embedding_sweep = False
    a = Args()
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def test_build_pairs_default_is_one_factor_at_a_time():
    pairs = benchmark_models.build_pairs(_fake_args())
    baseline_embed_id = benchmark_models.EMBEDDING_CANDIDATES[0]["id"]
    baseline_gen_id = benchmark_models.GENERATION_CANDIDATES[0]["id"]

    # every generation candidate appears paired with the baseline embed
    gen_ids_with_baseline_embed = {
        g["id"] for g, e in pairs if e["id"] == baseline_embed_id
    }
    assert gen_ids_with_baseline_embed == {g["id"] for g in benchmark_models.GENERATION_CANDIDATES}

    # every NON-baseline embedding candidate appears paired with the baseline gen
    embed_ids_with_baseline_gen = {
        e["id"] for g, e in pairs if g["id"] == baseline_gen_id
    }
    assert embed_ids_with_baseline_gen == {e["id"] for e in benchmark_models.EMBEDDING_CANDIDATES}

    # the (baseline_gen, baseline_embed) pair is not duplicated
    baseline_pair_count = sum(
        1 for g, e in pairs if g["id"] == baseline_gen_id and e["id"] == baseline_embed_id
    )
    assert baseline_pair_count == 1


def test_build_pairs_only_filter_still_pairs_with_baseline():
    pairs = benchmark_models.build_pairs(_fake_args(only=["glm4-9b"]))
    assert len(pairs) == 1
    gen, embed = pairs[0]
    assert gen["id"] == "glm4-9b"
    assert embed["id"] == benchmark_models.EMBEDDING_CANDIDATES[0]["id"]


def test_build_pairs_skip_flags_respected():
    pairs = benchmark_models.build_pairs(_fake_args(skip_embedding_sweep=True))
    baseline_embed_id = benchmark_models.EMBEDDING_CANDIDATES[0]["id"]
    assert all(e["id"] == baseline_embed_id for _, e in pairs)


# ---------------------------------------------------------------------------
# run_benchmark_workload (mocked pipeline.run_job)
# ---------------------------------------------------------------------------

class _FakeScoredItem:
    def __init__(self, faithfulness, answer_relevance, diversity):
        self.faithfulness = faithfulness
        self.answer_relevance = answer_relevance
        self.diversity = diversity


@pytest.mark.asyncio
async def test_run_benchmark_workload_aggregates_correctly(monkeypatch, tmp_path):
    async def fake_run_job(*, pdf_path, topic, config):
        return [_FakeScoredItem(1.0, 0.8, 0.3), _FakeScoredItem(0.5, 0.7, 0.4)]

    monkeypatch.setattr(benchmark_models, "run_job", fake_run_job)
    monkeypatch.setattr(benchmark_models, "collect_decode_tps_since", lambda ts: [90.0, 100.0])
    monkeypatch.setattr(benchmark_models, "count_thinking_tag_calls_since", lambda ts: 0)

    from quiz_ingest.config import PipelineConfig

    config = PipelineConfig(backend="instance", vllm_base_url="http://fake:8000", num_questions=1)
    result = await benchmark_models.run_benchmark_workload(
        pdf_path="fake.pdf", topics=["t1"], config=config,
        n_jobs=3, num_questions=2, top_k=8,
    )

    assert result["jobs_run"] == 3
    assert result["job_failures"] == 0
    assert result["delivered_total"] == 6  # 3 jobs * 2 items each
    assert result["requested_total"] == 6  # 3 jobs * num_questions=2
    assert result["delivery_rate"] == 1.0
    assert result["mean_faithfulness"] == pytest.approx(0.75)
    assert result["mean_decode_tokens_per_second"] == pytest.approx(95.0)


@pytest.mark.asyncio
async def test_run_benchmark_workload_survives_job_failures(monkeypatch):
    call_count = {"n": 0}

    async def flaky_run_job(*, pdf_path, topic, config):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated failure")
        return [_FakeScoredItem(1.0, 1.0, 1.0)]

    monkeypatch.setattr(benchmark_models, "run_job", flaky_run_job)
    monkeypatch.setattr(benchmark_models, "collect_decode_tps_since", lambda ts: [])
    monkeypatch.setattr(benchmark_models, "count_thinking_tag_calls_since", lambda ts: 0)

    from quiz_ingest.config import PipelineConfig

    config = PipelineConfig(backend="instance", vllm_base_url="http://fake:8000", num_questions=1)
    result = await benchmark_models.run_benchmark_workload(
        pdf_path="fake.pdf", topics=["t1"], config=config,
        n_jobs=3, num_questions=1, top_k=8,
    )

    assert result["jobs_run"] == 3
    assert result["job_failures"] == 1
    assert result["delivered_total"] == 2  # 2 successful jobs * 1 item each
    assert result["mean_decode_tokens_per_second"] is None  # no samples -- must not fabricate 0.0


def test_count_thinking_tag_calls_since_counts_only_flagged_calls(tmp_path, monkeypatch):
    import json as json_module

    fake_log_dir = tmp_path / "logs"
    fake_log_dir.mkdir()
    fake_log_path = fake_log_dir / "quiz-ingest-events-fake.jsonl"

    new_ts = time.time()
    with open(fake_log_path, "w") as f:
        f.write(json_module.dumps({"event": "api_call", "ts": new_ts, "contains_thinking_tags": True}) + "\n")
        f.write(json_module.dumps({"event": "api_call", "ts": new_ts, "contains_thinking_tags": False}) + "\n")
        f.write(json_module.dumps({"event": "api_call", "ts": new_ts}) + "\n")  # field absent -- must not count

    monkeypatch.setattr(benchmark_models, "_log_path", lambda: fake_log_path)

    count = benchmark_models.count_thinking_tag_calls_since(new_ts - 1)
    assert count == 1


# ---------------------------------------------------------------------------
# VLLMServerProcess.wait_until_healthy -- state machine, using a real local
# HTTP server (no actual vLLM needed) so the polling mechanics are genuinely
# exercised rather than just mocked away.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wait_until_healthy_detects_dead_process_without_waiting_full_timeout():
    import subprocess as sp

    server = benchmark_models.VLLMServerProcess(
        model="unused", port=39123, gpu_memory_utilization=0.1, max_model_len=1024,
        log_path=Path("/tmp/benchmark_models_test_dead_proc.log"),
    )
    # Simulate a process that exits immediately (e.g. bad model name) --
    # must be detected fast via .poll(), not by waiting out the full
    # timeout with repeated failed health-check requests.
    server._proc = sp.Popen(["python3", "-c", "import sys; sys.exit(1)"])
    t0 = time.monotonic()
    healthy = await server.wait_until_healthy(timeout_s=30.0, poll_interval_s=1.0)
    elapsed = time.monotonic() - t0

    assert healthy is False
    assert elapsed < 10.0  # detected quickly, not after the full 30s timeout
