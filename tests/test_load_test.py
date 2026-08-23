import asyncio
import importlib.util
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

_spec = importlib.util.spec_from_file_location("load_test", REPO_ROOT / "scripts" / "load_test.py")
load_test = importlib.util.module_from_spec(_spec)
sys.modules["load_test"] = load_test  # dataclass() needs the module registered to resolve string annotations
_spec.loader.exec_module(load_test)


@pytest.mark.asyncio
async def test_run_level_keeps_exactly_concurrency_in_flight_at_steady_state(monkeypatch):
    current_in_flight = 0
    in_flight_samples = []
    lock = asyncio.Lock()

    async def fake_run_job(*, pdf_path, topic, config):
        nonlocal current_in_flight
        async with lock:
            current_in_flight += 1
            in_flight_samples.append(current_in_flight)
        await asyncio.sleep(0.03)  # simulate real work, long enough for overlap to show up
        async with lock:
            current_in_flight -= 1
        return [object()] * config.num_questions  # len() used as "delivered"

    monkeypatch.setattr(load_test, "run_job", fake_run_job)

    from quiz_ingest.config import PipelineConfig

    base_config = PipelineConfig(backend="instance", vllm_base_url="http://fake:8000", num_questions=1)

    results = await load_test.run_level(
        concurrency=3, requests_per_worker=4, pdf_path="fake.pdf",
        topics=["t1", "t2"], base_config=base_config, top_k_range=(4, 8),
    )

    # total requests = concurrency * requests_per_worker, not an
    # independent fixed pool -- this is the methodology fix.
    assert len(results) == 3 * 4
    assert all(r.ok for r in results)
    # never exceeded the cap, AND reached it (steady state achieved, not
    # accidentally serialized down to 1 by a bug).
    assert max(in_flight_samples) == 3


@pytest.mark.asyncio
async def test_run_level_survives_individual_job_failures(monkeypatch):
    call_count = 0

    async def flaky_run_job(*, pdf_path, topic, config):
        nonlocal call_count
        call_count += 1
        if call_count % 3 == 0:
            raise RuntimeError("simulated vLLM error")
        return [object()] * config.num_questions

    monkeypatch.setattr(load_test, "run_job", flaky_run_job)

    from quiz_ingest.config import PipelineConfig

    base_config = PipelineConfig(backend="instance", vllm_base_url="http://fake:8000", num_questions=1)
    results = await load_test.run_level(
        concurrency=2, requests_per_worker=5, pdf_path="fake.pdf",
        topics=["t1"], base_config=base_config, top_k_range=(4, 8),
    )

    assert len(results) == 10  # 2 workers * 5 requests each
    assert sum(1 for r in results if not r.ok) > 0
    assert all(r.error is not None for r in results if not r.ok)


@pytest.mark.asyncio
async def test_run_level_total_requests_scales_with_concurrency(monkeypatch):
    async def fake_run_job(*, pdf_path, topic, config):
        return [object()] * config.num_questions

    monkeypatch.setattr(load_test, "run_job", fake_run_job)

    from quiz_ingest.config import PipelineConfig

    base_config = PipelineConfig(backend="instance", vllm_base_url="http://fake:8000", num_questions=1)

    results_c1 = await load_test.run_level(
        concurrency=1, requests_per_worker=5, pdf_path="fake.pdf",
        topics=["t1"], base_config=base_config, top_k_range=(4, 8),
    )
    results_c8 = await load_test.run_level(
        concurrency=8, requests_per_worker=5, pdf_path="fake.pdf",
        topics=["t1"], base_config=base_config, top_k_range=(4, 8),
    )
    # Fixed requests_per_worker, scaling concurrency scales total requests
    # -- this is exactly what an earlier, fixed-pool version did NOT do,
    # which caused the queue-wait measurement artifact this rewrite fixes.
    assert len(results_c1) == 5
    assert len(results_c8) == 40


def test_summarize_level_computes_percentiles_and_delivery_rate():
    JobResult = load_test.JobResult
    results = [
        JobResult(ok=True, wall_s=1.0, requested=5, top_k=8, delivered=5),
        JobResult(ok=True, wall_s=2.0, requested=5, top_k=8, delivered=5),
        JobResult(ok=True, wall_s=3.0, requested=5, top_k=8, delivered=4),
        JobResult(ok=False, wall_s=0.5, requested=5, top_k=8, error="boom"),
    ]
    summary = load_test.summarize_level(results, decode_tps_samples=[90.0, 100.0])

    assert summary["total_requests"] == 4
    assert summary["success_count"] == 3
    assert summary["success_rate"] == 0.75
    assert summary["latency_p50_s"] == 2.0
    assert summary["mean_decode_tokens_per_second"] == 95.0
    assert summary["delivered_total"] == 14
    assert summary["requested_total"] == 20
    assert summary["delivery_rate"] == 0.7


def test_summarize_level_delivery_rate_can_be_zero_despite_perfect_success_rate():
    # Regression test for the exact real-world scenario this rewrite was
    # built to surface: every job "succeeds" (no exception) but delivers
    # nothing it was asked for.
    JobResult = load_test.JobResult
    results = [
        JobResult(ok=True, wall_s=10.0, requested=6, top_k=8, delivered=0),
        JobResult(ok=True, wall_s=12.0, requested=3, top_k=8, delivered=0),
    ]
    summary = load_test.summarize_level(results, decode_tps_samples=[])
    assert summary["success_rate"] == 1.0
    assert summary["delivery_rate"] == 0.0  # must NOT be hidden by a perfect success_rate


def test_summarize_level_handles_all_failures_without_crashing():
    JobResult = load_test.JobResult
    results = [JobResult(ok=False, wall_s=0.1, requested=5, top_k=8, error="err1")]
    summary = load_test.summarize_level(results, decode_tps_samples=[])
    assert summary["success_rate"] == 0.0
    assert summary["delivery_rate"] == 0.0
    assert summary["latency_p50_s"] is None
    assert summary["mean_decode_tokens_per_second"] is None
    assert "sample_errors" in summary


def test_collect_decode_tps_since_filters_by_timestamp(tmp_path, monkeypatch):
    import json

    fake_log_dir = tmp_path / "logs"
    fake_log_dir.mkdir()
    fake_log_path = fake_log_dir / "quiz-ingest-events-fake.jsonl"

    old_ts = time.time() - 1000
    new_ts = time.time()
    with open(fake_log_path, "w") as f:
        f.write(json.dumps({"event": "api_call", "ts": old_ts, "decode_tokens_per_second": 50.0}) + "\n")
        f.write(json.dumps({"event": "api_call", "ts": new_ts, "decode_tokens_per_second": 100.0}) + "\n")
        f.write(json.dumps({"event": "job_summary", "ts": new_ts}) + "\n")  # no decode field -- must be skipped

    monkeypatch.setattr(load_test, "_log_path", lambda: fake_log_path)

    values = load_test.collect_decode_tps_since(new_ts - 1)
    assert values == [100.0]
