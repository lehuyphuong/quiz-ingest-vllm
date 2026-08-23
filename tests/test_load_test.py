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
async def test_run_level_never_exceeds_concurrency_cap(monkeypatch):
    current_in_flight = 0
    max_in_flight_seen = 0
    lock = asyncio.Lock()

    async def fake_run_job(*, pdf_path, topic, config):
        nonlocal current_in_flight, max_in_flight_seen
        async with lock:
            current_in_flight += 1
            max_in_flight_seen = max(max_in_flight_seen, current_in_flight)
        await asyncio.sleep(0.05)  # simulate real work, long enough for overlap to show up
        async with lock:
            current_in_flight -= 1
        return [object()] * config.num_questions  # len() used as "delivered"

    monkeypatch.setattr(load_test, "run_job", fake_run_job)

    from quiz_ingest.config import PipelineConfig

    base_config = PipelineConfig(backend="instance", vllm_base_url="http://fake:8000", num_questions=1)

    results = await load_test.run_level(
        concurrency=3, total_requests=12, pdf_path="fake.pdf",
        topics=["t1", "t2"], base_config=base_config, top_k_range=(4, 8),
    )

    assert len(results) == 12
    assert all(r.ok for r in results)
    assert max_in_flight_seen == 3  # never exceeded the cap, and actually reached it


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
        concurrency=2, total_requests=9, pdf_path="fake.pdf",
        topics=["t1"], base_config=base_config, top_k_range=(4, 8),
    )

    assert len(results) == 9
    assert sum(1 for r in results if not r.ok) == 3
    assert all(r.error is not None for r in results if not r.ok)


def test_summarize_level_computes_percentiles_and_success_rate():
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


def test_summarize_level_handles_all_failures_without_crashing():
    JobResult = load_test.JobResult
    results = [JobResult(ok=False, wall_s=0.1, requested=5, top_k=8, error="err1")]
    summary = load_test.summarize_level(results, decode_tps_samples=[])
    assert summary["success_rate"] == 0.0
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
