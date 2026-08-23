import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QUIZ_BACKEND", "instance")
os.environ.setdefault("VLLM_BASE_URL", "http://fake-instance:8000")

from fastapi.testclient import TestClient

from quiz_ingest import api as api_module

SAMPLE_OUTPUT = {
    "source_mode": "pdf",
    "source": "x.pdf",
    "items": [
        {
            "question": "What is X?",
            "options": [{"text": "A", "is_correct": True, "type": "correct"}],
            "num_correct": 1,
            "supporting_fact": "fact",
            "faithfulness": 1.0,
            "answer_relevance": 0.8,
            "diversity": 0.3,
            "source_chunk_ids": [0],
        }
    ],
    "usage": {"total_calls": 1},
    "latency": {"wall_time_s": 1.2},
}


def _make_fake_run_job_streaming(output=SAMPLE_OUTPUT):
    async def fake(*, pdf_path, topic, config, output_json_path=None):
        if output_json_path is not None:
            Path(output_json_path).write_text(json.dumps(output))
        return
        yield  # pragma: no cover -- makes this an async generator that yields nothing

    return fake


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_module, "run_job_streaming", _make_fake_run_job_streaming())
    return TestClient(api_module.app)


def _pdf_bytes() -> bytes:
    return b"%PDF-1.4 fake pdf content for testing"


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_generate_quiz_returns_the_consolidated_output_json(client):
    resp = client.post(
        "/generate-quiz",
        files={"pdf": ("x.pdf", _pdf_bytes(), "application/pdf")},
        data={"topic": "test topic", "num_questions": "3"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"][0]["question"] == "What is X?"
    assert body["usage"]["total_calls"] == 1


def test_generate_quiz_rejects_num_questions_over_the_cap(client):
    resp = client.post(
        "/generate-quiz",
        files={"pdf": ("x.pdf", _pdf_bytes(), "application/pdf")},
        data={"topic": "t", "num_questions": "999"},
    )
    assert resp.status_code == 422


def test_generate_quiz_rejects_topic_over_max_length(client):
    from quiz_ingest.config import MAX_TOPIC_CHARS

    resp = client.post(
        "/generate-quiz",
        files={"pdf": ("x.pdf", _pdf_bytes(), "application/pdf")},
        data={"topic": "x" * (MAX_TOPIC_CHARS + 1), "num_questions": "3"},
    )
    assert resp.status_code == 422


def test_generate_quiz_502s_when_no_questions_survive(monkeypatch):
    # run_job_streaming only writes output_json_path on status="completed"
    # (see pipeline.py) -- simulate that by never writing the file.
    async def fake_no_output(*, pdf_path, topic, config, output_json_path=None):
        return
        yield  # pragma: no cover

    monkeypatch.setattr(api_module, "run_job_streaming", fake_no_output)
    client = TestClient(api_module.app)
    resp = client.post(
        "/generate-quiz",
        files={"pdf": ("x.pdf", _pdf_bytes(), "application/pdf")},
        data={"topic": "t", "num_questions": "3"},
    )
    assert resp.status_code == 502


def test_api_key_required_when_configured(monkeypatch):
    monkeypatch.setenv("DEMO_API_KEY", "secret123")
    monkeypatch.setattr(api_module, "run_job_streaming", _make_fake_run_job_streaming())
    client = TestClient(api_module.app)

    resp = client.post(
        "/generate-quiz",
        files={"pdf": ("x.pdf", _pdf_bytes(), "application/pdf")},
        data={"topic": "t", "num_questions": "3"},
    )
    assert resp.status_code == 401

    resp = client.post(
        "/generate-quiz",
        files={"pdf": ("x.pdf", _pdf_bytes(), "application/pdf")},
        data={"topic": "t", "num_questions": "3"},
        headers={"X-API-Key": "secret123"},
    )
    assert resp.status_code == 200


def test_api_key_not_required_when_unset(client):
    # DEMO_API_KEY not set in this fixture's environment -- endpoint stays open.
    resp = client.post(
        "/generate-quiz",
        files={"pdf": ("x.pdf", _pdf_bytes(), "application/pdf")},
        data={"topic": "t", "num_questions": "3"},
    )
    assert resp.status_code == 200


def test_swagger_docs_page_is_served(client):
    resp = client.get("/docs")
    assert resp.status_code == 200
    assert "swagger" in resp.text.lower()
