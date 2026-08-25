import json

import httpx
import pytest

from quiz_ingest.llm.vllm_client import (
    MAX_PLAUSIBLE_DECODE_TOKENS_PER_SECOND,
    VLLMClient,
    VLLMClientConfig,
)


def _sse_response(chunks: list[dict]) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _make_client(handler) -> VLLMClient:
    transport = httpx.MockTransport(handler)
    client = VLLMClient(VLLMClientConfig(base_url="http://fake", model="fake-model"))
    client._client = httpx.AsyncClient(base_url="http://fake", transport=transport)
    return client


@pytest.mark.asyncio
async def test_normal_streaming_produces_plausible_decode_tps(monkeypatch):
    """
    Sanity baseline: a normal-looking stream (tokens spaced realistically
    apart) should produce a plausible decode_tps, not get discarded.
    """
    import quiz_ingest.llm.vllm_client as vllm_client_module

    t = [0.0]

    def fake_monotonic():
        t[0] += 0.01  # 10ms between each timestamp read
        return t[0]

    monkeypatch.setattr(vllm_client_module.time, "monotonic", fake_monotonic)

    def handler(request: httpx.Request) -> httpx.Response:
        chunks = [{"choices": [{"text": f"tok{i} "}]} for i in range(10)]
        chunks.append({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 10}})
        return httpx.Response(200, content=_sse_response(chunks))

    client = _make_client(handler)
    _text, telemetry = await client._stream_completion(prompt="hi", max_tokens=50)

    assert telemetry.decode_tokens_per_second is not None
    assert not telemetry.decode_tps_discarded_as_implausible
    assert telemetry.decode_tokens_per_second <= MAX_PLAUSIBLE_DECODE_TOKENS_PER_SECOND


@pytest.mark.asyncio
async def test_burst_delivered_stream_discards_implausible_decode_tps(monkeypatch):
    """
    Regression test for a real incident: under concurrent client load, an
    asyncio task can be starved of scheduling time while SSE chunks sit
    in the OS socket buffer, then drain a burst almost instantly once
    scheduled -- t_last-t_first collapses toward zero while
    completion_tokens stays normal, producing decode_tps in the tens of
    thousands (54,642 tok/s logged in one real run, on hardware whose
    real baseline is ~90-100 tok/s). Simulates that here by making EVERY
    monotonic() call after the first return (almost) the same value,
    mimicking a burst of chunks processed with no real elapsed time
    between them, and asserts the resulting bogus rate is DISCARDED
    (None, decode_tps_discarded_as_implausible=True) rather than reported.
    """
    import quiz_ingest.llm.vllm_client as vllm_client_module

    calls = {"n": 0}

    def fake_monotonic():
        calls["n"] += 1
        if calls["n"] == 1:
            return 0.0  # t_sent
        if calls["n"] == 2:
            return 5.0  # first token arrives after a real 5s scheduling delay
        return 5.0001  # every subsequent chunk "arrives" in a near-zero-time burst

    monkeypatch.setattr(vllm_client_module.time, "monotonic", fake_monotonic)

    def handler(request: httpx.Request) -> httpx.Response:
        chunks = [{"choices": [{"text": f"tok{i} "}]} for i in range(100)]
        chunks.append({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 100}})
        return httpx.Response(200, content=_sse_response(chunks))

    client = _make_client(handler)
    _text, telemetry = await client._stream_completion(prompt="hi", max_tokens=200)

    assert telemetry.decode_tokens_per_second is None
    assert telemetry.decode_tps_discarded_as_implausible is True
    # completion_tokens/ttft must stay real and unaffected -- only the
    # unreliable decode_tps measurement itself is discarded.
    assert telemetry.completion_tokens == 100
