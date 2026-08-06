"""OllamaBackend resilience tests.

Focus: a single chunk that makes Ollama 500 (the bge-m3 "unsupported value:
NaN" failure) or return a NaN-laced vector must NOT abort the whole batch.
The backend should retry once, then fall back to a zero vector so the chunk
stays stored (and keyword-searchable via BM25) without ever winning on cosine.
"""
from __future__ import annotations

import math

import httpx

from vecgrep.backend.embed.base import EmbedBackendError
from vecgrep.backend.embed.ollama import OllamaBackend


def _backend_with_handler(handler) -> OllamaBackend:
    """Build an OllamaBackend whose HTTP client is driven by `handler`.

    Uses model 'bge-m3' so __init__ skips the dim-probe call (1024 is a known
    dim), letting the handler see only the embed calls under test.
    """
    b = OllamaBackend(base_url="http://fake", model="bge-m3")
    b._client = httpx.Client(transport=httpx.MockTransport(handler))
    return b


def _ok(vec: list[float]) -> httpx.Response:
    return httpx.Response(200, json={"embedding": vec})


def test_clean_embeddings_pass_through() -> None:
    good = [0.1] * 1024
    b = _backend_with_handler(lambda req: _ok(good))
    out = b.embed(["a", "b", "c"])
    assert len(out) == 3
    assert all(v == good for v in out)


def test_nan_500_chunk_falls_back_to_zero_vector_not_abort() -> None:
    """The exact bge-m3 failure: one chunk 500s with a NaN message."""
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        body = req.content.decode()
        if "BAD" in body:
            calls["n"] += 1
            return httpx.Response(500, text="unsupported value: NaN")
        return _ok([0.2] * 1024)

    b = _backend_with_handler(handler)
    out = b.embed(["fine one", "BAD chunk", "fine two"])

    assert len(out) == 3  # batch survived
    assert out[0] == [0.2] * 1024
    assert out[2] == [0.2] * 1024
    assert out[1] == [0.0] * 1024  # bad chunk -> zero vector
    assert calls["n"] == 2  # tried twice (initial + one retry) before giving up


def test_nan_in_vector_payload_falls_back() -> None:
    """Ollama returns 200 but the vector itself contains NaN/inf.

    Real Ollama emits NaN with a lenient JSON encoder; stdlib json.dumps
    rejects it, so we build the body string by hand to mimic the wire format.
    """
    nan_body = '{"embedding": [NaN, ' + ", ".join(["0.1"] * 1023) + "]}"

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=nan_body, headers={"content-type": "application/json"})

    b = _backend_with_handler(handler)
    out = b.embed(["BAD"])
    assert out[0] == [0.0] * 1024
    assert all(math.isfinite(x) for x in out[0])


def test_retry_recovers_transient_failure() -> None:
    """A 500 that clears on retry yields the good vector, not a fallback."""
    state = {"failed": False}

    def handler(req: httpx.Request) -> httpx.Response:
        if not state["failed"]:
            state["failed"] = True
            return httpx.Response(500, text="unsupported value: NaN")
        return _ok([0.5] * 1024)

    b = _backend_with_handler(handler)
    out = b.embed(["flaky"])
    assert out[0] == [0.5] * 1024  # recovered on retry, no zero fallback


def test_model_not_found_still_hard_fails() -> None:
    """404 (model not pulled) is a whole-backend error — must raise, not mask."""
    b = _backend_with_handler(lambda req: httpx.Response(404, text="not found"))
    try:
        b.embed(["x"])
    except EmbedBackendError as e:
        assert "not available" in str(e) or "pull" in str(e).lower()
    else:
        raise AssertionError("expected EmbedBackendError on 404")


def test_bad_request_4xx_still_hard_fails() -> None:
    """A genuine 4xx (not NaN) should raise rather than silently zero-vector."""
    b = _backend_with_handler(lambda req: httpx.Response(422, text="bad request"))
    try:
        b.embed(["x"])
    except EmbedBackendError:
        pass
    else:
        raise AssertionError("expected EmbedBackendError on 422")


def test_uses_modern_embed_endpoint_with_truncate() -> None:
    """The legacy /api/embeddings endpoint IGNORES `truncate` and 500s on any
    input longer than the model context; /api/embed honours it and truncates.

    That difference silently degraded real workloads: oversized chunks 500'd
    twice and were then stored as ZERO VECTORS, i.e.
    present for BM25 but unreachable by semantic search. Pin both the endpoint
    and the payload shape so a refactor cannot quietly regress to the legacy one.
    """
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json as _json

        seen["url"] = str(req.url)
        seen["body"] = _json.loads(req.content)
        return httpx.Response(200, json={"embeddings": [[0.3] * 1024]})

    b = _backend_with_handler(handler)
    out = b.embed(["hello"])

    assert seen["url"].endswith("/api/embed"), f"legacy endpoint used: {seen['url']}"
    assert seen["body"]["truncate"] is True, "truncate must be set or long chunks 500"
    assert seen["body"]["input"] == "hello", "modern endpoint takes `input`, not `prompt`"
    assert out[0] == [0.3] * 1024


def test_oversized_chunk_is_truncated_not_zero_vectored() -> None:
    """A chunk far longer than the context window must still yield a REAL
    vector (Ollama truncates it), never the zero-vector degradation path."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.7] * 1024]})

    b = _backend_with_handler(handler)
    out = b.embed(["word " * 20000])
    assert out[0] == [0.7] * 1024
    assert out[0] != [0.0] * 1024, "must not degrade to the zero-vector fallback"
