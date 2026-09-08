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


def _backend_with_handler(handler, *, num_batch: int | None = None) -> OllamaBackend:
    """Build an OllamaBackend whose HTTP client is driven by `handler`.

    Uses model 'bge-m3' so __init__ skips the dim-probe call (1024 is a known
    dim), letting the handler see only the embed calls under test.
    """
    b = OllamaBackend(
        base_url="http://fake", model="bge-m3", num_batch=num_batch
    )
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
    # 3, not 2: the batched fast path attempts the window once and its 500
    # sends the whole window down the per-chunk path, where the bad chunk is
    # tried twice (initial + one retry) before giving up. The isolation this
    # test exists to prove -- good chunks survive, poison degrades to a zero
    # vector -- is asserted above and unchanged.
    assert calls["n"] == 3


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
    # `input`, never the legacy `prompt`. It is a LIST because the backend
    # batches; /api/embed accepts a bare string or an array, and this test is
    # pinning the endpoint and truncate flag, not which of the two shapes.
    assert "prompt" not in seen["body"], "legacy `prompt` key must not come back"
    assert seen["body"]["input"] == ["hello"]
    assert out[0] == [0.3] * 1024


def test_num_batch_option_is_sent_on_batch_and_resilient_requests() -> None:
    seen: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        import json as _json

        payload = _json.loads(req.content)
        seen.append(payload)
        if isinstance(payload["input"], list):
            return httpx.Response(500, text="retry individually")
        return httpx.Response(200, json={"embeddings": [[0.4] * 1024]})

    b = _backend_with_handler(handler, num_batch=2048)
    out = b.embed(["hello"])

    assert out == [[0.4] * 1024]
    assert len(seen) == 2
    assert all(payload["options"] == {"num_batch": 2048} for payload in seen)


def test_oversized_chunk_is_truncated_not_zero_vectored() -> None:
    """A chunk far longer than the context window must still yield a REAL
    vector (Ollama truncates it), never the zero-vector degradation path."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.7] * 1024]})

    b = _backend_with_handler(handler)
    out = b.embed(["word " * 20000])
    assert out[0] == [0.7] * 1024
    assert out[0] != [0.0] * 1024, "must not degrade to the zero-vector fallback"


# ─────────────── batching (added 2026-08-07) ───────────────
# The backend used to issue one HTTP request PER CHUNK. Measured against a
# live bge-m3 the endpoint served 32 inputs in 0.5s (~67/s) while a corpus
# rebuild driven by the per-chunk loop moved ~5.7/s -- a ~12x gap that was
# pure round-trip overhead, not GPU. Batching closes it, but must not lose
# the per-chunk isolation the tests above pin.

def _rows(vecs: list[list[float]]) -> httpx.Response:
    return httpx.Response(200, json={"embeddings": vecs})


def test_batch_sends_one_request_for_many_chunks() -> None:
    good = [0.1] * 1024
    seen = {"n": 0, "inputs": None}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        import json as _j
        seen["inputs"] = _j.loads(req.content)["input"]
        return _rows([good] * len(seen["inputs"]))

    b = _backend_with_handler(handler)
    out = b.embed([f"chunk-{i}" for i in range(32)])
    assert len(out) == 32
    assert seen["n"] == 1, "32 chunks must cost one request, not 32"
    assert isinstance(seen["inputs"], list) and len(seen["inputs"]) == 32


def test_batch_response_with_wrong_row_count_never_misaligns() -> None:
    """A short/long batch answer must NOT be zipped onto the inputs.

    Silently pairing 3 returned vectors with 4 chunks would attach the wrong
    embedding to the wrong text -- corruption that no error surfaces and that
    only shows up later as nonsense search results. Fall back to per-chunk,
    where each vector is provably the one for its own input.
    """
    good = [0.2] * 1024
    calls = {"batch": 0, "single": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        import json as _j
        payload = _j.loads(req.content)["input"]
        if isinstance(payload, list):
            calls["batch"] += 1
            return _rows([good] * (len(payload) - 1))   # one row short
        calls["single"] += 1
        return _ok(good)

    b = _backend_with_handler(handler)
    out = b.embed(["a", "b", "c", "d"])
    assert len(out) == 4
    assert all(v == good for v in out)
    assert calls["single"] == 4, "a mismatched batch must re-embed per chunk"


def test_batch_failure_falls_back_and_still_isolates_a_poison_chunk() -> None:
    """Batch 500s (the NaN case, now at batch scope) -> per-chunk path runs,
    and the one bad chunk still degrades to a zero vector instead of taking
    the whole document with it."""
    good = [0.3] * 1024

    def handler(req: httpx.Request) -> httpx.Response:
        import json as _j
        payload = _j.loads(req.content)["input"]
        if isinstance(payload, list):
            return httpx.Response(500, text="unsupported value: NaN")
        if payload == "poison":
            return httpx.Response(500, text="unsupported value: NaN")
        return _ok(good)

    b = _backend_with_handler(handler)
    out = b.embed(["a", "poison", "b"])
    assert len(out) == 3
    assert out[0] == good and out[2] == good
    assert all(x == 0.0 for x in out[1]), "poison chunk degrades to zero vector"


def test_batch_connect_error_still_hard_fails() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    b = _backend_with_handler(handler)
    try:
        b.embed(["a", "b"])
    except EmbedBackendError as e:
        assert "Could not reach Ollama" in str(e)
    else:
        raise AssertionError("an unreachable backend must not be swallowed")
