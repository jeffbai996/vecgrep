"""Loading a cross-encoder must never happen inside a search.

A search runs in FastAPI's sync threadpool, capped at 8
(VECGREP_THREAD_POOL_SIZE). `CrossEncoder(name)` imports torch and, on first
use, downloads weights — minutes. Held inside a search, that occupies one of
eight threads for the duration and starves every other sync route, health
check included; on 2026-08-18 squad-watchdog saw the timeout and restarted
vecgrep-serve mid-request at strikes=1.

So: warm in the background, wait a bounded time, and fall back to fusion order
rather than block. These tests use a fake loader — no torch, no network.
"""
import threading
import time
import types

import pytest

from vecgrep.backend import rerank as R


@pytest.fixture(autouse=True)
def _clean_state():
    R._cache.clear()
    R._warming.clear()
    R._failed_at.clear()
    yield
    R._cache.clear()
    R._warming.clear()
    R._failed_at.clear()


def test_wait_ready_is_false_while_the_model_is_still_loading(monkeypatch):
    """The whole point: a cold model reports not-ready fast instead of
    blocking the caller for the length of the load."""
    release = threading.Event()

    def _slow(name):
        release.wait(10)
        R._cache[name] = object()
        return R._cache[name]

    monkeypatch.setattr(R, "_load", _slow)
    started = time.monotonic()
    assert R.wait_ready("m", timeout=0.2) is False
    assert time.monotonic() - started < 5, "wait_ready blocked on the load"
    release.set()


def test_wait_ready_is_true_once_warm_finishes(monkeypatch):
    monkeypatch.setattr(R, "_load", lambda name: R._cache.setdefault(name, object()))
    assert R.ensure_warm("m").wait(5)
    assert R.is_ready("m") is True
    assert R.wait_ready("m", timeout=0) is True


def test_warm_is_idempotent(monkeypatch):
    """Called per request, so a second call must not start a second load."""
    calls = []

    def _count(name):
        calls.append(name)
        time.sleep(0.05)
        R._cache[name] = object()
        return R._cache[name]

    monkeypatch.setattr(R, "_load", _count)
    events = [R.ensure_warm("m") for _ in range(5)]
    for ev in events:
        assert ev.wait(5)
    assert len(calls) == 1


def test_a_failed_load_does_not_retry_on_every_call(monkeypatch):
    """Retrying a failing download per search would be its own outage."""
    calls = []

    def _boom(name):
        calls.append(name)
        raise R.RerankerError("no extra installed")

    monkeypatch.setattr(R, "_load", _boom)
    for _ in range(5):
        assert R.wait_ready("m", timeout=1) is False
    assert len(calls) == 1


def test_a_failed_load_retries_after_the_cooldown(monkeypatch):
    """...but it must not stay off until someone restarts the process."""
    calls = []

    def _boom(name):
        calls.append(name)
        raise R.RerankerError("transient")

    monkeypatch.setattr(R, "_load", _boom)
    assert R.wait_ready("m", timeout=1) is False
    assert len(calls) == 1
    R._failed_at["m"] = time.monotonic() - (R._RETRY_AFTER_S + 1)
    assert R.wait_ready("m", timeout=1) is False
    assert len(calls) == 2


def test_rerank_still_loads_synchronously_for_direct_callers(monkeypatch):
    """The CLI and the eval harness want the model, however long it takes —
    only the SERVER needs the bounded path."""
    monkeypatch.setattr(R, "_load", lambda name: _FakeModel())
    out = R.rerank("q", [("a", {"id": 1}), ("b", {"id": 2})], model_name="m")
    assert [p["id"] for _, p in out] == [2, 1]


def test_cross_encoder_loads_in_float16_when_cuda_is_available(monkeypatch):
    captured = {}
    float16 = object()

    class _FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            captured["name"] = name
            captured["kwargs"] = kwargs

    fake_torch = types.SimpleNamespace(
        float16=float16,
        cuda=types.SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "sentence_transformers",
        types.SimpleNamespace(CrossEncoder=_FakeCrossEncoder),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    model = R._load("reranker")

    assert isinstance(model, _FakeCrossEncoder)
    assert captured == {
        "name": "reranker",
        "kwargs": {"model_kwargs": {"torch_dtype": float16}},
    }


def test_cross_encoder_keeps_default_dtype_without_cuda(monkeypatch):
    captured = {}

    class _FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            captured["name"] = name
            captured["kwargs"] = kwargs

    fake_torch = types.SimpleNamespace(
        float16=object(),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "sentence_transformers",
        types.SimpleNamespace(CrossEncoder=_FakeCrossEncoder),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    R._load("reranker")

    assert captured == {"name": "reranker", "kwargs": {}}


class _FakeModel:
    def predict(self, pairs, **kwargs):  # kwargs: batch_size, see rerank.RERANK_BATCH
        return [float(i) for i in range(len(pairs))]
