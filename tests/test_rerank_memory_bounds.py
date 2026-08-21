"""Peak-VRAM bounds on the rerank path.

2026-08-20: a single reranked search on the transcript corpus took the board
from 5.0 GB to 17.5 GB and never released it. Three of those filled a 24 GB
card, and the next allocation failed with `CUDA error: unknown error` -- which
is how the card looked wedged that morning. Two independent things were wrong:
predict() ran at the library default batch size over chunks up to 4000 chars,
so the padded batch was enormous; and torch's caching allocator keeps the peak
reserved forever, so the high-water mark never came back down.

These pin both bounds. They are memory tests, not quality tests: the batch size
changes how many pairs are scored at once, never which score a pair gets.
"""
from __future__ import annotations

import importlib

import pytest

from vecgrep.backend import rerank as rr


class _FakeModel:
    """Records how predict() was called, returns one logit per pair."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict(self, pairs, **kwargs):
        self.calls.append({"n": len(pairs), **kwargs})
        return [0.5] * len(pairs)


@pytest.fixture
def fake_model(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(rr, "_load", lambda name: model)
    return model


def _pairs(n: int):
    return [(f"chunk {i}", {"id": i}) for i in range(n)]


def test_predict_gets_an_explicit_bounded_batch_size(fake_model):
    """The library default is 32. A padded batch of 32 long transcript chunks
    is what produced the 12 GB spike, so the batch size must be ours, not the
    library's, and it must be small."""
    rr.rerank("q", _pairs(50))
    assert fake_model.calls, "predict was never called"
    batch = fake_model.calls[0].get("batch_size")
    assert batch is not None, "predict() called without an explicit batch_size"
    assert batch <= 16


def test_batch_size_is_env_tunable(monkeypatch, fake_model):
    """A box with more VRAM than the 3090 should be able to raise this without
    a code change."""
    monkeypatch.setenv("VECGREP_RERANK_BATCH", "4")
    importlib.reload(rr)
    try:
        monkeypatch.setattr(rr, "_load", lambda name: fake_model)
        rr.rerank("q", _pairs(10))
        assert fake_model.calls[-1]["batch_size"] == 4
    finally:
        monkeypatch.delenv("VECGREP_RERANK_BATCH", raising=False)
        importlib.reload(rr)


def test_cuda_reserve_is_released_after_scoring(monkeypatch, fake_model):
    """torch reserves the peak and never returns it on its own. Without this
    the high-water mark of one unlucky search is charged to the card until the
    process restarts."""
    released = []
    monkeypatch.setattr(rr, "_release_cuda_cache", lambda: released.append(True))
    rr.rerank("q", _pairs(5))
    assert released == [True]


def test_cuda_reserve_is_released_even_when_scoring_raises(monkeypatch):
    """A rerank that dies mid-batch is exactly when the card is most full."""
    class _Boom:
        def predict(self, pairs, **kwargs):
            raise RuntimeError("CUDA error: unknown error")

    monkeypatch.setattr(rr, "_load", lambda name: _Boom())
    released = []
    monkeypatch.setattr(rr, "_release_cuda_cache", lambda: released.append(True))
    with pytest.raises(RuntimeError):
        rr.rerank("q", _pairs(5))
    assert released == [True]


def test_release_is_a_no_op_without_torch(monkeypatch):
    """Reranking is an optional extra. On an install without torch, or on a
    CPU-only box, releasing must not raise."""
    import builtins

    real_import = builtins.__import__

    def _no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_torch)
    rr._release_cuda_cache()  # must not raise


def test_empty_candidates_skip_the_model_entirely(fake_model):
    rr.rerank("q", [])
    assert fake_model.calls == []
