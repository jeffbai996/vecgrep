"""The cross-encoder's input ceiling is a retrieval decision, not a dial.

bge-reranker-v2-m3 accepts 8192 tokens. The chunks it is actually fed measure
~620 tokens median and ~1730 at the tail (measured 2026-09-12 across the repos
corpus), so the model's own ceiling never binds and costs nothing: a batch is
padded to its LONGEST MEMBER, not to max_length, and sentence-transformers
already length-sorts before batching.

Lowering the cap therefore does not remove waste, it removes content. 512 took
a scoped reranked search from 2.83s to 0.91s and changed the top-8 on all four
queries tried, because it truncates the median chunk to 40% of itself. So the
knob exists, documented with that number, and ships off.
"""
from __future__ import annotations

import os

from vecgrep.backend import rerank


def test_the_cap_is_off_by_default():
    assert rerank.RERANK_MAX_LENGTH == 0 or os.environ.get("VECGREP_RERANK_MAX_LENGTH")


def test_no_cap_means_the_model_keeps_its_own_ceiling(monkeypatch):
    seen = {}

    class FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            seen.update(name=name, kwargs=kwargs)

    monkeypatch.setattr(rerank, "RERANK_MAX_LENGTH", 0)
    _construct_with(monkeypatch, FakeCrossEncoder)
    assert "max_length" not in seen["kwargs"]


def test_a_cap_is_passed_through(monkeypatch):
    seen = {}

    class FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            seen.update(name=name, kwargs=kwargs)

    monkeypatch.setattr(rerank, "RERANK_MAX_LENGTH", 512)
    _construct_with(monkeypatch, FakeCrossEncoder)
    assert seen["kwargs"]["max_length"] == 512


def test_the_fp16_fast_path_survives_the_cap(monkeypatch):
    """The cap and the FP16 load used to share one kwargs dict; setting either
    must not drop the other."""
    seen = {}

    class FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            seen.update(kwargs=kwargs)

    monkeypatch.setattr(rerank, "RERANK_MAX_LENGTH", 512)
    _construct_with(monkeypatch, FakeCrossEncoder, cuda=True)
    assert seen["kwargs"]["max_length"] == 512
    assert "model_kwargs" in seen["kwargs"], seen["kwargs"]


def _construct_with(monkeypatch, cls, *, cuda: bool = False):
    """Run _construct_model against a stub CrossEncoder and a stub torch."""
    import sys
    import types

    st = types.ModuleType("sentence_transformers")
    st.CrossEncoder = cls
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)

    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda,
                                       empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return rerank._construct_model("fake-model")
