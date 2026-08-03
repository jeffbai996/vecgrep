"""Size-gated auto modes: rerank + breadth on large corpora only.

Corpus names here are deliberately generic — this repo is public.

Both gates share one rule — omitted means "decide by corpus size", explicit
true/false always wins. The explicit-wins half is what regressed once already:
the FastMCP wrapper declared `rerank: bool = False`, so every call passed an
explicit False and the auto path could never fire. These pin both halves.
"""
from __future__ import annotations

import pytest

pytest.importorskip("mcp")

from vecgrep.mcp import server as S


class _C:
    def __init__(self, name, chunk_count):
        self.name = name
        self.chunk_count = chunk_count


class _Svc:
    def list_corpora(self):
        return [_C("big-a", 103_339), _C("big-b", 258_955),
                _C("small-a", 2_436), _C("small-b", 714)]


SVC = _Svc()


@pytest.mark.parametrize("gate", [S._should_rerank, S._should_budget])
@pytest.mark.parametrize("corpus,want", [
    ("big-a", True), ("big-b", True),        # big -> on
    ("small-a", False), ("small-b", False),  # small -> off
])
def test_auto_by_corpus_size(gate, corpus, want):
    assert gate(SVC, {"corpus": corpus}) is want


@pytest.mark.parametrize("gate,key", [(S._should_rerank, "rerank"),
                                      (S._should_budget, "budget")])
def test_explicit_false_beats_auto_on_big_corpus(gate, key):
    # The regression that made the rerank gate a no-op: an explicit False
    # must win, which is exactly why the wrapper must default to None.
    assert gate(SVC, {"corpus": "big-a", key: False}) is False


@pytest.mark.parametrize("gate,key", [(S._should_rerank, "rerank"),
                                      (S._should_budget, "budget")])
def test_explicit_true_beats_auto_on_small_corpus(gate, key):
    assert gate(SVC, {"corpus": "small-a", key: True}) is True


@pytest.mark.parametrize("gate", [S._should_rerank, S._should_budget])
def test_no_corpus_does_not_pay_it_blind(gate):
    # Cross-corpus search can't be size-checked, so don't opt in silently.
    assert gate(SVC, {}) is False


@pytest.mark.parametrize("gate", [S._should_rerank, S._should_budget])
def test_unreadable_corpus_falls_back_to_off(gate):
    class Boom:
        def list_corpora(self):
            raise RuntimeError("registry down")
    assert gate(Boom(), {"corpus": "big-a"}) is False


def test_wrappers_default_to_none_not_false():
    """The signature contract: None means 'unspecified'. If either of these
    reverts to a bool default, the corresponding auto gate silently dies."""
    import inspect
    sig = inspect.signature(
        [t for t in [S.build_mcp_server] if t][0])  # module import smoke
    assert sig is not None
    src = inspect.getsource(S)
    assert "budget: bool | None = None," in src
    assert "rerank: bool | None = None," in src
