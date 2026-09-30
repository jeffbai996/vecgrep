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


def test_no_corpus_does_not_turn_budget_on_blind():
    # Breadth mode stays scoped-only: see the cross-corpus section below.
    assert S._should_budget(SVC, {}) is False


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


# ── cross-corpus fan-out ─────────────────────────────────────────────────────
# The default squad-bot call names no corpus. It used to skip rerank outright
# ("don't pay it blind"), which meant the highest-traffic call path never got
# the calibration the reranker exists for -- and the eval showed unreranked
# hybrid surfacing a confident top hit for almost every off-topic query. The
# fan-out merges every corpus into ONE candidate pool before the rerank step
# (service.search_with_diagnostics: filter -> dedup -> rerank), so a
# cross-corpus rerank is a single cross-encoder pass over the fused top-50,
# not N per-corpus passes. Sizing it on the summed searchable chunk_count is
# therefore the honest analogue of the single-corpus rule.


class _SvcSearchable:
    """list_corpora() is unfiltered (backup/eval read it); the fan-out uses
    _searchable_corpora(). The gate must size on what actually gets searched,
    so `hidden` is in the registry listing but never in the fan-out."""

    def __init__(self, searchable, hidden=()):
        self._searchable = list(searchable)
        self._hidden = list(hidden)

    def list_corpora(self):
        return self._searchable + self._hidden

    def _searchable_corpora(self):
        return self._searchable


def test_excluded_corpus_does_not_inflate_the_fanout_estimate():
    """An eval-* build copy is in the registry but out of the fan-out. Sizing
    on list_corpora() would rerank a tiny unscoped search on its strength."""
    svc = _SvcSearchable([_C("small-a", 700)], hidden=[_C("eval-big", 300_000)])
    assert S._should_rerank(svc, {}) is False


def test_no_corpus_reranks_when_searchable_corpora_are_big_enough():
    svc = _SvcSearchable([_C("small-a", 6_000), _C("small-b", 5_000)])
    # Neither corpus clears the bar alone; the fused pool they produce does.
    assert S._should_rerank(svc, {}) is True


def test_no_corpus_skips_rerank_when_searchable_corpora_are_tiny():
    svc = _SvcSearchable([_C("small-a", 700), _C("small-b", 300)])
    assert S._should_rerank(svc, {}) is False


def test_empty_corpora_list_argument_is_rejected_not_reranked():
    # `corpora: []` is an error at the service layer, not "search everything".
    svc = _SvcSearchable([_C("big-a", 103_339)])
    assert S._should_rerank(svc, {"corpora": []}) is False


def test_corpora_list_reranks_on_summed_chunk_count():
    svc = _SvcSearchable([_C("big-a", 103_339), _C("small-a", 6_000),
                          _C("small-b", 5_000)])
    assert S._should_rerank(svc, {"corpora": ["small-a", "small-b"]}) is True


def test_corpora_list_of_small_corpora_does_not_rerank():
    svc = _SvcSearchable([_C("big-a", 103_339), _C("small-a", 700),
                          _C("small-b", 300)])
    assert S._should_rerank(svc, {"corpora": ["small-a", "small-b"]}) is False


def test_explicit_false_still_beats_auto_on_cross_corpus():
    svc = _SvcSearchable([_C("big-a", 103_339)])
    assert S._should_rerank(svc, {"rerank": False}) is False


def test_explicit_true_still_beats_auto_on_tiny_cross_corpus():
    svc = _SvcSearchable([_C("small-b", 12)])
    assert S._should_rerank(svc, {"rerank": True}) is True


def test_unreadable_searchable_set_falls_back_to_off():
    class Boom:
        def _searchable_corpora(self):
            raise RuntimeError("registry down")

        def list_corpora(self):
            raise RuntimeError("registry down")

    assert S._should_rerank(Boom(), {}) is False


def test_service_without_searchable_helper_falls_back_to_list_corpora():
    # Older service objects (and the API dict mirror) expose only list_corpora.
    assert S._should_rerank(SVC, {}) is True


def test_budget_still_does_not_fire_blind_on_cross_corpus():
    """Breadth mode is deliberately NOT extended to unscoped search.

    Rerank only reorders a pool the caller already gets; budget changes the
    RESPONSE SHAPE (full hits + a `stubs` tail), so turning it on for every
    unscoped call would silently rewrite what every existing consumer reads.
    Same size rule, different blast radius -- keep it scoped-only."""
    svc = _SvcSearchable([_C("big-a", 103_339), _C("big-b", 258_955)])
    assert S._should_budget(svc, {}) is False
    assert S._should_budget(svc, {"corpora": ["big-a"]}) is False


def test_search_tool_exposes_include_superseded():
    import inspect
    src = inspect.getsource(S)
    assert "include_superseded: bool = False," in src
