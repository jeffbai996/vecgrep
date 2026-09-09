"""An unrecognized search mode raises instead of returning [].

`_search_one_locked` dispatches on `if mode in ("hybrid", "vector")` /
`("hybrid", "bm25")` and then falls through to the RRF block. An unknown mode
therefore missed both retrievers and fused two empty dicts into an empty list.

A silent `[]` is indistinguishable from "no results exist", so an agent
consumer reports it as a confident absence. The HTTP route already rejected
unknown modes; the in-process and MCP paths did not. Validation now lives at
the service entry point, which every caller funnels through.
"""
from __future__ import annotations

import pytest

from vecgrep.backend.service import (
    DEFAULT_MODE,
    SEARCH_MODES,
    _validate_search_mode,
)
from vecgrep.backend.store.corpora import CorpusError


def test_known_modes_are_the_literal_members() -> None:
    assert SEARCH_MODES == {"hybrid", "vector", "bm25"}
    assert DEFAULT_MODE in SEARCH_MODES


@pytest.mark.parametrize("mode", sorted(SEARCH_MODES))
def test_known_modes_pass(mode: str) -> None:
    _validate_search_mode(mode)  # must not raise


@pytest.mark.parametrize("mode", ["dense", "semantic", "embedding", "HYBRID", "", "  "])
def test_unknown_modes_raise(mode: str) -> None:
    with pytest.raises(CorpusError) as e:
        _validate_search_mode(mode)
    # The message must name the valid set — an agent that guessed "dense"
    # needs to learn what to guess next without reading the source.
    for known in SEARCH_MODES:
        assert known in str(e.value)


def test_search_rejects_unknown_mode_before_touching_a_corpus(svc, make_doc) -> None:
    p = make_doc("doc.md", "alpha content here. beta content here.")
    svc.index(str(p), "test")

    with pytest.raises(CorpusError):
        svc.search("alpha", "test", mode="dense")


def test_search_rejects_unknown_mode_with_no_corpora(svc) -> None:
    """Validation is at the entry point, so it fires even on an empty install
    — the old code returned [] here for a *valid* reason and [] for an invalid
    mode, which is exactly the ambiguity being removed."""
    with pytest.raises(CorpusError):
        svc.search("anything", mode="dense")


def test_valid_mode_still_searches(svc, make_doc) -> None:
    p = make_doc("doc.md", "alpha content here. beta content here.")
    svc.index(str(p), "test")
    assert svc.search("alpha", "test", mode="bm25")
