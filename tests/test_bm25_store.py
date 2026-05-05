"""BM25 store regression tests.

Past bugs covered:
- by_source map drifting after delete (chunk_count miscounts on reindex)
- pickle round-trip losing entries
- tokenizer not splitting underscore/CamelCase identifiers
- short-query partial-token matches dominating fused ranking
"""
from __future__ import annotations

import pytest

from vecgrep.backend.store.bm25_store import BM25Store, tokenize


def test_tokenize_splits_identifiers():
    assert tokenize("sharpe_ratio") == ["sharpe", "ratio"]
    assert tokenize("getUserName") == ["get", "user", "name"]
    assert tokenize("HTTPSRequest") == ["https", "request"]
    assert tokenize("plain English") == ["plain", "english"]
    assert tokenize("CPU2024") == ["cpu", "2024"]


def test_upsert_and_search(tmp_path):
    store = BM25Store(tmp_path)
    store.upsert(
        "demo",
        ids=["a", "b", "c"],
        texts=[
            "Federal Reserve holds rates",
            "CPI inflation cools",
            "Bonds rally on dovish surprise",
        ],
        payloads=[{"source_id": "/x", "text": t, "corpus": "demo"} for t in [
            "Federal Reserve holds rates",
            "CPI inflation cools",
            "Bonds rally on dovish surprise",
        ]],
    )
    hits = store.search("demo", "CPI inflation", top_k=3)
    assert hits, "BM25 should match exact tokens"
    assert hits[0][0] == "b"  # the CPI chunk wins


def test_delete_by_source_keeps_other_sources(tmp_path):
    store = BM25Store(tmp_path)
    store.upsert(
        "demo",
        ids=["1", "2"],
        texts=["alpha beta gamma", "delta epsilon zeta"],
        payloads=[{"source_id": "/x"}, {"source_id": "/y"}],
    )
    store.delete_by_source("demo", "/x")

    hits = store.search("demo", "alpha", top_k=3)
    assert not hits, "deleted source should not surface"
    hits = store.search("demo", "delta", top_k=3)
    assert hits, "untouched source should still match"
    assert hits[0][0] == "2"


def test_persistence_round_trip(tmp_path):
    store = BM25Store(tmp_path)
    store.upsert(
        "demo",
        ids=["1"],
        texts=["persist this for next time"],
        payloads=[{"source_id": "/x"}],
    )
    # New store instance, same path — must reload from pickle.
    fresh = BM25Store(tmp_path)
    hits = fresh.search("demo", "persist", top_k=3)
    assert hits
    assert hits[0][0] == "1"


def test_drop_removes_pickle(tmp_path):
    store = BM25Store(tmp_path)
    store.upsert(
        "demo",
        ids=["1"],
        texts=["something"],
        payloads=[{"source_id": "/x"}],
    )
    pkl = tmp_path / "demo.pkl"
    assert pkl.exists()
    store.drop("demo")
    assert not pkl.exists()


def test_short_query_requires_full_coverage():
    """2-token query against partial-match doc should not return that doc.

    Reproduces the 'glucose monitoring' floods-the-corpus bug: doc with
    only 'monitoring' (and no 'glucose') used to score on its IDF alone
    and beat genuine vector hits in the fused ranking.
    """
    store = BM25Store(None)
    store.upsert(
        "t",
        ids=["a", "b"],
        texts=[
            "ibkr terminal monitoring server",  # only "monitoring"
            "glucose monitoring CGM continuous",  # both tokens
        ],
        payloads=[{"source_id": "1"}, {"source_id": "2"}],
    )
    hits = store.search("t", "glucose monitoring", top_k=5)
    ids = [h[0] for h in hits]
    assert ids == ["b"], f"only the full-match doc should survive, got {ids}"


def test_disable_coverage_filter_via_env(monkeypatch: pytest.MonkeyPatch):
    """Safety hatch: VECGREP_BM25_DISABLE_COVERAGE_FILTER=1 restores old behavior."""
    monkeypatch.setenv("VECGREP_BM25_DISABLE_COVERAGE_FILTER", "1")
    store = BM25Store(None)
    store.upsert(
        "t",
        ids=["a", "b"],
        texts=[
            "ibkr terminal monitoring server",
            "glucose monitoring CGM continuous",
        ],
        payloads=[{"source_id": "1"}, {"source_id": "2"}],
    )
    hits = store.search("t", "glucose monitoring", top_k=5)
    ids = sorted(h[0] for h in hits)
    assert ids == ["a", "b"], f"both docs should return when filter is off, got {ids}"


def test_long_query_partial_match_meets_threshold():
    """5-token query: 50% threshold = need 3 distinct tokens. Doc with 3/5 wins."""
    store = BM25Store(None)
    store.upsert(
        "t",
        ids=["a", "b"],
        texts=[
            # Has alpha, beta, gamma -> 3/5 of query, meets 50%
            "alpha beta gamma corpus filler text here",
            # Only "alpha" -> 1/5, below threshold
            "alpha unrelated words filler content",
        ],
        payloads=[{"source_id": "1"}, {"source_id": "2"}],
    )
    hits = store.search("t", "alpha beta gamma delta epsilon", top_k=5)
    ids = [h[0] for h in hits]
    assert "a" in ids, f"doc covering 3/5 tokens should be returned, got {ids}"
    assert "b" not in ids, f"doc covering 1/5 tokens should be filtered, got {ids}"


def test_fallback_path_respects_coverage():
    """Fallback path (all BM25 scores zero) must also enforce coverage.

    Forced by inserting only one doc — single-doc corpus zeroes all IDF,
    so BM25Okapi.get_scores returns zeros and the token-overlap fallback
    fires. Without gating, the fallback would re-introduce the bug.
    """
    store = BM25Store(None)
    store.upsert(
        "t",
        ids=["only"],
        texts=["ibkr terminal monitoring server"],  # only "monitoring" overlaps
        payloads=[{"source_id": "1"}],
    )
    hits = store.search("t", "glucose monitoring", top_k=5)
    assert hits == [], f"partial-coverage doc must not surface via fallback, got {hits}"


def test_by_source_map_intact_after_partial_delete(tmp_path):
    """Regression: by_source map referenced freed array indices, breaking
    chunk-count math on subsequent reindexes.
    """
    store = BM25Store(tmp_path)
    store.upsert(
        "demo",
        ids=["1", "2", "3"],
        texts=["a", "b", "c"],
        payloads=[
            {"source_id": "/x"},
            {"source_id": "/x"},
            {"source_id": "/y"},
        ],
    )
    store.delete_by_source("demo", "/x")
    idx = store._load("demo")
    # /y still points to a valid array position
    positions = idx.by_source["/y"]
    assert positions
    for pos in positions:
        assert idx.payloads[pos]["source_id"] == "/y"
