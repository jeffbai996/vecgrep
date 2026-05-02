"""BM25 store regression tests.

Past bugs covered:
- by_source map drifting after delete (chunk_count miscounts on reindex)
- pickle round-trip losing entries
- tokenizer not splitting underscore/CamelCase identifiers
"""
from __future__ import annotations

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
