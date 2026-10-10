"""Stopwords leave the BM25 QUERY; the index is untouched.

Every query token becomes an OR clause, so "ferry to bowen island" matched
64-79k chunks on "to" alone, and FTS5 scores every match before it can return
the top few hundred. Measured 2026-10-10: 160 ms warm, 13.6 s on a cold `cli`
sidecar, and 7-8 s inside live searches. "to" carries almost no ranking
signal (its IDF is near zero), so dropping it changes cost, not meaning.

A query that is nothing BUT stopwords keeps them all ("to be or not to be"),
because an empty query would return nothing.
"""
from __future__ import annotations

from vecgrep.backend.store.bm25_sqlite import BM25SqliteStore
from vecgrep.backend.store.bm25_store import BM25Store, query_tokens


def test_stopwords_are_dropped_from_the_query():
    assert query_tokens("ferry to the island") == ["ferry", "island"]


def test_content_words_cjk_and_identifiers_survive():
    assert query_tokens("getUserName 测试 for qdrant") == ["get", "user", "name", "测试", "qdrant"]


def test_an_all_stopword_query_keeps_its_words():
    assert query_tokens("to be or not to be") == ["to", "be", "or", "not", "to", "be"]


def _store(kind, tmp_path):
    store = BM25Store(tmp_path / "pkl") if kind == "pkl" else BM25SqliteStore(tmp_path / "db")
    docs = [("a", "the ferry to the island leaves at noon"),
            ("b", "to and from the office and back to the house"),
            ("c", "ferry island timetable")]
    store.upsert("c", [d[0] for d in docs], [d[1] for d in docs],
                 [{"source_id": d[0], "chunk_index": 0} for d in docs])
    return store


def test_a_chunk_matching_only_stopwords_is_not_a_candidate(tmp_path):
    for kind in ("pkl", "sqlite"):
        hits = _store(kind, tmp_path / kind).search("c", "ferry to the island", top_k=10)
        assert {h[0] for h in hits} == {"a", "c"}, kind


def test_missing_a_stopword_costs_no_coverage(tmp_path):
    """"ferry island timetable" holds every CONTENT word of the query."""
    for kind in ("pkl", "sqlite"):
        hits = _store(kind, tmp_path / kind).search("c", "ferry to the island timetable", top_k=10)
        assert hits[0][0] == "c", kind
