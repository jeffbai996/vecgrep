"""The SQLite backend must rank like the pickle backend.

Storage can change; retrieval must not. These build both stores from identical
records and compare what comes back — order first, because the ranking is what
callers consume, then the payloads and the shape of the result tuple.

Absolute scores are NOT compared. FTS5's bm25() and rank_bm25's BM25Okapi are
different implementations of the same family and will not agree to the decimal.
What has to hold is the ORDER, since everything above this fuses by rank.
"""
from __future__ import annotations

import pytest

from vecgrep.backend.store.bm25_store import BM25Store
from vecgrep.backend.store.bm25_sqlite import BM25SqliteStore

DOCS = [
    ("a1", "the quick brown fox jumps over the lazy dog", "/animals"),
    ("a2", "a quick brown hare outruns the fox", "/animals"),
    ("a3", "lazy afternoon sunlight on the dog bed", "/animals"),
    ("b1", "memory pressure on the host forced a reclaim", "/ops"),
    ("b2", "the ingest ceiling was raised after five timeouts", "/ops"),
    ("b3", "swap exhaustion and page cache reclaim under pressure", "/ops"),
    ("c1", "getUserName and sharpe_ratio are identifiers", "/code"),
    ("c2", "the bm25 index is pickled next to the qdrant store", "/code"),
]


def _payload(source_id: str, i: int) -> dict:
    return {"source_id": source_id, "chunk_index": i, "metadata": {"n": i}}


def _both(tmp_path):
    pickled = BM25Store(tmp_path / "pkl")
    lite = BM25SqliteStore(tmp_path / "db")
    ids = [d[0] for d in DOCS]
    texts = [d[1] for d in DOCS]
    payloads = [_payload(d[2], i) for i, d in enumerate(DOCS)]
    pickled.upsert("c", ids, texts, payloads)
    lite.upsert("c", ids, texts, payloads)
    return pickled, lite


QUERIES = [
    "quick brown fox",
    "lazy dog",
    "memory pressure",
    "reclaim",
    "ingest ceiling timeouts",
    "getUserName",
    "sharpe_ratio",
    "bm25 qdrant",
    "nothing matches this phrase at all",
]
# "the" is deliberately NOT in this list. It appears in every document, so no
# ranking is defined: FTS5 scores the whole corpus at exactly 0 and the pickle
# store emits small distinct values off its IDF floor. Truncating a six-way tie
# to top_k=5 then yields different SETS, which says nothing about retrieval
# quality. That behaviour is pinned separately below.


def _tie_groups(hits) -> list[set[str]]:
    """The ranking as a sequence of equal-scoring groups.

    Two BM25 implementations agree on which documents beat which, but not on
    how to order documents they score identically — and they need not, because
    everything above this fuses by rank. Comparing raw lists would fail on
    ties that carry no information: "reclaim" hits two documents with the same
    score in BOTH backends, and each returns them in its own insertion order.
    """
    groups: list[set[str]] = []
    last = None
    for cid, score, _payload in hits:
        key = round(float(score), 6)
        if key != last:
            groups.append(set())
            last = key
        groups[-1].add(cid)
    return groups


@pytest.mark.parametrize("query", QUERIES)
def test_the_two_backends_return_the_same_ranking(tmp_path, query):
    pickled, lite = _both(tmp_path)
    a, b = pickled.search("c", query, top_k=5), lite.search("c", query, top_k=5)
    assert {c for c, _s, _p in a} == {c for c, _s, _p in b}, (
        f"{query!r}: different documents, not just a different order")
    assert _tie_groups(a) == _tie_groups(b), (
        f"{query!r}: pickle={[ (c, round(s,4)) for c,s,_ in a ]} "
        f"sqlite={[ (c, round(s,4)) for c,s,_ in b ]}")


def test_a_term_in_every_document_ranks_by_nothing(tmp_path):
    """FTS5 scores a term present in every document at exactly 0; BM25Okapi's
    IDF has a floor and emits a small positive value instead. Both are
    defensible and neither carries information — the term discriminates
    nothing — so this pins the behaviour rather than pretending they match."""
    _pickled, lite = _both(tmp_path)
    hits = lite.search("c", "the", top_k=8)
    assert len(hits) == 6
    assert len({round(s, 6) for _c, s, _p in hits}) == 1


@pytest.mark.parametrize("query", QUERIES)
def test_scores_are_higher_is_better_in_both(tmp_path, query):
    _pickled, lite = _both(tmp_path)
    scores = [s for _cid, s, _p in lite.search("c", query, top_k=5)]
    assert scores == sorted(scores, reverse=True), scores
    assert all(s > 0 for s in scores), scores


def test_payloads_round_trip_intact(tmp_path):
    pickled, lite = _both(tmp_path)
    a = pickled.search("c", "memory pressure", top_k=3)
    b = lite.search("c", "memory pressure", top_k=3)
    assert [p for _c, _s, p in a] == [p for _c, _s, p in b]


def test_delete_by_source_matches(tmp_path):
    pickled, lite = _both(tmp_path)
    pickled.delete_by_source("c", "/ops")
    lite.delete_by_source("c", "/ops")
    assert pickled.count("c") == lite.count("c")
    a = [cid for cid, _s, _p in pickled.search("c", "memory pressure reclaim", top_k=5)]
    b = [cid for cid, _s, _p in lite.search("c", "memory pressure reclaim", top_k=5)]
    assert a == b == []


def test_get_by_id_matches(tmp_path):
    pickled, lite = _both(tmp_path)
    assert pickled.get_by_id("c", "b2") == lite.get_by_id("c", "b2")
    assert pickled.get_by_id("c", "nope") is lite.get_by_id("c", "nope") is None


def test_replace_matches(tmp_path):
    pickled, lite = _both(tmp_path)
    records = [("z1", "entirely new content about turbines", {"source_id": "/new"})]
    assert pickled.replace("c", list(records)) == lite.replace("c", list(records)) == 1
    assert pickled.count("c") == lite.count("c") == 1
    a = [cid for cid, _s, _p in pickled.search("c", "turbines", top_k=3)]
    b = [cid for cid, _s, _p in lite.search("c", "turbines", top_k=3)]
    assert a == b == ["z1"]


def test_the_corpus_queries_agree_across_backends(tmp_path):
    """These are what service.py now calls instead of reaching into
    `_load(corpus).by_source` / `.payloads`, which is what actually kept a
    corpus resident. Both backends must answer identically, or migrating the
    call sites changed behaviour."""
    pickled, lite = _both(tmp_path)

    assert list(pickled.iter_payloads("c")) == list(lite.iter_payloads("c"))
    assert list(pickled.iter_sources("c")) == list(lite.iter_sources("c"))
    assert [(sid, n) for sid, _p, n in lite.iter_sources("c")] == [
        ("/animals", 3), ("/ops", 3), ("/code", 2)]

    for sid in ("/animals", "/ops", "/code", "/missing"):
        assert pickled.first_chunk_for_source("c", sid) == lite.first_chunk_for_source("c", sid)
    assert lite.first_chunk_for_source("c", "/ops")[1]["chunk_index"] == 3
    assert dict(lite.source_counts("c")) == {"/animals": 3, "/ops": 3, "/code": 2}


def test_update_payloads_matches_across_backends(tmp_path):
    """The corpus-rename path mutates every payload in place. Both backends
    must change the same rows and persist them."""
    pickled, lite = _both(tmp_path)

    def stamp(payload: dict) -> bool:
        if payload.get("corpus") == "renamed":
            return False
        payload["corpus"] = "renamed"
        return True

    assert pickled.update_payloads("c", stamp) == lite.update_payloads("c", stamp) == 8
    # second pass changes nothing, in both
    assert pickled.update_payloads("c", stamp) == lite.update_payloads("c", stamp) == 0
    assert all(p["corpus"] == "renamed" for p in lite.iter_payloads("c"))
    assert list(pickled.iter_payloads("c")) == list(lite.iter_payloads("c"))


def test_cjk_and_camelcase_tokenisation_survives(tmp_path):
    # The custom tokeniser is fed to FTS5 as pre-split text precisely so these
    # keep working; FTS5's own tokenisers do neither.
    pickled = BM25Store(tmp_path / "pkl")
    lite = BM25SqliteStore(tmp_path / "db")
    ids, texts = ["k1", "k2"], ["getUserName parses fine", "今天天气很好"]
    payloads = [{"source_id": "/x"}, {"source_id": "/x"}]
    pickled.upsert("c", ids, texts, payloads)
    lite.upsert("c", ids, texts, payloads)
    for q in ("getUserName", "user", "今天"):
        a = [c for c, _s, _p in pickled.search("c", q, top_k=3)]
        b = [c for c, _s, _p in lite.search("c", q, top_k=3)]
        assert a == b, f"{q!r}: pickle={a} sqlite={b}"


def test_migration_from_pickle_preserves_search_and_payloads(tmp_path):
    """Switching backends without migrating is a SILENT downgrade: the SQLite
    store starts empty, lexical queries return nothing, and hybrid search
    quietly becomes vector-only with no error anywhere, because an empty index
    is a legitimate state. This pins that a migrated corpus answers the same."""
    from vecgrep.backend.store.bm25_sqlite import migrate_all, migrate_from_pickle

    root = tmp_path / "bm25"
    pickled = BM25Store(root)
    ids = [d[0] for d in DOCS]
    texts = [d[1] for d in DOCS]
    payloads = [_payload(d[2], i) for i, d in enumerate(DOCS)]
    pickled.upsert("c", ids, texts, payloads)

    assert migrate_from_pickle(root, "c") == len(DOCS)
    assert (root / "c.db").exists()

    lite = BM25SqliteStore(root)
    assert lite.count("c") == pickled.count("c")
    assert list(lite.iter_payloads("c")) == list(pickled.iter_payloads("c"))
    assert list(lite.iter_sources("c")) == list(pickled.iter_sources("c"))
    for q in ("quick brown fox", "memory pressure", "getUserName"):
        a = _tie_groups(pickled.search("c", q, top_k=5))
        b = _tie_groups(lite.search("c", q, top_k=5))
        assert a == b, q


def test_migration_refuses_to_clobber_without_force(tmp_path):
    from vecgrep.backend.store.bm25_sqlite import migrate_from_pickle

    root = tmp_path / "bm25"
    pickled = BM25Store(root)
    pickled.upsert("c", ["a"], ["alpha"], [{"source_id": "/a"}])
    migrate_from_pickle(root, "c")
    with pytest.raises(FileExistsError):
        migrate_from_pickle(root, "c")
    assert migrate_from_pickle(root, "c", force=True) == 1


def test_migrate_all_converts_every_corpus(tmp_path):
    from vecgrep.backend.store.bm25_sqlite import migrate_all

    root = tmp_path / "bm25"
    pickled = BM25Store(root)
    pickled.upsert("one", ["a"], ["alpha beta"], [{"source_id": "/a"}])
    pickled.upsert("two", ["b", "c"], ["gamma", "delta"],
                   [{"source_id": "/b"}, {"source_id": "/c"}])
    assert migrate_all(root) == {"one": 1, "two": 2}


def test_migration_of_a_missing_corpus_is_an_error_not_an_empty_index(tmp_path):
    from vecgrep.backend.store.bm25_sqlite import migrate_from_pickle

    with pytest.raises(FileNotFoundError):
        migrate_from_pickle(tmp_path, "nope")


def test_sqlite_bulk_is_one_atomic_transaction(tmp_path):
    lite = BM25SqliteStore(tmp_path / "db")
    with lite.bulk("c"):
        lite.upsert("c", ["a"], ["alpha"], [{"source_id": "/a"}])
        lite.upsert("c", ["b"], ["beta"], [{"source_id": "/b"}])
    assert lite.count("c") == 2

    with pytest.raises(RuntimeError):
        with lite.bulk("c"):
            lite.upsert("c", ["z"], ["zeta"], [{"source_id": "/z"}])
            raise RuntimeError("abort")
    assert lite.get_by_id("c", "z") is None
    assert lite.dirty_corpora() == []


def test_sqlite_connection_sets_explicit_wal_bounds(tmp_path):
    from vecgrep.backend.store import bm25_sqlite as module

    lite = BM25SqliteStore(tmp_path / "db")
    conn = lite._conn("c")

    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == (
        module.WAL_AUTOCHECKPOINT_PAGES
    )
    assert conn.execute("PRAGMA journal_size_limit").fetchone()[0] == (
        module.WAL_JOURNAL_SIZE_LIMIT_BYTES
    )


def test_large_sqlite_mutations_checkpoint_without_changing_ranking(
    tmp_path, monkeypatch
):
    lite = BM25SqliteStore(tmp_path / "db")
    checkpoints = []
    real = BM25SqliteStore._checkpoint

    def recording(self, corpus, conn):
        checkpoints.append(corpus)
        return real(self, corpus, conn)

    monkeypatch.setattr(BM25SqliteStore, "_checkpoint", recording)
    records = [
        (cid, text, _payload(source, i))
        for i, (cid, text, source) in enumerate(DOCS)
    ]
    lite.replace("c", records)
    before = [cid for cid, _score, _payload in lite.search("c", "memory pressure", 5)]

    with lite.bulk("c"):
        lite.upsert("c", ["d1"], ["memory pressure circuit breaker"], [_payload("/ops", 9)])
    before_checkpoint = lite.search("c", "memory pressure", 5)
    real(lite, "c", lite._conn("c"))
    after_checkpoint = lite.search("c", "memory pressure", 5)

    assert checkpoints == ["c", "c"]
    assert before
    assert after_checkpoint == before_checkpoint
