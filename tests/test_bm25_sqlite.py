from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

import pytest

from vecgrep.backend.config import Settings
from vecgrep.backend.service import CorpusError, VecgrepService
from vecgrep.backend.store.bm25_sqlite import BM25SqliteStore
from vecgrep.backend.store.bm25_store import BM25Store

# ── test_bm25_sqlite_equivalence ──

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
    ops_chunk = lite.first_chunk_for_source("c", "/ops")
    assert ops_chunk is not None
    assert ops_chunk[1]["chunk_index"] == 3
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
    from vecgrep.backend.store.bm25_sqlite import migrate_from_pickle

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

    with pytest.raises(RuntimeError), lite.bulk("c"):
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


def test_sqlite_search_filter_clauses_handles_iso_and_numeric_timestamps(tmp_path):
    lite = BM25SqliteStore(tmp_path / "db")
    lite.upsert(
        "c",
        ["old", "new", "undated"],
        ["needle alpha", "needle beta", "needle gamma"],
        [
            {"source_id": "/old", "doc_timestamp": 1704067200.0},  # 2024-01-01T00:00:00Z
            {"source_id": "/new", "doc_timestamp": 1768435200.0},  # 2026-01-15T00:00:00Z
            {"source_id": "/undated", "doc_timestamp": None},
        ],
    )
    hits_iso = lite.search("c", "needle", 5, filter_clauses=["after:2026-01-01"])
    assert [cid for cid, _s, _p in hits_iso] == ["new"]

    hits_num = lite.search("c", "needle", 5, filter_clauses=["after:1760000000"])
    assert [cid for cid, _s, _p in hits_num] == ["new"]

    hits_neg = lite.search("c", "needle", 5, filter_clauses=["-after:2026-01-01"])
    assert {cid for cid, _s, _p in hits_neg} == {"old", "undated"}

    hits_invalid = lite.search("c", "needle", 5, filter_clauses=["after:not-a-date"])
    assert hits_invalid == []

# ── test_bm25_backend_selection ──

def _close(service: VecgrepService) -> None:
    service.store.client.close()
    close_all = getattr(service.bm25, "close_all", None)
    if close_all:
        close_all()


def test_service_uses_pickle_bm25_by_default(vg_home, monkeypatch):
    monkeypatch.delenv("VECGREP_BM25_BACKEND", raising=False)
    service = VecgrepService(ephemeral=False)
    try:
        assert isinstance(service.bm25, BM25Store)
    finally:
        _close(service)


def test_service_uses_sqlite_bm25_when_requested(vg_home, monkeypatch):
    monkeypatch.setenv("VECGREP_BM25_BACKEND", "sqlite")
    service = VecgrepService(ephemeral=False)
    try:
        assert isinstance(service.bm25, BM25SqliteStore)
    finally:
        _close(service)


def test_sqlite_backend_refuses_unmigrated_pickle_indexes(vg_home, monkeypatch):
    bm25 = vg_home / "bm25"
    bm25.mkdir(parents=True)
    (bm25 / "important.pkl").write_bytes(b"not migrated")
    monkeypatch.setenv("VECGREP_BM25_BACKEND", "sqlite")

    with pytest.raises(RuntimeError, match="important.*migrate"):
        VecgrepService(ephemeral=False)


def test_sqlite_backend_refuses_a_database_older_than_its_pickle(vg_home, monkeypatch):
    bm25 = vg_home / "bm25"
    bm25.mkdir(parents=True)
    database = bm25 / "important.db"
    pickle = bm25 / "important.pkl"
    database.write_bytes(b"old database")
    pickle.write_bytes(b"new pickle")
    os.utime(database, ns=(1, 1))
    os.utime(pickle, ns=(2, 2))
    monkeypatch.setenv("VECGREP_BM25_BACKEND", "sqlite")

    with pytest.raises(RuntimeError, match="important.*migrate"):
        VecgrepService(ephemeral=False)

# ── test_bm25_sqlite_drop ──

def _seed(store: BM25SqliteStore, corpus: str, text: str) -> None:
    store.upsert(corpus, ["c1"], [text],
                 [{"source_id": f"/{corpus}", "chunk_index": 0, "metadata": {}}])


def test_dropping_a_corpus_leaves_none_of_its_bytes_behind(tmp_path):
    root = tmp_path / "bm25"
    store = BM25SqliteStore(root)
    _seed(store, "gone", "the quick brown fox jumps over the lazy dog")
    # A converted corpus keeps the pickle it was built from, plus the dirty
    # marker the pickle backend writes beside it.
    (root / "gone.pkl").write_bytes(b"legacy inverted index")
    (root / "gone.pkl.dirty").write_bytes(b"")

    store.drop("gone")

    survivors = sorted(p.name for p in root.iterdir() if p.name.startswith("gone"))
    assert survivors == [], f"drop left bytes behind: {survivors}"


def test_dropping_one_corpus_does_not_touch_its_neighbour(tmp_path):
    """The new deletion builds paths from the corpus name, so a prefix collision
    or a stray glob would take a live corpus with it."""
    root = tmp_path / "bm25"
    store = BM25SqliteStore(root)
    _seed(store, "gone", "the quick brown fox")
    _seed(store, "gone-too", "memory pressure on the host forced a reclaim")
    (root / "gone.pkl").write_bytes(b"legacy")
    (root / "gone-too.pkl").write_bytes(b"legacy")

    store.drop("gone")

    assert (root / "gone-too.db").exists()
    assert (root / "gone-too.pkl").exists(), "a name that merely starts the same"
    assert store.search("gone-too", "memory pressure", 3), "neighbour still ranks"


def test_dropping_a_corpus_that_was_never_converted_is_not_an_error(tmp_path):
    """Most corpora have no .pkl at all. missing_ok covers them."""
    root = tmp_path / "bm25"
    store = BM25SqliteStore(root)
    _seed(store, "fresh", "swap exhaustion and page cache reclaim")

    store.drop("fresh")
    store.drop("never-existed")

    assert sorted(p.name for p in root.iterdir()) == []

# ── test_bm25_runtime_integrity ──

@pytest.fixture
def sqlite_svc(vg_home, stub_backend):
    from vecgrep.backend.embed.cache import CachedBackend

    settings = Settings(home=vg_home, bm25_backend="sqlite")
    service = VecgrepService(settings=settings)
    assert service._embed_cache is not None
    service._backend_cache["auto"] = CachedBackend(stub_backend, service._embed_cache)
    yield service
    assert isinstance(service.bm25, BM25SqliteStore)
    service.bm25.close_all()
    service.store.client.close()


@pytest.mark.parametrize("empty", [False, True])
def test_missing_or_empty_sidecar_rejects_search_and_can_be_repaired(
    sqlite_svc, make_doc, empty
):
    sqlite_svc.index(str(make_doc("sample.txt", "Unique error E1234 occurred.")), "notes")
    sqlite_svc.bm25.drop("notes")
    if empty:
        sqlite_svc.bm25.count("notes")
    with pytest.raises(CorpusError, match="BM25.*rebuild"):
        sqlite_svc.search("E1234", corpus_name="notes", mode="hybrid")
    assert sqlite_svc.rebuild_bm25("notes") > 0
    assert sqlite_svc.search("E1234", corpus_name="notes", mode="hybrid")


def test_sqlite_operational_failure_is_not_an_empty_result(tmp_path):
    store = BM25SqliteStore(tmp_path)
    store.upsert("notes", ["a"], ["E1234"], [{"source_id": "sample"}])
    store._conn("notes").execute("DROP TABLE chunks_fts")
    with pytest.raises(sqlite3.OperationalError):
        store.search("notes", "E1234", 5)
    store.close_all()


def test_source_intent_retires_only_after_sqlite_is_committed(sqlite_svc, tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Unique error E1234 occurred.")
    (docs / "b.txt").write_text("A second source says E5678.")
    observer = BM25SqliteStore(sqlite_svc.settings.home / "bm25")
    finish = sqlite_svc.mutations.finish
    observed = []

    def checked_finish(corpus):
        expected = sqlite_svc.registry.get(corpus).chunk_count
        assert observer.count(corpus) == expected
        observed.append(expected)
        return finish(corpus)

    monkeypatch.setattr(sqlite_svc.mutations, "finish", checked_finish)
    try:
        sqlite_svc.index(str(docs), "notes")
        assert len(observed) == 2
    finally:
        observer.close_all()

# ── test_bm25_wal_maintenance ──

def _wal_path(root: Path, corpus: str) -> Path:
    return root / f"{corpus}.db-wal"


def _fill(lite: BM25SqliteStore, corpus: str, rows: int = 400) -> None:
    ids = [f"id{i}" for i in range(rows)]
    texts = [f"alpha beta gamma delta epsilon token{i} " * 8 for i in range(rows)]
    payloads = [{"source_id": f"/src/{i % 7}", "text": texts[i]} for i in range(rows)]
    with lite.bulk(corpus):
        lite.upsert(corpus, ids, texts, payloads)


def test_bulk_exceeding_limit_truncates_wal(tmp_path, monkeypatch):
    from vecgrep.backend.store import bm25_sqlite as module

    limit = 8192
    monkeypatch.setattr(module, "WAL_JOURNAL_SIZE_LIMIT_BYTES", limit)
    root = tmp_path / "db"
    lite = BM25SqliteStore(root)

    _fill(lite, "c")

    wal = _wal_path(root, "c")
    # With no reader pinning frames, the post-bulk checkpoint must bring the
    # WAL back under its configured limit rather than stranding it at bulk
    # size until some future writer happens to reset the log.
    assert wal.stat().st_size <= limit


def test_reader_defers_truncation_then_next_mutation_reclaims(tmp_path, monkeypatch):
    from vecgrep.backend.store import bm25_sqlite as module

    limit = 8192
    monkeypatch.setattr(module, "WAL_JOURNAL_SIZE_LIMIT_BYTES", limit)
    root = tmp_path / "db"
    lite = BM25SqliteStore(root)
    _fill(lite, "c", rows=50)  # create the database file first

    # A reader holding an open snapshot pins WAL frames; escalation must give
    # up quickly instead of blocking the mutation path behind it.
    reader = sqlite3.connect(str(root / "c.db"))
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM chunks").fetchone()
    try:
        _fill(lite, "c")
        # Deferred, not raised: the bulk completed even though the WAL could
        # not be truncated past the reader.
        assert _wal_path(root, "c").stat().st_size > limit
    finally:
        reader.rollback()
        reader.close()

    # Reader gone: the next mutation's checkpoint reclaims the file.
    with lite.bulk("c"):
        lite.upsert("c", ["late"], ["omega token"], [{"source_id": "/late"}])
    assert _wal_path(root, "c").stat().st_size <= limit


def test_small_writes_do_not_escalate(tmp_path, caplog):
    root = tmp_path / "db"
    lite = BM25SqliteStore(root)

    with caplog.at_level(logging.INFO, logger="vecgrep.backend.store.bm25_sqlite"), lite.bulk("c"):
        lite.upsert("c", ["a"], ["alpha token"], [{"source_id": "/a"}])

    # A WAL already inside its limit needs no TRUNCATE attempt.
    assert "truncate" not in caplog.text.lower()
