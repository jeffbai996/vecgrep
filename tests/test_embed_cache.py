"""Embedding cache tests.

Past bugs / risks covered:
- Cached vectors keyed wrongly across (model, text)
- Cache miss leaks the wrong text -> wrong vector
"""
from __future__ import annotations

import sqlite3

from vecgrep.backend.embed.base import EmbedBackend
from vecgrep.backend.embed.cache import DEFAULT_MAX_ROWS, CachedBackend, EmbedCache


class _Counting(EmbedBackend):
    name = "fake"
    model = "fake-1"
    dim = 4

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[float(len(t) % 7), 0.0, 0.0, 0.0] for t in texts]


def test_cache_hits_skip_inner_calls(tmp_path):
    cache = EmbedCache(tmp_path / "embed.db")
    inner = _Counting()
    wrapped = CachedBackend(inner, cache)

    wrapped.embed(["hello", "world"])
    assert inner.calls == 1

    # Same texts: should hit cache, no new inner call.
    wrapped.embed(["hello", "world"])
    assert inner.calls == 1

    # New text: only the new one goes through.
    wrapped.embed(["hello", "world", "newbie"])
    assert inner.calls == 2


def test_cache_keys_per_identity(tmp_path):
    cache = EmbedCache(tmp_path / "embed.db")

    inner_a = _Counting()
    inner_a.name = "fake"
    inner_a.model = "model-a"
    a = CachedBackend(inner_a, cache)

    inner_b = _Counting()
    inner_b.name = "fake"
    inner_b.model = "model-b"
    b = CachedBackend(inner_b, cache)

    a.embed(["hello"])
    b.embed(["hello"])  # should NOT hit a's cache despite identical text
    assert inner_b.calls == 1


def test_cache_persists_across_instances(tmp_path):
    db = tmp_path / "embed.db"

    cache1 = EmbedCache(db)
    inner1 = _Counting()
    CachedBackend(inner1, cache1).embed(["persistent"])
    assert inner1.calls == 1

    # Fresh process-equivalent: new EmbedCache pointing at same db.
    cache2 = EmbedCache(db)
    inner2 = _Counting()
    CachedBackend(inner2, cache2).embed(["persistent"])
    assert inner2.calls == 0  # cache hit


def test_clear_drops_entries(tmp_path):
    cache = EmbedCache(tmp_path / "embed.db")
    inner = _Counting()
    wrapped = CachedBackend(inner, cache)
    wrapped.embed(["a", "b", "c"])
    assert sum(cache.stats().values()) == 3

    cache.clear()
    assert cache.stats() == {}


def test_cap_evicts_least_recently_used(tmp_path, monkeypatch):
    """Eviction is LRU, not insertion order.

    This used to assert FIFO (drop the lowest rowid). That policy is actively
    harmful: a sequential re-index walks a corpus in roughly insertion order,
    so evicting the oldest rows drops precisely the entries the walk is about
    to ask for. Reading an entry has to protect it.
    """
    monkeypatch.setenv("VECGREP_EMBED_CACHE_MAX_ROWS", "10")
    cache = EmbedCache(tmp_path / "embed.db")

    first = [f"first-{i}" for i in range(5)]
    second = [f"second-{i}" for i in range(5)]
    cache.put_many("id", first, [[float(i)] for i in range(5)])
    cache.put_many("id", second, [[float(i)] for i in range(5)])
    assert sum(cache.stats().values()) == 10

    # Touch the OLDER batch. Under FIFO these are next to die; under LRU the
    # read makes them the freshest thing in the cache.
    assert len(cache.get_many("id", first)) == 5

    third = [f"third-{i}" for i in range(5)]
    cache.put_many("id", third, [[float(i)] for i in range(5)])
    assert sum(cache.stats().values()) == 10

    assert len(cache.get_many("id", first)) == 5, "recently read rows must survive"
    assert cache.get_many("id", second) == {}, "untouched rows are the eviction target"
    assert len(cache.get_many("id", third)) == 5


def test_warm_corpus_reindexes_without_embedding(tmp_path, monkeypatch):
    """The whole point of the cache: repairing a corpus that is already warm
    must not call the embedding backend at all. Holds only while the cap is at
    least the corpus size -- see the thrash test below for what happens when it
    is not."""
    corpus = [f"chunk-{i:04d}" for i in range(60)]
    monkeypatch.setenv("VECGREP_EMBED_CACHE_MAX_ROWS", str(len(corpus)))
    cache = EmbedCache(tmp_path / "embed.db")
    inner = _Counting()
    cache.put_many(inner.identity, corpus, [[float(i)] for i in range(len(corpus))])

    wrapped = CachedBackend(inner, cache)
    for i in range(0, len(corpus), 10):
        wrapped.embed(corpus[i : i + 10])

    assert inner.calls == 0, "a warm corpus must re-index without any embedding"


def test_cap_below_corpus_size_thrashes_every_pass(tmp_path, monkeypatch):
    """Why the default cap must exceed a real corpus.

    A cap smaller than the corpus cannot hold it, so a one-pass walk evicts
    entries it has not reached yet and the next pass re-embeds everything. No
    eviction policy saves this -- LRU protects entries that have been *read*,
    and a repair walk has not read its pending entries yet. Sizing the cap is
    the fix; this test exists so nobody "optimises" the cap back down and
    quietly reintroduces a full re-embed on every repair.
    """
    corpus = [f"chunk-{i:04d}" for i in range(60)]
    monkeypatch.setenv("VECGREP_EMBED_CACHE_MAX_ROWS", "20")  # a third of the corpus
    cache = EmbedCache(tmp_path / "embed.db")
    inner = _Counting()
    wrapped = CachedBackend(inner, cache)

    for _ in range(2):
        for i in range(0, len(corpus), 10):
            wrapped.embed(corpus[i : i + 10])

    # 6 batches per pass; an adequately sized cache would make pass 2 free.
    assert inner.calls > 6, "undersized cache re-embeds on the second pass too"


def test_legacy_db_without_last_used_migrates(tmp_path):
    """An existing cache predates the LRU column and must not be thrown away."""
    db = tmp_path / "embed.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE embed_cache (identity TEXT NOT NULL, text_sha TEXT NOT NULL, "
        "vector TEXT NOT NULL, PRIMARY KEY (identity, text_sha))"
    )
    conn.execute(
        "INSERT INTO embed_cache (identity, text_sha, vector) VALUES (?, ?, ?)",
        ("fake:fake-1", EmbedCache._sha("legacy"), "[1.0, 0.0, 0.0, 0.0]"),
    )
    conn.commit()
    conn.close()

    cache = EmbedCache(db)
    cols = {r[1] for r in cache._conn.execute("PRAGMA table_info(embed_cache)")}
    assert "last_used" in cols
    # The pre-existing row survives the migration and still serves a hit.
    inner = _Counting()
    assert CachedBackend(inner, cache).embed(["legacy"]) == [[1.0, 0.0, 0.0, 0.0]]
    assert inner.calls == 0


def test_default_cap_fits_a_large_corpus():
    """Guard the constant. A cap below a real corpus makes the cache worse than
    useless: the walk evicts its own pending hits and every repair re-embeds
    from scratch. Transcript corpora reach hundreds of thousands of chunks."""
    assert DEFAULT_MAX_ROWS >= 500_000


def test_cap_disabled_when_nonpositive(tmp_path, monkeypatch):
    monkeypatch.setenv("VECGREP_EMBED_CACHE_MAX_ROWS", "0")
    cache = EmbedCache(tmp_path / "embed.db")
    texts = [f"t{i}" for i in range(20)]
    cache.put_many("id", texts, [[float(i)] for i in range(20)])
    assert sum(cache.stats().values()) == 20


# ─────────── concurrency (added 2026-08-07, after a live failure) ───────────
# The cache file is shared by every vecgrep process on a host: the serve
# daemon, the watchers, and any one-shot reindex. When get_many started
# touching last_used for LRU it turned every READER into a WRITER, and in
# sqlite's default rollback-journal mode one writer takes an EXCLUSIVE lock
# that shuts readers out. A long corpus recovery died mid-run on
# "sqlite3.OperationalError: database is locked" raised from a plain SELECT.

def test_reader_survives_a_concurrent_writer_holding_a_transaction(tmp_path):
    db = tmp_path / "embed.db"
    reader = EmbedCache(db)
    reader.put_many("id", ["a", "b"], [[1.0], [2.0]])

    # A second process mid-write, holding the lock.
    other = sqlite3.connect(str(db))
    other.execute("PRAGMA busy_timeout=10000")
    other.execute("BEGIN IMMEDIATE")
    other.execute(
        "INSERT OR REPLACE INTO embed_cache (identity, text_sha, vector, last_used) "
        "VALUES (?, ?, ?, ?)",
        ("id", EmbedCache._sha("c"), "[3.0]", 1),
    )
    try:
        # Must still serve the hit rather than raising.
        got = reader.get_many("id", ["a", "b"])
        assert len(got) == 2
    finally:
        other.rollback()
        other.close()


def test_cache_uses_wal_so_readers_do_not_block(tmp_path):
    cache = EmbedCache(tmp_path / "embed.db")
    mode = cache._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_touch_failure_never_breaks_a_lookup(tmp_path, monkeypatch):
    """Eviction bookkeeping is optional; the vector it returns is not."""
    cache = EmbedCache(tmp_path / "embed.db")
    cache.put_many("id", ["a"], [[9.0]])

    def boom(*_a, **_k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(cache, "_touch_locked", boom)
    assert cache.get_many("id", ["a"]) == {EmbedCache._sha("a"): [9.0]}
