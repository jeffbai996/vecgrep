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


def test_cache_write_contention_never_discards_fresh_vectors(tmp_path, monkeypatch):
    """Persistence is optional; a computed embedding is the real result."""
    cache = EmbedCache(tmp_path / "embed.db")
    inner = _Counting()
    backend = CachedBackend(inner, cache)

    class LockedConnection:
        def __init__(self, inner):
            self.inner = inner

        def execute(self, *args, **kwargs):
            return self.inner.execute(*args, **kwargs)

        def executemany(self, *_a, **_k):
            raise sqlite3.OperationalError("database is locked")

        def rollback(self):
            return self.inner.rollback()

    monkeypatch.setattr(cache, "_conn", LockedConnection(cache._conn))
    assert backend.embed(["fresh"]) == [[5.0, 0.0, 0.0, 0.0]]
    assert inner.calls == 1


def test_read_only_cache_serves_hits_without_any_mutation(tmp_path):
    """Recovery may read a shared warm cache but must never contend on it."""
    db = tmp_path / "embed.db"
    writer = EmbedCache(db)
    writer.put_many("id", ["warm"], [[3.0]])
    before = writer._conn.execute(
        "SELECT last_used FROM embed_cache WHERE identity = ? AND text_sha = ?",
        ("id", EmbedCache._sha("warm")),
    ).fetchone()[0]

    reader = EmbedCache(db, read_only=True)
    assert reader.get_many("id", ["warm"]) == {EmbedCache._sha("warm"): [3.0]}
    reader.put_many("id", ["cold"], [[4.0]])

    after = writer._conn.execute(
        "SELECT last_used FROM embed_cache WHERE identity = ? AND text_sha = ?",
        ("id", EmbedCache._sha("warm")),
    ).fetchone()[0]
    cold = writer._conn.execute(
        "SELECT 1 FROM embed_cache WHERE identity = ? AND text_sha = ?",
        ("id", EmbedCache._sha("cold")),
    ).fetchone()
    assert after == before
    assert cold is None


# ── storage format + housekeeping (M4, 2026-08-17) ───────────────────────────
#
# The live cache reached 4.0 GB for 292k rows because every vector was a JSON
# text of ~13.8 KB. Rows are now float32 blobs (4 KB at 1024-dim); legacy JSON
# rows still read; `compact()` rewrites them; `sweep()` deletes rows no live
# corpus references so the cache stops being append-only.

def _raw_rows(db):
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT identity, text_sha, vector FROM embed_cache").fetchall()
    finally:
        conn.close()


def test_new_rows_are_float32_blobs_that_round_trip(tmp_path):
    db = tmp_path / "embed.db"
    cache = EmbedCache(db)
    vec = [0.1234567, -0.75, 1e-3, 2.5]
    cache.put_many("id", ["t"], [vec])
    rows = _raw_rows(db)
    assert len(rows) == 1 and isinstance(rows[0][2], bytes)
    assert len(rows[0][2]) == 4 * len(vec)          # float32, not JSON text
    got = cache.get_many("id", ["t"])[EmbedCache._sha("t")]
    assert len(got) == 4
    for a, b in zip(got, vec):
        assert abs(a - b) < 1e-6                     # float32 precision


def test_legacy_json_rows_still_read_and_compact_converts_them(tmp_path):
    db = tmp_path / "embed.db"
    cache = EmbedCache(db)
    sha = EmbedCache._sha("old")
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO embed_cache (identity, text_sha, vector, last_used) VALUES (?,?,?,0)",
                 ("id", sha, "[1.0, 2.0, 3.0]"))
    conn.commit(); conn.close()
    assert cache.get_many("id", ["old"])[sha] == [1.0, 2.0, 3.0]
    cache.put_many("id", ["new"], [[4.0, 5.0, 6.0]])
    rep = cache.compact(vacuum=False)
    assert rep["converted"] == 1 and rep["already_blob"] == 1
    rows = {r[1]: r[2] for r in _raw_rows(db)}
    assert isinstance(rows[sha], bytes) and len(rows[sha]) == 12
    assert cache.get_many("id", ["old"])[sha] == [1.0, 2.0, 3.0]
    # idempotent
    assert cache.compact(vacuum=False)["converted"] == 0


def test_sweep_deletes_only_rows_outside_the_keep_set(tmp_path):
    cache = EmbedCache(tmp_path / "embed.db")
    live = ["a", "b"]
    dead = ["c", "d", "e"]
    cache.put_many("id", live + dead, [[1.0]] * 5)
    cache.put_many("other", ["x"], [[1.0]])
    keep = {"id": {EmbedCache._sha(t) for t in live}}
    dry = cache.sweep(keep, dry_run=True)
    assert dry == {"id": 3, "other": 1}
    assert sum(cache.stats().values()) == 6, "dry run must not delete"
    real = cache.sweep(keep)
    assert real == {"id": 3, "other": 1}
    assert cache.stats() == {"id": 2}
    assert len(cache.get_many("id", live)) == 2


def test_sweep_can_be_scoped_to_one_identity(tmp_path):
    cache = EmbedCache(tmp_path / "embed.db")
    cache.put_many("id", ["a", "dead"], [[1.0]] * 2)
    cache.put_many("other", ["x"], [[1.0]])
    got = cache.sweep({"id": {EmbedCache._sha("a")}}, identities=["id"])
    assert got == {"id": 1}
    assert cache.stats() == {"id": 1, "other": 1}, "identities outside the scope are untouched"


def test_stats_bytes_reports_per_identity_size(tmp_path):
    cache = EmbedCache(tmp_path / "embed.db")
    cache.put_many("id", ["a", "b"], [[1.0] * 8] * 2)
    s = cache.stats_bytes()
    assert s["id"]["rows"] == 2 and s["id"]["bytes"] == 2 * 8 * 4


# ── cap headroom warning ────────────────────────────────────────────────────
# The cap comment documents the failure mode (a cap below the corpus size
# turns a repair into a full overnight re-embed) but nothing watched for the
# approach. Found 2026-09-08 with the cache at 83% of cap and growing.

def _vecs(n: int) -> tuple[list[str], list[list[float]]]:
    return [f"t{i}" for i in range(n)], [[1.0, 0.0, 0.0, 0.0]] * n


def test_cap_warning_fires_near_cap(tmp_path, caplog, monkeypatch):
    import logging

    monkeypatch.setenv("VECGREP_EMBED_CACHE_MAX_ROWS", "10")
    cache = EmbedCache(tmp_path / "embed.db")
    texts, vecs = _vecs(9)  # 90% of cap
    with caplog.at_level(logging.WARNING, logger="vecgrep.backend.embed.cache"):
        cache.put_many("id", texts, vecs)
    assert "VECGREP_EMBED_CACHE_MAX_ROWS" in caplog.text, (
        "operator must be told which knob to raise before the cap binds"
    )


def test_cap_warning_silent_below_threshold(tmp_path, caplog, monkeypatch):
    import logging

    monkeypatch.setenv("VECGREP_EMBED_CACHE_MAX_ROWS", "10")
    cache = EmbedCache(tmp_path / "embed.db")
    texts, vecs = _vecs(8)  # below the 90% line
    with caplog.at_level(logging.WARNING, logger="vecgrep.backend.embed.cache"):
        cache.put_many("id", texts, vecs)
    assert "VECGREP_EMBED_CACHE_MAX_ROWS" not in caplog.text


def test_cap_warning_fires_once_not_per_put(tmp_path, caplog, monkeypatch):
    import logging

    monkeypatch.setenv("VECGREP_EMBED_CACHE_MAX_ROWS", "10")
    cache = EmbedCache(tmp_path / "embed.db")
    texts, vecs = _vecs(9)
    with caplog.at_level(logging.WARNING, logger="vecgrep.backend.embed.cache"):
        cache.put_many("id", texts, vecs)
        cache.put_many("id", *_vecs(9))  # same rows again, still at 90%
    warnings = [r for r in caplog.records if "VECGREP_EMBED_CACHE_MAX_ROWS" in r.message]
    assert len(warnings) == 1


def test_eviction_is_logged_when_cap_binds(tmp_path, caplog, monkeypatch):
    import logging

    monkeypatch.setenv("VECGREP_EMBED_CACHE_MAX_ROWS", "5")
    cache = EmbedCache(tmp_path / "embed.db")
    texts, vecs = _vecs(8)
    with caplog.at_level(logging.WARNING, logger="vecgrep.backend.embed.cache"):
        cache.put_many("id", texts, vecs)
    assert cache.stats() == {"id": 5}
    assert any("evict" in r.message.lower() for r in caplog.records), (
        "the cap binding is the documented danger state; it must be visible"
    )


def test_cap_disabled_stays_silent(tmp_path, caplog, monkeypatch):
    import logging

    monkeypatch.setenv("VECGREP_EMBED_CACHE_MAX_ROWS", "0")
    cache = EmbedCache(tmp_path / "embed.db")
    texts, vecs = _vecs(50)
    with caplog.at_level(logging.WARNING, logger="vecgrep.backend.embed.cache"):
        cache.put_many("id", texts, vecs)
    assert "VECGREP_EMBED_CACHE_MAX_ROWS" not in caplog.text


# ── sweep delete-fraction guard ─────────────────────────────────────────────
# An unattended sweep that runs while a corpus is empty or mid-rebuild would
# delete exactly the vectors that make the rebuild cheap. The guard makes a
# grossly lopsided sweep abort loudly instead of proceeding.

def test_sweep_guard_aborts_on_excessive_deletion(tmp_path):
    import pytest

    cache = EmbedCache(tmp_path / "embed.db")
    texts, vecs = _vecs(10)
    cache.put_many("id", texts, vecs)
    keep = {"id": {EmbedCache._sha(t) for t in texts[:2]}}  # would delete 8/10

    with pytest.raises(RuntimeError, match="fraction"):
        cache.sweep(keep, max_delete_fraction=0.5)
    assert cache.stats() == {"id": 10}, "an aborted sweep must delete nothing"


def test_sweep_guard_allows_within_fraction(tmp_path):
    cache = EmbedCache(tmp_path / "embed.db")
    texts, vecs = _vecs(10)
    cache.put_many("id", texts, vecs)
    keep = {"id": {EmbedCache._sha(t) for t in texts[:8]}}  # deletes 2/10

    got = cache.sweep(keep, max_delete_fraction=0.5)
    assert got == {"id": 2}
    assert cache.stats() == {"id": 8}


def test_sweep_guard_not_applied_to_dry_run(tmp_path):
    cache = EmbedCache(tmp_path / "embed.db")
    texts, vecs = _vecs(10)
    cache.put_many("id", texts, vecs)
    keep = {"id": set()}  # would delete everything

    got = cache.sweep(keep, dry_run=True, max_delete_fraction=0.5)
    assert got == {"id": 10}, "dry run reports the damage instead of refusing"
    assert cache.stats() == {"id": 10}


# --- steady-state write churn -------------------------------------------------
#
# The LRU bookkeeping turned every cache HIT into an UPDATE + COMMIT. With an
# index on last_used each touched row dirties a table page and an index page,
# measured at ~12 KB of WAL for a single-text lookup and 2.1 MB for a 500-text
# batch -- write amplification on a pure read path, on the SSD, forever. The
# fix buffers the timestamps in memory and flushes them on a timer, at
# shutdown, and before eviction needs them to pick a victim.

def _wal_size(cache) -> int:
    from pathlib import Path
    p = Path(cache._conn.execute("PRAGMA database_list").fetchone()[2] + "-wal")
    return p.stat().st_size if p.exists() else 0


def test_cache_hit_writes_nothing_to_disk(tmp_path):
    """A read must not cost a write. This is the whole point of the change."""
    cache = EmbedCache(tmp_path / "embed.db")
    cache.put_many("id", [f"t{i}" for i in range(200)], [[1.0]] * 200)
    cache._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    before = _wal_size(cache)

    for _ in range(5):
        cache.get_many("id", [f"t{i}" for i in range(200)])

    assert _wal_size(cache) == before


def test_buffered_touches_flush_on_close(tmp_path):
    """Deferred is not discarded: a clean shutdown persists the LRU order."""
    db = tmp_path / "embed.db"
    cache = EmbedCache(db)
    cache.put_many("id", ["warm"], [[1.0]])
    sha = EmbedCache._sha("warm")
    cache._conn.execute("UPDATE embed_cache SET last_used = 0")
    cache._conn.commit()

    cache.get_many("id", ["warm"])
    cache.close()

    conn = sqlite3.connect(str(db))
    (stored,) = conn.execute(
        "SELECT last_used FROM embed_cache WHERE text_sha = ?", (sha,)
    ).fetchone()
    assert stored > 0


def test_eviction_sees_pending_touches(tmp_path, monkeypatch):
    """A buffered read must still protect its row from being evicted."""
    monkeypatch.setenv("VECGREP_EMBED_CACHE_MAX_ROWS", "3")
    cache = EmbedCache(tmp_path / "embed.db")
    cache.put_many("id", ["a", "b", "c"], [[1.0], [2.0], [3.0]])
    cache._conn.execute("UPDATE embed_cache SET last_used = 0")
    cache._conn.commit()

    # "a" is read (hot, buffered only), then a new row pushes us over the cap.
    cache.get_many("id", ["a"])
    cache.put_many("id", ["d"], [[4.0]])

    assert cache.get_many("id", ["a"]) == {EmbedCache._sha("a"): [1.0]}


def test_touches_flush_once_the_interval_elapses(tmp_path, monkeypatch):
    """The buffer is bounded in time, so a crash loses minutes, not months."""
    cache = EmbedCache(tmp_path / "embed.db")
    cache.put_many("id", ["warm"], [[1.0]])
    cache._conn.execute("UPDATE embed_cache SET last_used = 0")
    cache._conn.commit()

    cache.get_many("id", ["warm"])
    (stored,) = cache._conn.execute("SELECT last_used FROM embed_cache").fetchone()
    assert stored == 0  # still buffered

    cache._last_flush = 0.0  # pretend the interval has elapsed
    cache.get_many("id", ["warm"])

    (stored,) = cache._conn.execute("SELECT last_used FROM embed_cache").fetchone()
    assert stored > 0
