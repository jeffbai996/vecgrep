"""Embedding cache.

Keyed on (backend.identity, sha256(text)). Hits skip the network call
entirely. Backed by sqlite at ~/.vecgrep/embed_cache.db so it survives
across processes — useful when re-indexing the same corpus with a new
chunker, or migrating between embedding models (the hash flips on the
new model, but unrelated corpora using the old model still hit cache).

Wrapped around any EmbedBackend via CachedBackend(backend, db_path).
The wrapper is transparent: same interface, identical results.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from array import array
from pathlib import Path

from .base import EmbedBackend


_SCHEMA = """
CREATE TABLE IF NOT EXISTS embed_cache (
    identity TEXT NOT NULL,
    text_sha TEXT NOT NULL,
    vector TEXT NOT NULL,
    last_used INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (identity, text_sha)
);
"""

# Created separately from _SCHEMA, never alongside it: on a pre-LRU database the
# CREATE TABLE above is a no-op and last_used does not exist yet, so building the
# index has to wait until the migration has actually added the column.
_LRU_INDEX = "CREATE INDEX IF NOT EXISTS idx_embed_cache_lru ON embed_cache (last_used)"

# Row cap so the cache can't grow without bound.
#
# This was 50k, chosen on the assumption that a live working set is "a few
# thousand chunks" and eviction would only shed stale orphans. That assumption
# does not survive contact with a real transcript corpus, which reaches
# hundreds of thousands of chunks. Once the cap is below the corpus size the
# cache stops being an optimisation and becomes a liability: a repair walk
# cannot fit the corpus, evicts entries it has not reached yet, and the next
# repair re-embeds all of it from scratch -- turning an hour into a night.
# No eviction policy rescues that; only a cap above the corpus does.
#
# Cost of the default: entries are JSON-encoded vectors, ~14 KB for a 1024-dim
# model, so 500k rows is roughly 7 GB on disk. That is a deliberate trade -- the
# cache exists to make a full re-index cheap, and a cache too small to hold the
# corpus buys nothing at all. Hosts short on disk should lower it explicitly via
# VECGREP_EMBED_CACHE_MAX_ROWS rather than have a too-small default chosen for
# them. Set VECGREP_EMBED_CACHE_MAX_ROWS <= 0 to disable the cap entirely.
DEFAULT_MAX_ROWS = 500_000


def _max_rows() -> int:
    try:
        return int(os.environ.get("VECGREP_EMBED_CACHE_MAX_ROWS", DEFAULT_MAX_ROWS))
    except ValueError:
        return DEFAULT_MAX_ROWS


# Vector storage format. Rows written before 2026-08-17 hold the vector as
# JSON text -- ~13.8 KB for a 1024-dim model, which is how a 292k-row cache
# came to weigh 4.0 GB, three times the qdrant store it was meant to make
# cheap to rebuild. New rows are packed little-endian float32 (4 KB at
# 1024-dim, 3.4x smaller); reads accept both so nothing has to be re-embedded,
# and compact() rewrites the legacy rows in place.

def _encode(vec: list[float]) -> bytes:
    a = array("f", vec)
    if sys.byteorder != "little":
        a.byteswap()
    return a.tobytes()


def _decode(raw) -> list[float]:
    if isinstance(raw, (bytes, memoryview)):
        a = array("f")
        a.frombytes(bytes(raw))
        if sys.byteorder != "little":
            a.byteswap()
        return a.tolist()
    return json.loads(raw)


class EmbedCache:
    def __init__(self, db_path: Path, *, read_only: bool = False) -> None:
        self._read_only = read_only
        if read_only:
            # A repair needs the warm vectors, not cache bookkeeping. Opening
            # this connection as SQLite read-only prevents a large reindex from
            # competing with live serving/watchers for a write transaction.
            self._conn = sqlite3.connect(
                f"{db_path.resolve().as_uri()}?mode=ro",
                uri=True,
                check_same_thread=False,
            )
            self._conn.execute("PRAGMA busy_timeout=10000")
        else:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False because the FastAPI server may serve
            # concurrent requests; the lock below serializes writes.
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
            # This file is shared by every vecgrep process on the box -- the serve
            # daemon, the watchers, and any one-shot reindex, all at once. In the
            # default rollback-journal mode a single writer takes an EXCLUSIVE lock
            # that blocks readers outright, so once get_many started touching
            # last_used (LRU), concurrent readers began failing with "database is
            # locked" and killed a long recovery run mid-flight. WAL lets readers
            # continue while one process writes; busy_timeout makes the writers
            # themselves queue instead of raising.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=10000")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._migrate_last_used()
            self._conn.execute(_LRU_INDEX)
            self._conn.commit()
        self._lock = threading.Lock()
        # Resolved once per instance so tests can set the env before construction.
        self._max_rows = _max_rows()

    def _migrate_last_used(self) -> None:
        """Add the LRU column to a cache created before it existed.

        An established cache is worth real money in embedding calls, so this
        upgrades in place rather than dropping the table. ADD COLUMN with a
        constant default is metadata-only in sqlite, so it stays fast even on a
        multi-gigabyte file. Existing rows land at last_used=0, which correctly
        marks them as the coldest things in the cache until something reads them.
        """
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(embed_cache)")}
        if "last_used" not in cols:
            self._conn.execute(
                "ALTER TABLE embed_cache ADD COLUMN last_used INTEGER NOT NULL DEFAULT 0"
            )

    @staticmethod
    def _sha(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _now() -> int:
        # Wall clock in ns: it has to persist across processes, so a per-process
        # counter would not order correctly between runs.
        return time.time_ns()

    def get_many(self, identity: str, texts: list[str]) -> dict[str, list[float]]:
        """Return {sha: vector} for every cached text. Missing keys are absent."""
        if not texts:
            return {}
        shas = [self._sha(t) for t in texts]
        out: dict[str, list[float]] = {}
        # Chunk in groups of 500 to keep the SQL parameter count safe.
        with self._lock:
            for i in range(0, len(shas), 500):
                batch = shas[i : i + 500]
                placeholders = ",".join("?" * len(batch))
                cur = self._conn.execute(
                    f"SELECT text_sha, vector FROM embed_cache "
                    f"WHERE identity = ? AND text_sha IN ({placeholders})",
                    [identity, *batch],
                )
                for sha, raw in cur.fetchall():
                    out[sha] = _decode(raw)
            if out and not self._read_only:
                # A read is what makes an entry worth keeping. One UPDATE per
                # batch, not per row, so this stays cheap on the hot path.
                #
                # BEST EFFORT, always. This is bookkeeping for the eviction
                # order -- losing it costs nothing but a slightly worse choice
                # of victim later. Letting it raise costs a whole reindex, which
                # is exactly what happened when contention on the shared cache
                # first surfaced. A cache lookup must never be able to fail the
                # embed it was supposed to make cheaper.
                try:
                    self._touch_locked(identity, list(out))
                    self._conn.commit()
                except sqlite3.Error:
                    try:
                        self._conn.rollback()
                    except sqlite3.Error:
                        pass
        return out

    def _touch_locked(self, identity: str, shas: list[str]) -> None:
        """Mark rows as freshly used. Caller must hold self._lock."""
        now = self._now()
        for i in range(0, len(shas), 500):
            batch = shas[i : i + 500]
            placeholders = ",".join("?" * len(batch))
            self._conn.execute(
                f"UPDATE embed_cache SET last_used = ? "
                f"WHERE identity = ? AND text_sha IN ({placeholders})",
                [now, identity, *batch],
            )

    def put_many(self, identity: str, texts: list[str], vectors: list[list[float]]) -> None:
        if not texts:
            return
        if self._read_only:
            return
        now = self._now()
        rows = [
            (identity, self._sha(t), _encode(v), now)
            for t, v in zip(texts, vectors)
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embed_cache "
                "(identity, text_sha, vector, last_used) VALUES (?, ?, ?, ?)",
                rows,
            )
            self._evict_over_cap_locked()
            self._conn.commit()

    def _evict_over_cap_locked(self) -> None:
        """Drop the coldest rows if over the cap. Caller must hold self._lock.

        Coldest = lowest last_used (LRU), ties broken by insertion order. This
        replaced eviction by rowid (FIFO), which discards an entry that is read
        every day purely because it was written first. Note that LRU does NOT
        rescue an undersized cache: it protects entries that have been read, and
        a one-pass repair has not read its pending entries yet. Sizing the cap
        above the corpus is what makes repair cheap; this only stops steady-state
        search traffic from evicting its own hot set.

        sqlite reuses the freed pages for subsequent inserts, so the file
        stabilizes near the cap instead of growing forever. No VACUUM needed.
        """
        if self._max_rows <= 0:
            return
        (count,) = self._conn.execute("SELECT COUNT(*) FROM embed_cache").fetchone()
        overage = count - self._max_rows
        if overage <= 0:
            return
        self._conn.execute(
            "DELETE FROM embed_cache WHERE rowid IN "
            "(SELECT rowid FROM embed_cache ORDER BY last_used ASC, rowid ASC LIMIT ?)",
            (overage,),
        )

    def stats(self) -> dict:
        with self._lock:
            cur = self._conn.execute(
                "SELECT identity, COUNT(*) FROM embed_cache GROUP BY identity"
            )
            return {row[0]: row[1] for row in cur.fetchall()}

    def stats_bytes(self) -> dict[str, dict]:
        """{identity: {rows, bytes}} -- bytes is the stored vector payload
        (JSON text or float32 blob), which is what actually fills the file."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT identity, COUNT(*), COALESCE(SUM(LENGTH(vector)), 0) "
                "FROM embed_cache GROUP BY identity"
            )
            return {row[0]: {"rows": row[1], "bytes": row[2]} for row in cur.fetchall()}

    def compact(self, *, vacuum: bool = True, batch: int = 2000) -> dict:
        """Rewrite legacy JSON rows as float32 blobs, then optionally VACUUM so
        the file actually shrinks (sqlite reuses freed pages but never returns
        them without a VACUUM; VACUUM needs up to 2x the file in free disk).
        Safe to run against a live cache: work is done in small committed
        batches under the busy timeout, so writers queue rather than fail."""
        if self._read_only:
            raise RuntimeError("embed cache is read-only")
        converted = already = 0
        while True:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT rowid, vector FROM embed_cache "
                    "WHERE typeof(vector) = 'text' LIMIT ?", (batch,)
                ).fetchall()
                if not rows:
                    break
                self._conn.executemany(
                    "UPDATE embed_cache SET vector = ? WHERE rowid = ?",
                    [(_encode(json.loads(raw)), rid) for rid, raw in rows],
                )
                self._conn.commit()
            converted += len(rows)
        with self._lock:
            (already,) = self._conn.execute(
                "SELECT COUNT(*) FROM embed_cache WHERE typeof(vector) = 'blob'"
            ).fetchone()
            already -= converted
            if vacuum:
                self._conn.execute("VACUUM")
        return {"converted": converted, "already_blob": already, "vacuumed": vacuum}

    def sweep(
        self,
        keep: dict[str, set[str]],
        *,
        identities: list[str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """Delete every row whose (identity, sha) is not in `keep`.

        `keep` maps identity -> set of text shas some live corpus still holds.
        An identity absent from `keep` (a model no corpus uses any more) is
        entirely orphaned. `identities` restricts the sweep to those
        identities; default is every identity present. Returns
        {identity: rows deleted (or would be, under dry_run)}.
        """
        if self._read_only and not dry_run:
            raise RuntimeError("embed cache is read-only")
        out: dict[str, int] = {}
        with self._lock:
            present = [r[0] for r in self._conn.execute(
                "SELECT DISTINCT identity FROM embed_cache").fetchall()]
            for ident in present:
                if identities is not None and ident not in identities:
                    continue
                keep_shas = keep.get(ident, set())
                cur = self._conn.execute(
                    "SELECT text_sha FROM embed_cache WHERE identity = ?", (ident,))
                victims = [row[0] for row in cur.fetchall() if row[0] not in keep_shas]
                out[ident] = len(victims)
                if dry_run or not victims:
                    continue
                for i in range(0, len(victims), 500):
                    part = victims[i:i + 500]
                    self._conn.execute(
                        f"DELETE FROM embed_cache WHERE identity = ? AND text_sha IN "
                        f"({','.join('?' * len(part))})",
                        [ident, *part],
                    )
                self._conn.commit()
        return {k: v for k, v in out.items() if v or dry_run}

    def clear(self, identity: str | None = None) -> int:
        if self._read_only:
            raise RuntimeError("embed cache is read-only")
        with self._lock:
            if identity:
                cur = self._conn.execute(
                    "DELETE FROM embed_cache WHERE identity = ?", (identity,)
                )
            else:
                cur = self._conn.execute("DELETE FROM embed_cache")
            self._conn.commit()
            return cur.rowcount


class CachedBackend(EmbedBackend):
    """Transparent caching wrapper around any EmbedBackend.

    Cache is keyed on the wrapped backend's `identity` so two backends with
    different models live in the same DB without collision.

    Set `bypass=True` on a per-call basis (e.g. during --force reindex) to
    skip the cache and overwrite stale entries with fresh embeddings.
    """

    def __init__(self, inner: EmbedBackend, cache: EmbedCache) -> None:
        self._inner = inner
        self._cache = cache
        self.name = inner.name
        self.model = inner.model
        self.dim = inner.dim
        self.bypass = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        identity = self._inner.identity
        if self.bypass:
            # Refresh: re-embed everything, overwrite cache entries.
            vectors = self._inner.embed(texts)
            self._cache.put_many(identity, texts, vectors)
            return vectors
        cached = self._cache.get_many(identity, texts)
        shas = [EmbedCache._sha(t) for t in texts]
        missing_idx = [i for i, s in enumerate(shas) if s not in cached]
        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            new_vectors = self._inner.embed(missing_texts)
            self._cache.put_many(identity, missing_texts, new_vectors)
            for i, vec in zip(missing_idx, new_vectors):
                cached[shas[i]] = vec
        return [cached[s] for s in shas]
