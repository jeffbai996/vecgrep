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
import threading
import time
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


class EmbedCache:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because the FastAPI server may serve
        # concurrent requests; the lock below serializes writes.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
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
                for sha, vec_json in cur.fetchall():
                    out[sha] = json.loads(vec_json)
            if out:
                # A read is what makes an entry worth keeping. One UPDATE per
                # batch, not per row, so this stays cheap on the hot path.
                self._touch_locked(identity, list(out))
                self._conn.commit()
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
        now = self._now()
        rows = [
            (identity, self._sha(t), json.dumps(v), now)
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

    def clear(self, identity: str | None = None) -> int:
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
