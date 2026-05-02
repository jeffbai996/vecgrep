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
import sqlite3
import threading
from pathlib import Path

from .base import EmbedBackend


_SCHEMA = """
CREATE TABLE IF NOT EXISTS embed_cache (
    identity TEXT NOT NULL,
    text_sha TEXT NOT NULL,
    vector TEXT NOT NULL,
    PRIMARY KEY (identity, text_sha)
);
"""


class EmbedCache:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because the FastAPI server may serve
        # concurrent requests; the lock below serializes writes.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()

    @staticmethod
    def _sha(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

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
        return out

    def put_many(self, identity: str, texts: list[str], vectors: list[list[float]]) -> None:
        if not texts:
            return
        rows = [
            (identity, self._sha(t), json.dumps(v))
            for t, v in zip(texts, vectors)
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embed_cache (identity, text_sha, vector) "
                "VALUES (?, ?, ?)",
                rows,
            )
            self._conn.commit()

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
