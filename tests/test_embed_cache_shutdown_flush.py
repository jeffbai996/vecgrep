"""Deferred LRU stamps must survive a clean shutdown.

Buffering cache-hit bookkeeping in memory (see _TOUCH_FLUSH_SECONDS) removed
gigabytes a day of WAL churn, but it also means a restart can drop up to one
interval of reads. The service exposes a flush so the server's lifespan can
land them on the way down; without it, a daily restart would keep the whole
cache looking cold and eviction would pick victims at random.
"""
from __future__ import annotations

import sqlite3

from vecgrep.backend.embed.cache import EmbedCache


def test_service_flush_persists_pending_touches(tmp_path, monkeypatch):
    from vecgrep.backend.service import VecgrepService

    monkeypatch.setenv("VECGREP_HOME", str(tmp_path))
    svc = VecgrepService()
    cache = svc._embed_cache
    assert cache is not None

    cache.put_many("id", ["warm"], [[1.0]])
    cache._conn.execute("UPDATE embed_cache SET last_used = 0")
    cache._conn.commit()
    cache.get_many("id", ["warm"])

    (buffered,) = cache._conn.execute("SELECT last_used FROM embed_cache").fetchone()
    assert buffered == 0

    svc.flush_embed_cache()

    (stored,) = cache._conn.execute("SELECT last_used FROM embed_cache").fetchone()
    assert stored > 0


def test_flush_is_safe_without_a_cache(tmp_path, monkeypatch):
    """Ephemeral services have no cache; shutdown must not raise."""
    from vecgrep.backend.service import VecgrepService

    monkeypatch.setenv("VECGREP_HOME", str(tmp_path))
    svc = VecgrepService(ephemeral=True)
    assert svc._embed_cache is None
    svc.flush_embed_cache()  # must not raise
