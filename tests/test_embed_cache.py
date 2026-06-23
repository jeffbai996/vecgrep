"""Embedding cache tests.

Past bugs / risks covered:
- Cached vectors keyed wrongly across (model, text)
- Cache miss leaks the wrong text -> wrong vector
"""
from __future__ import annotations

from vecgrep.backend.embed.base import EmbedBackend
from vecgrep.backend.embed.cache import CachedBackend, EmbedCache


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


def test_cap_evicts_oldest_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("VECGREP_EMBED_CACHE_MAX_ROWS", "5")
    cache = EmbedCache(tmp_path / "embed.db")

    old = [f"old-{i}" for i in range(5)]
    new = [f"new-{i}" for i in range(5)]

    # First batch fills exactly to the cap — nothing evicted.
    cache.put_many("id", old, [[float(i)] for i in range(5)])
    assert sum(cache.stats().values()) == 5

    # Second batch pushes over — the 5 oldest (first batch) get dropped.
    cache.put_many("id", new, [[float(i)] for i in range(5)])
    assert sum(cache.stats().values()) == 5

    # Oldest gone, newest survive.
    assert cache.get_many("id", old) == {}
    assert len(cache.get_many("id", new)) == 5


def test_cap_disabled_when_nonpositive(tmp_path, monkeypatch):
    monkeypatch.setenv("VECGREP_EMBED_CACHE_MAX_ROWS", "0")
    cache = EmbedCache(tmp_path / "embed.db")
    texts = [f"t{i}" for i in range(20)]
    cache.put_many("id", texts, [[float(i)] for i in range(20)])
    assert sum(cache.stats().values()) == 20
