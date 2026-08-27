"""The BM25 cache is budgeted in bytes, not corpora.

Counting corpora was the wrong unit. Real sidecars span four orders of
magnitude — 27 kB for a tiny corpus, 270 MB for a large one — so "keep 1
corpus" is not a limit on memory at all, and it forced a re-unpickle of the
largest index every time a caller touched a different corpus.

That is not hypothetical. The squad memory hook searches three corpora on every
prompt submission of every bot, so it happened on every turn. Measured on a live
deployment 2026-08-26: twelve searches against ONE corpus cost -3 MB; twelve
searches ROTATING three corpora cost +957 MB and the memory never came back,
because repeatedly building and discarding a 270 MB object graph shreds the
allocator's arenas. The server grew 859 MB -> 2782 MB in 8.5 hours.
"""
from __future__ import annotations

import pickle

from vecgrep.backend.store.bm25_store import BM25Store, DEFAULT_CACHE_BYTES


def _corpus(store: BM25Store, name: str, filler_bytes: int) -> None:
    """Write a sidecar of roughly `filler_bytes` so the budget can see it."""
    p = store._path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    idx = store._load(name)
    idx.ids.append("x" * filler_bytes)
    p.write_bytes(pickle.dumps(idx))
    store.evict(name)


def test_small_corpora_all_stay_resident(tmp_path):
    # The whole point: a handful of tiny corpora must not evict each other.
    store = BM25Store(tmp_path, max_cached_bytes=1_000_000)
    for name in ("a", "b", "c", "d"):
        _corpus(store, name, 1000)
    for name in ("a", "b", "c", "d"):
        store._load(name)
    assert set(store._cache) == {"a", "b", "c", "d"}


def test_a_large_corpus_evicts_to_stay_inside_the_budget(tmp_path):
    store = BM25Store(tmp_path, max_cached_bytes=200_000)
    _corpus(store, "small", 1000)
    _corpus(store, "big", 150_000)
    store._load("small")
    store._load("big")
    assert "big" in store._cache
    assert sum(store._cached_bytes.values()) <= 200_000


def test_rotating_corpora_does_not_thrash_when_they_fit(tmp_path):
    # The regression this exists for: rotating between corpora that all fit
    # inside the budget must load each ONCE, not once per visit.
    store = BM25Store(tmp_path, max_cached_bytes=1_000_000)
    for name in ("one", "two", "three"):
        _corpus(store, name, 1000)
    # Count real UNPICKLES, not visits: _load consults _disk_version on every
    # call, including cache hits, so counting that would count visits.
    import vecgrep.backend.store.bm25_store as mod

    unpickles = []
    real_loads = mod.pickle.loads

    def counting(data):
        unpickles.append(len(data))
        return real_loads(data)

    mod.pickle.loads = counting
    try:
        for _ in range(5):
            for name in ("one", "two", "three"):
                store._load(name)
    finally:
        mod.pickle.loads = real_loads
    # three unpickles total, not fifteen
    assert len(unpickles) == 3, len(unpickles)


def test_one_corpus_bigger_than_the_budget_is_still_servable(tmp_path):
    store = BM25Store(tmp_path, max_cached_bytes=1000)
    _corpus(store, "huge", 50_000)
    idx = store._load("huge")
    assert idx is not None
    assert "huge" in store._cache


def test_default_budget_is_bytes_and_generous_enough_to_matter(tmp_path):
    store = BM25Store(tmp_path)
    assert store.max_cached_bytes == DEFAULT_CACHE_BYTES
    assert store.max_cached_corpora is None
    assert DEFAULT_CACHE_BYTES >= 128 * 1024 * 1024


def test_ephemeral_store_is_still_unbounded(tmp_path):
    store = BM25Store(None)
    assert store.max_cached_bytes is None
    assert store.max_cached_corpora is None


def test_a_count_cap_still_works_when_asked_for(tmp_path):
    store = BM25Store(tmp_path, max_cached_corpora=1, max_cached_bytes=10**9)
    for name in ("p", "q"):
        _corpus(store, name, 1000)
    store._load("p")
    store._load("q")
    assert set(store._cache) == {"q"}
