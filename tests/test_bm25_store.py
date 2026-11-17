from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from vecgrep.backend import service as sm
from vecgrep.backend.store.bm25_sqlite import BM25SqliteStore
from vecgrep.backend.store.bm25_store import (
    DEFAULT_CACHE_BYTES,
    BM25Store,
    query_tokens,
    tokenize,
)
from vecgrep.backend.store.corpora import CorpusError


def test_tokenize_splits_identifiers():
    assert tokenize("sharpe_ratio") == ["sharpe", "ratio"]
    assert tokenize("getUserName") == ["get", "user", "name"]
    assert tokenize("HTTPSRequest") == ["https", "request"]
    assert tokenize("plain English") == ["plain", "english"]
    assert tokenize("CPU2024") == ["cpu", "2024"]


def test_upsert_and_search(tmp_path):
    store = BM25Store(tmp_path)
    store.upsert(
        "demo",
        ids=["a", "b", "c"],
        texts=[
            "Federal Reserve holds rates",
            "CPI inflation cools",
            "Bonds rally on dovish surprise",
        ],
        payloads=[{"source_id": "/x", "text": t, "corpus": "demo"} for t in [
            "Federal Reserve holds rates",
            "CPI inflation cools",
            "Bonds rally on dovish surprise",
        ]],
    )
    hits = store.search("demo", "CPI inflation", top_k=3)
    assert hits, "BM25 should match exact tokens"
    assert hits[0][0] == "b"  # the CPI chunk wins


def test_delete_by_source_keeps_other_sources(tmp_path):
    store = BM25Store(tmp_path)
    store.upsert(
        "demo",
        ids=["1", "2"],
        texts=["alpha beta gamma", "delta epsilon zeta"],
        payloads=[{"source_id": "/x"}, {"source_id": "/y"}],
    )
    store.delete_by_source("demo", "/x")

    hits = store.search("demo", "alpha", top_k=3)
    assert not hits, "deleted source should not surface"
    hits = store.search("demo", "delta", top_k=3)
    assert hits, "untouched source should still match"
    assert hits[0][0] == "2"


def test_persistence_round_trip(tmp_path):
    store = BM25Store(tmp_path)
    store.upsert(
        "demo",
        ids=["1"],
        texts=["persist this for next time"],
        payloads=[{"source_id": "/x"}],
    )
    # New store instance, same path — must reload from pickle.
    fresh = BM25Store(tmp_path)
    hits = fresh.search("demo", "persist", top_k=3)
    assert hits
    assert hits[0][0] == "1"


def test_cached_empty_index_reloads_when_a_rebuild_appears(tmp_path):
    """A long-lived API worker sees an atomically rebuilt index without restart."""
    reader = BM25Store(tmp_path)
    assert reader.search("demo", "quartzite", top_k=3) == []

    writer = BM25Store(tmp_path)
    writer.replace(
        "demo",
        [("1", "Quartzite is now keyword-searchable", {"source_id": "/x"})],
    )

    hits = reader.search("demo", "quartzite", top_k=3)
    assert hits
    assert hits[0][0] == "1"


def test_drop_removes_pickle(tmp_path):
    store = BM25Store(tmp_path)
    store.upsert(
        "demo",
        ids=["1"],
        texts=["something"],
        payloads=[{"source_id": "/x"}],
    )
    pkl = tmp_path / "demo.pkl"
    assert pkl.exists()
    store.drop("demo")
    assert not pkl.exists()


def test_persistent_store_bounds_loaded_corpora_by_default(tmp_path):
    # The default budget is BYTES, not corpora (see DEFAULT_CACHE_BYTES), so two
    # tiny corpora now coexist rather than evicting each other. This test keeps
    # its original subject — that a persistent store IS bounded — by asking for
    # a byte budget small enough that the second corpus must displace the first.
    store = BM25Store(tmp_path, max_cached_bytes=1)
    for corpus in ("one", "two"):
        store.upsert(
            corpus,
            ids=[corpus],
            texts=[f"{corpus} searchable text"],
            payloads=[{"source_id": f"/{corpus}"}],
        )
        store.search(corpus, corpus, top_k=1)

    assert list(store._cache) == ["two"]
    assert list(store._bm25_instances) == ["two"]


def test_evicted_corpus_reloads_with_identical_results(tmp_path):
    # Force eviction explicitly: under the default byte budget these toy
    # corpora both fit, and the point of this test is that a corpus which HAS
    # been evicted reloads to identical results.
    store = BM25Store(tmp_path, max_cached_bytes=1)
    store.upsert(
        "one",
        ids=["a", "b"],
        texts=["alpha beta gamma", "alpha delta epsilon"],
        payloads=[{"source_id": "/a"}, {"source_id": "/b"}],
    )
    before = store.search("one", "alpha beta", top_k=2)

    store.upsert(
        "two",
        ids=["c"],
        texts=["other corpus content"],
        payloads=[{"source_id": "/c"}],
    )
    assert "one" not in store._cache

    after = store.search("one", "alpha beta", top_k=2)
    assert after == before


def test_cache_limit_uses_least_recently_used_order(tmp_path):
    store = BM25Store(tmp_path, max_cached_corpora=2)
    for corpus in ("one", "two"):
        store.upsert(
            corpus,
            ids=[corpus],
            texts=[f"{corpus} searchable text"],
            payloads=[{"source_id": f"/{corpus}"}],
        )

    store.search("one", "one", top_k=1)
    store.upsert(
        "three",
        ids=["three"],
        texts=["three searchable text"],
        payloads=[{"source_id": "/three"}],
    )

    assert list(store._cache) == ["one", "three"]
    assert "two" not in store._bm25_instances


def test_ephemeral_store_keeps_all_corpora_without_persistence():
    store = BM25Store(None)
    for corpus in ("one", "two"):
        store.upsert(
            corpus,
            ids=[corpus],
            texts=[f"{corpus} searchable text"],
            payloads=[{"source_id": f"/{corpus}"}],
        )

    assert list(store._cache) == ["one", "two"]
    # an in-memory store has no budget to evict by
    assert store.max_cached_bytes is None
    assert store.max_cached_corpora is None


def test_short_query_requires_full_coverage(monkeypatch: pytest.MonkeyPatch):
    """2-token query against partial-match doc should not return that doc.

    Filter mode is now opt-in (penalty is the default), so pin it explicitly.
    Reproduces the 'glucose monitoring' floods-the-corpus bug: doc with
    only 'monitoring' (and no 'glucose') used to score on its IDF alone
    and beat genuine vector hits in the fused ranking.
    """
    monkeypatch.setenv("VECGREP_BM25_COVERAGE_MODE", "filter")
    store = BM25Store(None)
    store.upsert(
        "t",
        ids=["a", "b"],
        texts=[
            "ibkr terminal monitoring server",  # only "monitoring"
            "glucose monitoring CGM continuous",  # both tokens
        ],
        payloads=[{"source_id": "1"}, {"source_id": "2"}],
    )
    hits = store.search("t", "glucose monitoring", top_k=5)
    ids = [h[0] for h in hits]
    assert ids == ["b"], f"only the full-match doc should survive, got {ids}"


def test_disable_coverage_filter_via_env(monkeypatch: pytest.MonkeyPatch):
    """Safety hatch: VECGREP_BM25_DISABLE_COVERAGE_FILTER=1 restores old behavior."""
    monkeypatch.setenv("VECGREP_BM25_DISABLE_COVERAGE_FILTER", "1")
    store = BM25Store(None)
    store.upsert(
        "t",
        ids=["a", "b"],
        texts=[
            "ibkr terminal monitoring server",
            "glucose monitoring CGM continuous",
        ],
        payloads=[{"source_id": "1"}, {"source_id": "2"}],
    )
    hits = store.search("t", "glucose monitoring", top_k=5)
    ids = sorted(h[0] for h in hits)
    assert ids == ["a", "b"], f"both docs should return when filter is off, got {ids}"


def test_long_query_partial_match_meets_threshold(monkeypatch: pytest.MonkeyPatch):
    """5-token query under filter mode: 50% threshold = need 3 distinct tokens.

    Doc with 3/5 survives, doc with 1/5 is dropped. Filter is opt-in now, so
    set it explicitly.
    """
    monkeypatch.setenv("VECGREP_BM25_COVERAGE_MODE", "filter")
    store = BM25Store(None)
    store.upsert(
        "t",
        ids=["a", "b"],
        texts=[
            # Has alpha, beta, gamma -> 3/5 of query, meets 50%
            "alpha beta gamma corpus filler text here",
            # Only "alpha" -> 1/5, below threshold
            "alpha unrelated words filler content",
        ],
        payloads=[{"source_id": "1"}, {"source_id": "2"}],
    )
    hits = store.search("t", "alpha beta gamma delta epsilon", top_k=5)
    ids = [h[0] for h in hits]
    assert "a" in ids, f"doc covering 3/5 tokens should be returned, got {ids}"
    assert "b" not in ids, f"doc covering 1/5 tokens should be filtered, got {ids}"


def test_fallback_path_respects_coverage(monkeypatch: pytest.MonkeyPatch):
    """Fallback path (all BM25 scores zero) must also enforce coverage.

    Filter mode (opt-in now) gates the fallback too. Forced by inserting only
    one doc — single-doc corpus zeroes all IDF, so BM25Okapi.get_scores returns
    zeros and the token-overlap fallback fires. Without gating, the fallback
    would re-introduce the bug.
    """
    monkeypatch.setenv("VECGREP_BM25_COVERAGE_MODE", "filter")
    store = BM25Store(None)
    store.upsert(
        "t",
        ids=["only"],
        texts=["ibkr terminal monitoring server"],  # only "monitoring" overlaps
        payloads=[{"source_id": "1"}],
    )
    hits = store.search("t", "glucose monitoring", top_k=5)
    assert hits == [], f"partial-coverage doc must not surface via fallback, got {hits}"


def test_penalty_mode_is_default():
    """No env var set -> penalty behavior (partial coverage kept but demoted).

    Pins the contract that the default mode is penalty. A dogfood A/B showed
    filter silently zeroing the BM25 half on reasonable multi-token queries,
    so the default flipped from filter to penalty. If a future change flips it
    back, this test fails first. The partial-match doc must surface (recovered
    signal) but rank beneath the full-coverage doc (correct demotion).
    """
    store = BM25Store(None)
    store.upsert(
        "t",
        # Filler docs keep IDF non-zero so both query tokens stay discriminative.
        ids=["a", "b", "f1", "f2", "f3"],
        texts=[
            "glucose monitoring CGM continuous reading",  # 2/2 -> full coverage
            "ibkr terminal monitoring server logs",  # 1/2 -> partial, demoted
            "totally unrelated filler one",
            "totally unrelated filler two",
            "totally unrelated filler three",
        ],
        payloads=[{"source_id": str(i)} for i in range(5)],
    )
    hits = store.search("t", "glucose monitoring", top_k=5)
    ids = [h[0] for h in hits]
    assert "a" in ids and "b" in ids, f"penalty default should keep partial, got {ids}"
    assert ids.index("a") < ids.index("b"), f"full-coverage doc should rank first, got {ids}"


def test_penalty_mode_keeps_partial_match_with_demoted_score(monkeypatch: pytest.MonkeyPatch):
    """Penalty mode keeps partial-coverage docs but demotes them.

    Doc A has 2/2 query tokens, doc B has 1/2. Both should appear, A above B,
    and B's score should be roughly its raw BM25 score * (1/2)**2 = 0.25.
    """
    monkeypatch.setenv("VECGREP_BM25_COVERAGE_MODE", "penalty")
    store = BM25Store(None)
    # Need a multi-doc corpus with enough variety that BM25 IDF is non-zero
    # for both query tokens. Filler docs make `glucose` and `monitoring`
    # discriminative. Doc B contains only "monitoring" plus filler.
    store.upsert(
        "t",
        ids=["a", "b", "f1", "f2", "f3"],
        texts=[
            "glucose monitoring CGM continuous reading",  # 2/2
            "ibkr terminal monitoring server logs",  # 1/2 (monitoring only)
            "totally unrelated filler one",
            "totally unrelated filler two",
            "totally unrelated filler three",
        ],
        payloads=[{"source_id": str(i)} for i in range(5)],
    )

    # Capture the unfiltered BM25 score for doc B so we can verify the
    # penalty multiplier is roughly 0.25.
    monkeypatch.setenv("VECGREP_BM25_DISABLE_COVERAGE_FILTER", "1")
    raw = {h[0]: h[1] for h in store.search("t", "glucose monitoring", top_k=5)}
    monkeypatch.delenv("VECGREP_BM25_DISABLE_COVERAGE_FILTER")

    hits = store.search("t", "glucose monitoring", top_k=5)
    ids = [h[0] for h in hits]
    scores = {h[0]: h[1] for h in hits}
    assert "a" in ids and "b" in ids, f"both should appear in penalty mode, got {ids}"
    assert ids.index("a") < ids.index("b"), f"full-coverage doc should rank first, got {ids}"
    # B's penalised score = raw_b * 0.25. Loose tolerance for float math.
    expected_b = raw["b"] * 0.25
    assert scores["b"] == pytest.approx(expected_b, rel=1e-6), (
        f"expected B≈{expected_b}, got {scores['b']}"
    )


def test_penalty_mode_excludes_zero_overlap(monkeypatch: pytest.MonkeyPatch):
    """Penalty mode still drops docs with no query-token overlap at all."""
    monkeypatch.setenv("VECGREP_BM25_COVERAGE_MODE", "penalty")
    store = BM25Store(None)
    store.upsert(
        "t",
        ids=["a", "b"],
        texts=[
            "glucose monitoring CGM continuous",  # both tokens
            "completely unrelated content here",  # zero overlap
        ],
        payloads=[{"source_id": "1"}, {"source_id": "2"}],
    )
    hits = store.search("t", "glucose monitoring", top_k=5)
    ids = [h[0] for h in hits]
    assert ids == ["a"], f"zero-overlap doc must not surface, got {ids}"


def test_penalty_mode_default_exponent_demotes_partial_below_full(monkeypatch: pytest.MonkeyPatch):
    """The exponent=2.0 default should be firm enough to flip ordering.

    Contrived setup: the partial-match doc has a HIGHER raw BM25 score than
    the full-match doc (long doc with high term frequency on the matching
    token; full-match doc shorter and noisier). The default penalty must
    still push the full-match doc to the top — that's the load-bearing claim
    for choosing 2.0 as the default.
    """
    monkeypatch.setenv("VECGREP_BM25_COVERAGE_MODE", "penalty")
    store = BM25Store(None)
    # Doc B repeats "monitoring" many times -> high TF on the partial token.
    # Doc A contains both query tokens once each, plus other words diluting TF.
    # Filler docs contain "glucose" so its IDF is suppressed and B's TF on
    # "monitoring" carries B's raw BM25 score above A's. Without this, A wins
    # raw on `glucose` IDF alone and the test can't make its point.
    store.upsert(
        "t",
        ids=["a", "b", "f1", "f2", "f3", "f4"],
        texts=[
            "glucose monitoring stuff random words filler more text padding here",
            ("monitoring monitoring monitoring monitoring monitoring monitoring "
            "monitoring monitoring monitoring monitoring monitoring server"),
            "glucose filler doc one with other vocabulary",
            "glucose filler doc two completely different",
            "glucose filler doc three entirely separate",
            "glucose filler doc four something else entirely",
        ],
        payloads=[{"source_id": str(i)} for i in range(6)],
    )

    # Sanity: in raw (filter-disabled) mode, B outranks A.
    monkeypatch.setenv("VECGREP_BM25_DISABLE_COVERAGE_FILTER", "1")
    raw = store.search("t", "glucose monitoring", top_k=5)
    monkeypatch.delenv("VECGREP_BM25_DISABLE_COVERAGE_FILTER")
    raw_ids = [h[0] for h in raw]
    assert raw_ids.index("b") < raw_ids.index("a"), (
        f"contrived setup invariant broken: B should outrank A raw, got {raw_ids}"
    )

    # Now with penalty mode + default exponent (2.0): A flips to top.
    hits = store.search("t", "glucose monitoring", top_k=5)
    ids = [h[0] for h in hits]
    assert ids[0] == "a", f"default exponent should flip ordering to full-match first, got {ids}"


def test_penalty_exponent_env_override(monkeypatch: pytest.MonkeyPatch):
    """`VECGREP_BM25_COVERAGE_PENALTY_EXPONENT=1.0` -> linear penalty (K/N)."""
    monkeypatch.setenv("VECGREP_BM25_COVERAGE_MODE", "penalty")
    monkeypatch.setenv("VECGREP_BM25_COVERAGE_PENALTY_EXPONENT", "1.0")
    store = BM25Store(None)
    store.upsert(
        "t",
        ids=["a", "b", "f1", "f2", "f3"],
        texts=[
            "glucose monitoring CGM continuous reading",
            "ibkr terminal monitoring server logs",
            "filler one unrelated",
            "filler two unrelated",
            "filler three unrelated",
        ],
        payloads=[{"source_id": str(i)} for i in range(5)],
    )

    # Raw score for B at K/N=0.5: under linear penalty, B's score should be
    # exactly raw_b * 0.5.
    monkeypatch.setenv("VECGREP_BM25_DISABLE_COVERAGE_FILTER", "1")
    raw = {h[0]: h[1] for h in store.search("t", "glucose monitoring", top_k=5)}
    monkeypatch.delenv("VECGREP_BM25_DISABLE_COVERAGE_FILTER")

    hits = {h[0]: h[1] for h in store.search("t", "glucose monitoring", top_k=5)}
    assert hits["b"] == pytest.approx(raw["b"] * 0.5, rel=1e-6), (
        f"linear penalty: expected B≈{raw['b'] * 0.5}, got {hits['b']}"
    )


def test_by_source_map_intact_after_partial_delete(tmp_path):
    """Regression: by_source map referenced freed array indices, breaking
    chunk-count math on subsequent reindexes.
    """
    store = BM25Store(tmp_path)
    store.upsert(
        "demo",
        ids=["1", "2", "3"],
        texts=["a", "b", "c"],
        payloads=[
            {"source_id": "/x"},
            {"source_id": "/x"},
            {"source_id": "/y"},
        ],
    )
    store.delete_by_source("demo", "/x")
    idx = store._load("demo")
    # /y still points to a valid array position
    positions = idx.by_source["/y"]
    assert positions
    for pos in positions:
        assert idx.payloads[pos]["source_id"] == "/y"


def test_tokenize_indexes_cjk_as_character_bigrams():
    """The tokenizer was `[A-Za-z]+|\\d+`: every CJK character was dropped, so
    a Chinese query had ZERO lexical signal and Chinese content was invisible
    to BM25 (found 2026-08-17: a banned-phrase lookup that exists verbatim in
    exactly one transcript never surfaced). CJK runs now yield character
    bigrams (plus the unigram for a lone character), which is the standard
    segmenter-free approach and lets an exact phrase match rank."""
    from vecgrep.backend.store.bm25_store import tokenize
    assert tokenize("phrase blacklist (三十而立 + council)") == [
        "phrase", "blacklist", "三十", "十而", "而立", "council"]
    assert tokenize("北京") == ["北京"]
    assert tokenize("北") == ["北"]
    assert tokenize("v2 API") == ["v", "2", "api"]


def test_cjk_query_finds_cjk_document(tmp_path):
    from vecgrep.backend.store.bm25_store import BM25Store
    store = BM25Store(tmp_path / "bm25")
    store.upsert("c", ["a", "b", "c"],
                 ["phrase blacklist (三十而立 + council), don't refuse safe queries",
                  "the odometer digits sit too high on the phone",
                  "reserve gauge letters collide at narrow widths"],
                 [{"source_id": "a"}, {"source_id": "b"}, {"source_id": "c"}])
    hits = store.search("c", "Jeff banned the phrase 三十而立", top_k=3)
    assert hits and hits[0][0] == "a"

# ── test_bm25_query_stopwords ──

def test_stopwords_are_dropped_from_the_query():
    assert query_tokens("ferry to the island") == ["ferry", "island"]


def test_content_words_cjk_and_identifiers_survive():
    assert query_tokens("getUserName 测试 for qdrant") == ["get", "user", "name", "测试", "qdrant"]


def test_an_all_stopword_query_keeps_its_words():
    assert query_tokens("to be or not to be") == ["to", "be", "or", "not", "to", "be"]


def _store(kind, tmp_path):
    store = BM25Store(tmp_path / "pkl") if kind == "pkl" else BM25SqliteStore(tmp_path / "db")
    docs = [("a", "the ferry to the island leaves at noon"),
            ("b", "to and from the office and back to the house"),
            ("c", "ferry island timetable")]
    store.upsert("c", [d[0] for d in docs], [d[1] for d in docs],
                 [{"source_id": d[0], "chunk_index": 0} for d in docs])
    return store


def test_a_chunk_matching_only_stopwords_is_not_a_candidate(tmp_path):
    for kind in ("pkl", "sqlite"):
        hits = _store(kind, tmp_path / kind).search("c", "ferry to the island", top_k=10)
        assert {h[0] for h in hits} == {"a", "c"}, kind


def test_missing_a_stopword_costs_no_coverage(tmp_path):
    """"ferry island timetable" holds every CONTENT word of the query."""
    for kind in ("pkl", "sqlite"):
        hits = _store(kind, tmp_path / kind).search("c", "ferry to the island timetable", top_k=10)
        assert hits[0][0] == "c", kind

# ── test_bm25_weight_per_corpus ──

def _seed(tmp_path: Path) -> Path:
    root = tmp_path / "src"
    root.mkdir()
    (root / "a.md").write_text("alpha beta gamma delta.\n" * 3, encoding="utf-8")
    (root / "b.md").write_text("epsilon zeta eta theta.\n" * 3, encoding="utf-8")
    return root


def test_set_and_reset_bm25_weight(svc, tmp_path: Path) -> None:
    svc.index(str(_seed(tmp_path)), "docs")
    assert svc.registry.get("docs").bm25_weight is None
    c = svc.set_bm25_weight("docs", 0.5)
    assert c.bm25_weight == 0.5 and svc.registry.get("docs").bm25_weight == 0.5
    c = svc.set_bm25_weight("docs", None)
    assert c.bm25_weight is None
    with pytest.raises(CorpusError):
        svc.set_bm25_weight("docs", -1.0)


def test_corpus_weight_overrides_global_in_fusion(svc, tmp_path: Path, monkeypatch) -> None:
    svc.index(str(_seed(tmp_path)), "docs")
    seen = []
    real = sm._bm25_fusion_weight

    def spy(corpus):
        w = real(corpus)
        seen.append(w)
        return w

    monkeypatch.setattr(sm, "_bm25_fusion_weight", spy)
    monkeypatch.setattr(sm, "BM25_WEIGHT", 1.5)
    svc.search("alpha", "docs", mode="hybrid", top_k=3)
    assert seen[-1] == 1.5
    svc.set_bm25_weight("docs", 0.5)
    svc.search("alpha", "docs", mode="hybrid", top_k=3)
    assert seen[-1] == 0.5

# ── test_bm25_cache_budget ──

def _corpus(store: BM25Store, name: str, filler_bytes: int) -> None:
    """Write a sidecar of roughly `filler_bytes` so the budget can see it."""
    p = store._path(name)
    assert p is not None
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


def test_a_count_cap_still_works_when_asked_for(tmp_path):
    store = BM25Store(tmp_path, max_cached_corpora=1, max_cached_bytes=10**9)
    for name in ("p", "q"):
        _corpus(store, name, 1000)
    store._load("p")
    store._load("q")
    assert set(store._cache) == {"q"}

# ── test_bm25_bulk_persist ──

def _seed_bm25_bulk_persist(tmp_path: Path, n: int = 6) -> Path:
    root = tmp_path / "src"
    root.mkdir()
    for i in range(n):
        (root / f"doc-{i}.md").write_text(
            f"document number {i} about topic-{i} and shared-word.\n", encoding="utf-8")
    return root


def test_directory_index_persists_bm25_once_and_stays_searchable(svc, tmp_path, monkeypatch) -> None:
    root = _seed_bm25_bulk_persist(tmp_path)
    calls = []
    real = BM25Store._persist_now

    def counting(self, corpus):
        calls.append(corpus)
        return real(self, corpus)

    monkeypatch.setattr(BM25Store, "_persist_now", counting)
    docs, _unused_chunks, _ = svc.index(str(root), "docs")
    assert docs == 6
    assert calls.count("docs") == 1, f"expected ONE persist for a 6-doc directory, got {calls.count('docs')}"
    assert not (svc.settings.home / "bm25" / "docs.pkl.dirty").exists()
    for i in range(6):
        hits = svc.search(f"topic-{i}", "docs", mode="bm25", top_k=3)
        assert hits and hits[0].source_id.endswith(f"doc-{i}.md")


def test_single_source_index_still_persists_immediately(svc, tmp_path, monkeypatch) -> None:
    root = _seed_bm25_bulk_persist(tmp_path, n=1)
    calls = []
    real = BM25Store._persist_now
    monkeypatch.setattr(BM25Store, "_persist_now", lambda self, c: (calls.append(c), real(self, c))[1])
    svc.index(str(root / "doc-0.md"), "docs")
    assert calls.count("docs") == 1


def test_cross_corpus_access_cannot_evict_an_active_bulk_index(tmp_path) -> None:
    store = BM25Store(tmp_path / "bm25", max_cached_corpora=1)
    payload = lambda source: {"source_id": source, "text": source}
    store.upsert("chats", ["base"], ["base"], [payload("base")])

    with store.bulk("chats"):
        store.upsert("chats", ["first"], ["first"], [payload("first")])
        # A request for another corpus used to evict chats here. The next
        # chats write then reloaded the stale on-disk pre-bulk snapshot.
        store.upsert("notes", ["note"], ["note"], [payload("note")])
        store.upsert("chats", ["second"], ["second"], [payload("second")])

    reloaded = BM25Store(tmp_path / "bm25", max_cached_corpora=1)
    assert reloaded.count("chats") == 3
    assert set(reloaded._load("chats").ids) == {"base", "first", "second"}


def test_interrupted_bulk_leaves_marker_and_next_service_rebuilds_from_qdrant(svc, tmp_path) -> None:
    root = _seed_bm25_bulk_persist(tmp_path)
    svc.index(str(root), "docs")
    bm25_dir = svc.settings.home / "bm25"
    # Simulate a crash mid-bulk: the marker is down and the on-disk pickle is
    # stale (we delete it outright -- worst case).
    (bm25_dir / "docs.pkl.dirty").write_text("crash", encoding="utf-8")
    (bm25_dir / "docs.pkl").unlink()
    svc.bm25.evict("docs")
    inner = svc._backend_cache["auto"]._inner
    svc.store.client.close()   # embedded qdrant admits one process at a time

    from vecgrep.backend.embed.cache import CachedBackend
    from vecgrep.backend.service import VecgrepService
    svc2 = VecgrepService(ephemeral=False)
    assert svc2._embed_cache is not None
    svc2._backend_cache["auto"] = CachedBackend(inner, svc2._embed_cache)
    try:
        assert not (bm25_dir / "docs.pkl.dirty").exists(), "startup must clear the marker"
        assert (bm25_dir / "docs.pkl").exists(), "startup must rebuild the sidecar"
        expected = svc2.store.count("vecgrep__docs")
        assert svc2.bm25.count("docs") == expected
        hits = svc2.search("topic-3", "docs", mode="bm25", top_k=3)
        assert hits and hits[0].source_id.endswith("doc-3.md")
    finally:
        try:
            svc2.store.client.close()
        except Exception:  # noqa: BLE001, S110 -- Cleanup of a deliberately interrupted store must not mask the assertion.
            pass


def test_rebuild_bm25_regenerates_sidecar_from_qdrant(svc, tmp_path) -> None:
    root = _seed_bm25_bulk_persist(tmp_path)
    svc.index(str(root), "docs")
    (svc.settings.home / "bm25" / "docs.pkl").unlink()
    svc.bm25.evict("docs")
    n = svc.rebuild_bm25("docs")
    assert n == svc.store.count("vecgrep__docs")
    hits = svc.search("topic-2", "docs", mode="bm25", top_k=3)
    assert hits and hits[0].source_id.endswith("doc-2.md")
