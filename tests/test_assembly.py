from __future__ import annotations

import pytest

from vecgrep.backend.assembly import (
    ResultStub,
    dedup_near_duplicates,
    estimate_tokens,
    mmr_select,
    pair_similarity,
    split_full_and_stubs,
)
from vecgrep.backend.service import SearchResult, _dedup_overlapping

# ── test_dedup ──

def _hit(source_id: str, start: int, end: int, score: float, corpus: str = "c") -> SearchResult:
    return SearchResult(
        score=score,
        similarity_pct=score * 100,
        chunk="x",
        chunk_start=start,
        chunk_end=end,
        context_before="",
        context_after="",
        source_id=source_id,
        corpus=corpus,
        metadata={},
        chunk_id=f"{source_id}:{start}",
        matched_by=["vector"],
    )


def test_overlapping_same_source_collapses_keeping_higher_score() -> None:
    results = [
        _hit("a.md", 0, 100, 0.4),
        _hit("a.md", 50, 150, 0.9),  # overlaps the first by 50/100 = 50%
    ]
    kept = _dedup_overlapping(results)
    assert len(kept) == 1
    assert kept[0].score == 0.9  # higher-scoring of the pair survives


@pytest.mark.parametrize("hits", [
    pytest.param([_hit("a.md", 0, 100, 0.8), _hit("a.md", 200, 300, 0.7)],
                 id="same-source-no-overlap"),
    pytest.param([_hit("a.md", 0, 100, 0.8), _hit("b.md", 0, 100, 0.7)],
                 id="different-sources-identical-span"),
    pytest.param([_hit("a.md", 0, 100, 0.8, corpus="c1"), _hit("a.md", 0, 100, 0.7, corpus="c2")],
                 id="different-corpus-same-span"),
    # 10-char overlap on 100-char chunks = 10% < 50% threshold.
    pytest.param([_hit("a.md", 0, 100, 0.8), _hit("a.md", 90, 190, 0.7)],
                 id="minor-overlap-below-threshold"),
])
def test_distinct_hits_are_both_kept(hits) -> None:
    assert len(_dedup_overlapping(hits)) == 2


def test_search_dedups_overlapping_windows(svc, make_doc) -> None:
    """End-to-end: a doc whose overlapping windows all match shouldn't return
    the same passage multiple times in top_k."""
    p = make_doc(
        "d.md",
        "Quantum widgets hum. Quantum widgets hum loudly. Quantum widgets hum "
        "loudly today. Quantum widgets hum loudly today and tomorrow.",
    )
    svc.index(str(p), "c")
    hits = svc.search("quantum widgets hum", "c", top_k=10)
    # Every returned hit should be a distinct (non-overlapping) span.
    spans = [(h.source_id, h.chunk_start, h.chunk_end) for h in hits]
    assert len(spans) == len(set(spans)), "duplicate spans returned"

# ── test_mmr ──

ALERT = (
    "ALERT: pipeline health check failed. ERR-9999 on every request. "
    "Investigating the upstream connection pool for the root cause."
)


def _hit_mmr(
    source_id: str,
    start: int,
    end: int,
    score: float,
    chunk: str = "x",
    corpus: str = "c",
) -> SearchResult:
    return SearchResult(
        score=score,
        similarity_pct=score * 100,
        chunk=chunk,
        chunk_start=start,
        chunk_end=end,
        context_before="",
        context_after="",
        source_id=source_id,
        corpus=corpus,
        metadata={},
        chunk_id=f"{source_id}:{start}",
        matched_by=["vector"],
    )


# ── pair_similarity ──────────────────────────────────────────────────────────

def test_pair_similarity_cross_source_is_zero() -> None:
    a = _hit_mmr("a.md", 0, 100, 0.9, chunk=ALERT)
    b = _hit_mmr("b.md", 0, 100, 0.8, chunk=ALERT)
    assert pair_similarity(a, b) == 0.0


def test_pair_similarity_identical_text_different_spans_is_high() -> None:
    a = _hit_mmr("a.md", 0, 100, 0.9, chunk=ALERT)
    b = _hit_mmr("a.md", 500, 600, 0.8, chunk=ALERT)
    assert pair_similarity(a, b) >= 0.9


def test_pair_similarity_shared_message_in_different_context_is_moderate() -> None:
    # Same alert line embedded in different surrounding conversation: the
    # shared-shingle signal fires, but it's not a full clone.
    a = _hit_mmr("a.md", 0, 200, 0.9, chunk=f"{ALERT} someone: restarting now")
    b = _hit_mmr("a.md", 700, 900, 0.8, chunk=f"other person: agreed, wait. {ALERT}")
    sim = pair_similarity(a, b)
    assert 0.6 <= sim < 1.0


def test_pair_similarity_distinct_same_source_text_is_low() -> None:
    a = _hit_mmr("a.md", 0, 100, 0.9, chunk="the deploy finished cleanly at noon today")
    b = _hit_mmr("a.md", 500, 600, 0.8, chunk="lunch poll results: noodles win once again")
    assert pair_similarity(a, b) < 0.5


def test_pair_similarity_overlapping_spans_dominates() -> None:
    a = _hit_mmr("a.md", 0, 100, 0.9, chunk="text one version here")
    b = _hit_mmr("a.md", 20, 120, 0.8, chunk="entirely different words")
    assert pair_similarity(a, b) >= 0.8  # 80% span overlap of the shorter


# ── dedup_near_duplicates (the hard collapse) ────────────────────────────────

def test_dedup_collapses_span_overlap_like_before() -> None:
    results = [
        _hit_mmr("a.md", 0, 100, 0.4),
        _hit_mmr("a.md", 50, 150, 0.9),
    ]
    kept = dedup_near_duplicates(results)
    assert len(kept) == 1 and kept[0].score == 0.9


def test_dedup_collapses_text_clones_at_distant_spans() -> None:
    # The class the old dedup missed: identical message, non-overlapping spans.
    results = [
        _hit_mmr("a.md", 0, 130, 0.9, chunk=ALERT),
        _hit_mmr("a.md", 800, 930, 0.7, chunk=ALERT),
    ]
    kept = dedup_near_duplicates(results)
    assert len(kept) == 1 and kept[0].score == 0.9


def test_dedup_keeps_trivial_short_texts_apart() -> None:
    # Tiny chunks carry no clone signal — never collapse on text alone.
    results = [
        _hit_mmr("a.md", 0, 100, 0.8, chunk="x"),
        _hit_mmr("a.md", 200, 300, 0.7, chunk="x"),
    ]
    assert len(dedup_near_duplicates(results)) == 2


def test_dedup_keeps_cross_source_clones() -> None:
    # A quote propagating across channels is distinct evidence, not a dup.
    results = [
        _hit_mmr("a.md", 0, 130, 0.9, chunk=ALERT),
        _hit_mmr("b.md", 0, 130, 0.8, chunk=ALERT),
    ]
    assert len(dedup_near_duplicates(results)) == 2


# ── mmr_select (diversity-aware selection) ───────────────────────────────────

def test_mmr_five_slices_collapse_to_diverse_representatives() -> None:
    """The brief's done-when: 5 near-identical slices of one exchange plus
    distinct sub-events. Selection must return ONE slice representative and
    fill the rest with the distinct sub-events, even though the slices
    outscore them."""
    slices = [
        _hit_mmr("a.md", i * 60, i * 60 + 130, 0.95 - i * 0.01, chunk=ALERT)
        for i in range(5)
    ]
    sub_events = [
        _hit_mmr("a.md", 2000, 2100, 0.60, chunk="escalation: paging the on-call, this is now a sev-2"),
        _hit_mmr("a.md", 3000, 3100, 0.55, chunk="postmortem scheduled for thursday, notes doc is up"),
        _hit_mmr("a.md", 4000, 4100, 0.50, chunk="resolution confirmed, cert renewed and pool recycled"),
    ]
    picked = mmr_select(slices + sub_events, k=4)
    texts = [p.chunk for p in picked]
    assert texts.count(ALERT) == 1, "exactly one slice representative"
    for ev in sub_events:
        assert ev.chunk in texts, f"lost distinct evidence: {ev.chunk[:40]}"


def test_mmr_no_duplicates_degrades_to_score_order() -> None:
    distinct = [
        _hit_mmr("a.md", 0, 100, 0.9, chunk="first topic entirely about apples and orchards"),
        _hit_mmr("b.md", 0, 100, 0.8, chunk="second topic entirely about bridges and rivets"),
        _hit_mmr("c.md", 0, 100, 0.7, chunk="third topic entirely about compilers and lexers"),
    ]
    picked = mmr_select(distinct, k=3)
    assert [p.score for p in picked] == [0.9, 0.8, 0.7]


def test_mmr_k_larger_than_pool_returns_all() -> None:
    pool = [_hit_mmr("a.md", 0, 100, 0.9), _hit_mmr("b.md", 0, 100, 0.8)]
    assert len(mmr_select(pool, k=10)) == 2


def test_mmr_empty_pool() -> None:
    assert mmr_select([], k=5) == []


# ── end-to-end: the eval fixtures improve ────────────────────────────────────

def test_dup_stress_redundancy_eliminated(svc) -> None:
    """Baseline dup_stress redundancy was 0.089 (bot alert spam surviving).
    With clone collapse + MMR the top-10 must be effectively dup-free."""
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    report = eh.run_eval(svc)
    assert report["queries"]["dup_stress"]["redundancy"] <= 0.023, (
        "duplicate slices still eating top_k slots"
    )


def test_no_loss_of_distinct_evidence(svc) -> None:
    """Diversity must not cost recall: every graded golden query keeps
    expected_recall at its baseline level (all 1.0 except exact_keyword's
    0.667, which may only improve)."""
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    report = eh.run_eval(svc)
    for q in eh.GOLDEN_QUERIES:
        m = report["queries"][q.id]
        floor = 0.6667 if q.id == "exact_keyword" else 1.0
        assert m["expected_recall"] >= floor, f"{q.id}: recall regressed"

# ── test_budget ──

LONG_CHUNK = (
    "this chunk is long enough to cost a meaningful number of tokens in the "
    "stub budget accounting so ceilings actually bite during the test run"
)


def _hit_budget(i: int, score: float = 0.5, chunk: str = LONG_CHUNK) -> SearchResult:
    return SearchResult(
        score=score,
        similarity_pct=score * 100,
        chunk=f"[{i}] {chunk}",
        chunk_start=i * 200,
        chunk_end=i * 200 + 150,
        context_before="ctx-before " * 30,
        context_after="ctx-after " * 30,
        source_id=f"channel-x/2026-01-{(i % 28) + 1:02d}.md",
        corpus="c",
        metadata={},
        chunk_id=f"cid-{i}",
        matched_by=["vector"],
        doc_timestamp=1_700_000_000.0 + i,
        explain={"vector_cosine": score, "vector_rank": i + 1},
    )


# ── token estimation ─────────────────────────────────────────────────────────

def test_estimate_tokens_scales_with_length() -> None:
    assert estimate_tokens("") == 0
    short, long = estimate_tokens("word " * 10), estimate_tokens("word " * 1000)
    assert 0 < short < long
    # ~4 chars/token heuristic: 5000 chars ≈ 1250 tokens, sanity band only.
    assert 800 < long < 2000


# ── split_full_and_stubs ─────────────────────────────────────────────────────

def test_split_gives_full_then_stubs_capped_at_max_total() -> None:
    results = [_hit_budget(i, score=1.0 - i * 0.01) for i in range(100)]
    full, stubs = split_full_and_stubs(results, full_k=8, max_total=80)
    assert len(full) == 8
    assert len(stubs) <= 72
    assert len(full) + len(stubs) <= 80
    # Order preserved: full tier is the head of the ranking.
    assert [f.chunk_id for f in full] == [r.chunk_id for r in results[:8]]


def test_split_stub_shape() -> None:
    results = [_hit_budget(i) for i in range(12)]
    _full, stubs = split_full_and_stubs(results, full_k=8, max_total=80)
    s: ResultStub = stubs[0]
    assert s.chunk_id == "cid-8"
    assert s.source_id.endswith(".md")
    assert s.doc_timestamp is not None
    assert "\n" not in s.snippet, "stub snippet must be one line"
    assert len(s.snippet) <= 200
    assert s.score > 0 and s.similarity_pct > 0
    assert s.relevance_pct == s.similarity_pct
    assert s.relevance_label == "related"
    assert s.matched_by == ("vector",)
    assert s.scores == {"vector_cosine": 0.5, "vector_rank": 9}
    # Stubs must NOT carry context windows — that's the whole point.
    assert not hasattr(s, "context_before")


def test_split_token_ceiling_bites() -> None:
    results = [_hit_budget(i) for i in range(80)]
    _f_uncapped, stubs_uncapped = split_full_and_stubs(
        results, full_k=8, max_total=80, token_ceiling=10_000_000
    )
    _f_capped, stubs_capped = split_full_and_stubs(
        results, full_k=8, max_total=80, token_ceiling=200
    )
    assert len(stubs_capped) < len(stubs_uncapped)
    assert stubs_capped, "a small ceiling should still emit some stubs"
    total = sum(estimate_tokens(s.snippet + s.source_id) for s in stubs_capped)
    assert total <= 200 + estimate_tokens(
        stubs_capped[-1].snippet + stubs_capped[-1].source_id
    ), "ceiling may be crossed by at most the final stub"


def test_split_fewer_results_than_full_k() -> None:
    results = [_hit_budget(i) for i in range(3)]
    full, stubs = split_full_and_stubs(results, full_k=8, max_total=80)
    assert len(full) == 3 and stubs == []


def test_split_empty() -> None:
    assert split_full_and_stubs([], full_k=8, max_total=80) == ([], [])


# ── doc_timestamp now rides on SearchResult ──────────────────────────────────

def test_search_results_carry_doc_timestamp(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    hits = svc.search("relay-service", "evalchat", top_k=5)
    assert hits, "fixtures must match"
    dated = [h for h in hits if h.doc_timestamp is not None]
    assert dated, "frontmatter date: must populate doc_timestamp"
    # 2026-01-xx fixture dates land in a sane epoch range.
    assert all(1_700_000_000 < h.doc_timestamp < 1_900_000_000 for h in dated)


# ── service.search_budgeted end-to-end ───────────────────────────────────────

def test_search_budgeted_returns_full_and_stub_tiers(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    full, stubs = svc.search_budgeted("relay-service ERR-0451", "evalchat", full_k=8)
    assert 0 < len(full) <= 8
    assert isinstance(stubs, list)
    if stubs:
        assert isinstance(stubs[0], ResultStub)
        # The stub tail continues the ranking below the full tier.
        assert stubs[0].score <= full[-1].score
    # 55 fixture chunks exist; breadth means we surface well beyond top-10.
    assert len(full) + len(stubs) > 10


def test_search_budgeted_widens_candidate_pool(svc, monkeypatch) -> None:
    """A budgeted call must not be silently capped by CANDIDATE_POOL=50 when
    max_total exceeds it: the per-corpus pool must widen to top_k. (Retrieval
    still only returns what actually MATCHES — the widened pool is a ceiling,
    not a corpus dump — so this pins the mechanism, not result counts.)"""
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    seen: list[int] = []
    orig = type(svc)._search_one

    def spy(
        self, corpus, query, top_k, mode, explain=False, query_vectors=None,
        timer=None,
    ):
        seen.append(top_k)
        return orig(
            self, corpus, query, top_k, mode, explain=explain,
            query_vectors=query_vectors, timer=timer,
        )

    monkeypatch.setattr(type(svc), "_search_one", spy)
    svc.search_budgeted("relay-service", "evalchat", max_total=80)
    assert seen and seen[0] >= 80, f"pool not widened: {seen}"
    seen.clear()
    svc.search("relay-service", "evalchat", top_k=5)
    assert seen and seen[0] == 50, "normal searches keep the standard pool"


def test_search_budgeted_stub_chunk_ids_resolve(svc) -> None:
    """Every stub's chunk_id must round-trip through get_chunk_window — the
    re-query path an assistant uses to expand a stub."""
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    _full, stubs = svc.search_budgeted("relay-service ERR-0451", "evalchat")
    assert stubs
    win = svc.get_chunk_window("evalchat", stubs[0].chunk_id, 400)
    assert win and (win.get("chunk") or win.get("text"))


# ── web UI default ───────────────────────────────────────────────────────────



def test_default_budget_is_10_full_100_total():
    """The default is 10 full + up to 90 stubs. max_total stays a CAP
    — token_ceiling is the real limit, so a dense set legitimately yields
    fewer stubs. These defaults are what the API/CLI/MCP inherit."""
    from vecgrep.backend.assembly import DEFAULT_FULL_K, DEFAULT_MAX_TOTAL
    assert DEFAULT_FULL_K == 10
    assert DEFAULT_MAX_TOTAL == 100


def test_default_split_head_is_10():
    results = [_hit_budget(i, score=1.0 - i * 0.01) for i in range(30)]
    full, stubs = split_full_and_stubs(results)  # defaults
    assert len(full) == 10
    assert len(full) + len(stubs) <= 100


# ── enrichment lands in indexed payloads (F-1..F-3 integration) ─────────────

def test_indexed_chunks_carry_enrichment_and_filter(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    hits = svc.search("relay-service", "evalchat", top_k=10)
    assert hits
    enriched = [h for h in hits if isinstance(h.metadata.get("speakers"), list)]
    assert enriched, "index path must attach chunk enrichment metadata"

    by_alice = svc.search("relay-service", "evalchat", top_k=10,
                          filters=["speaker:alice"])
    assert by_alice, "alice speaks in the fixtures"
    assert all("alice" in [s.lower().removesuffix(" [bot]")
                           for s in h.metadata.get("speakers", [])]
               for h in by_alice)

    bots_only = svc.search("relay-service", "evalchat", top_k=10,
                           filters=["bot:true"])
    assert bots_only, "ops-bot [bot] speaks in the fixtures"
    no_alice = svc.search("relay-service", "evalchat", top_k=10,
                          filters=["-speaker:alice"])
    assert all(
        "alice" not in [str(s).lower() for s in h.metadata.get("speakers", [])]
        for h in no_alice)
