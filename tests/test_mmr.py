"""Phase 1 — source-span dedup / MMR (diversity-aware selection).

The v0.7.0 dedup only collapses same-source chunks whose CHAR SPANS overlap.
The transcript-native duplicate class — the same message body repeated at
different spans (bot alert spam, quoted replies) — survives it and eats top_k
slots (baseline dup_stress redundancy 0.089). Phase 1 adds:

  - clone collapse: same-source chunks with near-identical text collapse even
    when their spans don't overlap;
  - MMR selection: remaining candidates are picked for relevance AND diversity,
    so five slices of one exchange yield one representative plus genuinely
    distinct evidence, not five near-identical hits.
"""
from __future__ import annotations

from vecgrep.backend.assembly import (
    dedup_near_duplicates,
    mmr_select,
    pair_similarity,
)
from vecgrep.backend.service import SearchResult, _dedup_overlapping

ALERT = (
    "ALERT: pipeline health check failed. ERR-9999 on every request. "
    "Investigating the upstream connection pool for the root cause."
)


def _hit(
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
    a = _hit("a.md", 0, 100, 0.9, chunk=ALERT)
    b = _hit("b.md", 0, 100, 0.8, chunk=ALERT)
    assert pair_similarity(a, b) == 0.0


def test_pair_similarity_identical_text_different_spans_is_high() -> None:
    a = _hit("a.md", 0, 100, 0.9, chunk=ALERT)
    b = _hit("a.md", 500, 600, 0.8, chunk=ALERT)
    assert pair_similarity(a, b) >= 0.9


def test_pair_similarity_shared_message_in_different_context_is_moderate() -> None:
    # Same alert line embedded in different surrounding conversation: the
    # shared-shingle signal fires, but it's not a full clone.
    a = _hit("a.md", 0, 200, 0.9, chunk=f"{ALERT} someone: restarting now")
    b = _hit("a.md", 700, 900, 0.8, chunk=f"other person: agreed, wait. {ALERT}")
    sim = pair_similarity(a, b)
    assert 0.6 <= sim < 1.0


def test_pair_similarity_distinct_same_source_text_is_low() -> None:
    a = _hit("a.md", 0, 100, 0.9, chunk="the deploy finished cleanly at noon today")
    b = _hit("a.md", 500, 600, 0.8, chunk="lunch poll results: noodles win once again")
    assert pair_similarity(a, b) < 0.5


def test_pair_similarity_overlapping_spans_dominates() -> None:
    a = _hit("a.md", 0, 100, 0.9, chunk="text one version here")
    b = _hit("a.md", 20, 120, 0.8, chunk="entirely different words")
    assert pair_similarity(a, b) >= 0.8  # 80% span overlap of the shorter


# ── dedup_near_duplicates (the hard collapse) ────────────────────────────────

def test_dedup_collapses_span_overlap_like_before() -> None:
    results = [
        _hit("a.md", 0, 100, 0.4),
        _hit("a.md", 50, 150, 0.9),
    ]
    kept = dedup_near_duplicates(results)
    assert len(kept) == 1 and kept[0].score == 0.9


def test_dedup_collapses_text_clones_at_distant_spans() -> None:
    # The class the old dedup missed: identical message, non-overlapping spans.
    results = [
        _hit("a.md", 0, 130, 0.9, chunk=ALERT),
        _hit("a.md", 800, 930, 0.7, chunk=ALERT),
    ]
    kept = dedup_near_duplicates(results)
    assert len(kept) == 1 and kept[0].score == 0.9


def test_dedup_keeps_trivial_short_texts_apart() -> None:
    # Tiny chunks carry no clone signal — never collapse on text alone.
    results = [
        _hit("a.md", 0, 100, 0.8, chunk="x"),
        _hit("a.md", 200, 300, 0.7, chunk="x"),
    ]
    assert len(dedup_near_duplicates(results)) == 2


def test_dedup_keeps_cross_source_clones() -> None:
    # A quote propagating across channels is distinct evidence, not a dup.
    results = [
        _hit("a.md", 0, 130, 0.9, chunk=ALERT),
        _hit("b.md", 0, 130, 0.8, chunk=ALERT),
    ]
    assert len(dedup_near_duplicates(results)) == 2


def test_legacy_dedup_overlapping_alias_still_works() -> None:
    # Backcompat: service._dedup_overlapping remains importable + functional.
    results = [
        _hit("a.md", 0, 100, 0.4),
        _hit("a.md", 50, 150, 0.9),
    ]
    assert len(_dedup_overlapping(results)) == 1


# ── mmr_select (diversity-aware selection) ───────────────────────────────────

def test_mmr_five_slices_collapse_to_diverse_representatives() -> None:
    """The brief's done-when: 5 near-identical slices of one exchange plus
    distinct sub-events. Selection must return ONE slice representative and
    fill the rest with the distinct sub-events, even though the slices
    outscore them."""
    slices = [
        _hit("a.md", i * 60, i * 60 + 130, 0.95 - i * 0.01, chunk=ALERT)
        for i in range(5)
    ]
    sub_events = [
        _hit("a.md", 2000, 2100, 0.60, chunk="escalation: paging the on-call, this is now a sev-2"),
        _hit("a.md", 3000, 3100, 0.55, chunk="postmortem scheduled for thursday, notes doc is up"),
        _hit("a.md", 4000, 4100, 0.50, chunk="resolution confirmed, cert renewed and pool recycled"),
    ]
    picked = mmr_select(slices + sub_events, k=4)
    texts = [p.chunk for p in picked]
    assert texts.count(ALERT) == 1, "exactly one slice representative"
    for ev in sub_events:
        assert ev.chunk in texts, f"lost distinct evidence: {ev.chunk[:40]}"


def test_mmr_no_duplicates_degrades_to_score_order() -> None:
    distinct = [
        _hit("a.md", 0, 100, 0.9, chunk="first topic entirely about apples and orchards"),
        _hit("b.md", 0, 100, 0.8, chunk="second topic entirely about bridges and rivets"),
        _hit("c.md", 0, 100, 0.7, chunk="third topic entirely about compilers and lexers"),
    ]
    picked = mmr_select(distinct, k=3)
    assert [p.score for p in picked] == [0.9, 0.8, 0.7]


def test_mmr_k_larger_than_pool_returns_all() -> None:
    pool = [_hit("a.md", 0, 100, 0.9), _hit("b.md", 0, 100, 0.8)]
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
