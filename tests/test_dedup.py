"""Post-retrieval dedup of overlapping chunks (item 5).

Overlapping sentence-windows from one source shouldn't each consume a top_k
slot. _dedup_overlapping collapses same-source hits whose char ranges overlap,
keeping the higher-scoring one.
"""
from __future__ import annotations

from vecgrep.backend.service import SearchResult, _dedup_overlapping


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


def test_non_overlapping_same_source_both_kept() -> None:
    results = [
        _hit("a.md", 0, 100, 0.8),
        _hit("a.md", 200, 300, 0.7),  # no overlap
    ]
    assert len(_dedup_overlapping(results)) == 2


def test_different_sources_never_collide() -> None:
    results = [
        _hit("a.md", 0, 100, 0.8),
        _hit("b.md", 0, 100, 0.7),  # identical span, different source
    ]
    assert len(_dedup_overlapping(results)) == 2


def test_different_corpus_same_span_kept() -> None:
    results = [
        _hit("a.md", 0, 100, 0.8, corpus="c1"),
        _hit("a.md", 0, 100, 0.7, corpus="c2"),
    ]
    assert len(_dedup_overlapping(results)) == 2


def test_minor_overlap_below_threshold_kept() -> None:
    # 10-char overlap on 100-char chunks = 10% < 50% threshold -> both kept.
    results = [
        _hit("a.md", 0, 100, 0.8),
        _hit("a.md", 90, 190, 0.7),
    ]
    assert len(_dedup_overlapping(results)) == 2


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
