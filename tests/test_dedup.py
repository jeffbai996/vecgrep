"""Post-retrieval dedup of overlapping chunks (item 5).

Overlapping sentence-windows from one source shouldn't each consume a top_k
slot. _dedup_overlapping collapses same-source hits whose char ranges overlap,
keeping the higher-scoring one.
"""
from __future__ import annotations

import pytest

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
