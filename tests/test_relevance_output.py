"""Phase 4 — clearer score output.

Three changes, all display-side (ranking untouched):

1. `relevance_pct` — the going-forward name for the calibrated display
   percent; `similarity_pct` stays as a compatibility alias (same value).
2. `relevance_label` — a qualitative bucket (exact / strong / related /
   weak) so a caller doesn't have to interpret percentages.
3. The rerank display sigmoid is flattened (slope 35 → 18) so the top of
   the range stops bunching at 99.x: strong-vs-stronger hits now read
   ~91 vs ~96 instead of 99.2 vs 99.6.
"""
from __future__ import annotations

from vecgrep.backend.service import SearchResult, _rerank_to_pct


def _hit(pct: float) -> SearchResult:
    return SearchResult(
        score=0.5,
        similarity_pct=pct,
        chunk="text",
        chunk_start=0,
        chunk_end=100,
        context_before="",
        context_after="",
        source_id="a.md",
        corpus="c",
        metadata={},
        chunk_id="cid",
        matched_by=["vector"],
    )


# ── relevance_pct alias ──────────────────────────────────────────────────────

def test_relevance_pct_mirrors_similarity_pct() -> None:
    r = _hit(87.3)
    assert r.relevance_pct == r.similarity_pct == 87.3
    r.similarity_pct = 12.0
    assert r.relevance_pct == 12.0  # alias, not a snapshot


# ── relevance_label buckets ──────────────────────────────────────────────────

def test_relevance_labels() -> None:
    assert _hit(98.0).relevance_label == "exact"
    assert _hit(95.0).relevance_label == "exact"
    assert _hit(94.9).relevance_label == "strong"
    assert _hit(75.0).relevance_label == "strong"
    assert _hit(74.9).relevance_label == "related"
    assert _hit(40.0).relevance_label == "related"
    assert _hit(39.9).relevance_label == "weak"
    assert _hit(5.0).relevance_label == "weak"


# ── rerank display spread (the 99.x de-cluster) ──────────────────────────────

def test_rerank_display_spreads_strong_hits() -> None:
    """Real strong rerank scores (0.66–0.75) must span a visible range, not
    all read 96–99.9. Under the old slope-35 sigmoid this span was ~4 points;
    it must now be at least 10."""
    lo, hi = _rerank_to_pct(0.66), _rerank_to_pct(0.75)
    assert hi - lo >= 10.0
    # And 0.70 vs 0.73 (the "99.2 vs 99.6" complaint) now differ visibly.
    assert _rerank_to_pct(0.73) - _rerank_to_pct(0.70) >= 2.0


def test_rerank_calibration_contract_still_holds() -> None:
    # The documented anchor points survive the flattening.
    assert _rerank_to_pct(0.51) < 30       # noise
    assert abs(_rerank_to_pct(0.57) - 50.0) < 0.01  # boundary
    assert _rerank_to_pct(0.66) > 70       # strong


# ── surfaces carry the new fields ────────────────────────────────────────────

def test_api_hit_carries_relevance_fields() -> None:
    from vecgrep.backend.api.schemas import SearchHit

    h = SearchHit(
        similarity_pct=96.0,
        chunk="x",
        context_before="",
        context_after="",
        source_id="a.md",
        corpus="c",
        metadata={},
        relevance_pct=96.0,
        relevance_label="exact",
    )
    assert h.relevance_pct == 96.0 and h.relevance_label == "exact"


def test_mcp_payload_carries_relevance_fields() -> None:
    from vecgrep.mcp.server import _result_payload

    p = _result_payload(_hit(82.5))
    assert p["relevance_pct"] == 82.5
    assert p["relevance_label"] == "strong"
    assert p["similarity_pct"] == 82.5  # compat alias stays
