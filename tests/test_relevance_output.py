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


# ── lexical-only hits are labelled, not bucketed ─────────────────────────────
#
# A BM25-only hit's percentage is rank-relative *within one corpus*: it is
# rescaled so the best BM25 score for this query reads BM25_DISPLAY_TOP (~90).
# The top hit of every corpus therefore reads ~90 whether its absolute BM25
# score is 7.5 or 0.067. Bucketing that number as "strong" hands a consumer
# semantic confidence the dense channel never supplied.

def _hit_mb(pct: float, matched_by: list[str]) -> SearchResult:
    r = _hit(pct)
    r.matched_by = matched_by
    return r


def test_bm25_only_hit_is_labelled_lexical_only() -> None:
    assert _hit_mb(90.0, ["bm25"]).relevance_label == "lexical-only"
    assert _hit_mb(25.0, ["bm25"]).relevance_label == "lexical-only"
    # rerank riding along doesn't make the dense channel speak
    assert _hit_mb(96.0, ["bm25", "rerank"]).relevance_label == "lexical-only"


def test_vector_corroborated_hits_keep_their_bucket() -> None:
    assert _hit_mb(96.0, ["vector"]).relevance_label == "exact"
    assert _hit_mb(80.0, ["vector", "bm25"]).relevance_label == "strong"
    assert _hit_mb(50.0, ["vector", "bm25", "rerank"]).relevance_label == "related"
    assert _hit_mb(10.0, ["vector"]).relevance_label == "weak"


# ── fusion display: the calibrated cosine is the number, not max() ───────────

def test_hybrid_pct_is_the_cosine_not_the_lexical_rank(svc, make_doc, monkeypatch):
    """A chunk both retrievers found must display its absolute semantic score.

    Regression: display used max(cos_pct, bm_pct). bm_pct for the rank-1
    lexical hit is always ~BM25_DISPLAY_TOP, so max() overwrote a weak
    semantic verdict — the same chunk read 90.0 "strong" in hybrid mode and
    25.8 "weak" in vector mode on the same query.
    """
    from vecgrep.backend.service import (
        _cosine_floor,
        _cosine_to_pct,
        _hit_payload,
        _id_for,
    )
    from vecgrep.backend.store.qdrant_store import StoredHit

    p = make_doc("doc.md", "alpha content here. beta content here.")
    svc.index(str(p), "test")
    corpus = svc.list_corpora()[0]

    # Clears the pre-fusion floor, but is only mediocre semantically.
    cos = _cosine_floor(corpus.embed_model) + 0.02
    text = "alpha content here"
    hit = StoredHit(
        score=cos, chunk_text=text, source_id="s0", source_text=text,
        chunk_start=0, chunk_end=len(text), chunk_index=0, metadata={},
        corpus="test",
    )
    cid = _id_for(hit)

    monkeypatch.setattr(svc.store, "search", lambda *a, **k: [hit])
    monkeypatch.setattr(svc.store, "get_many_by_id", lambda *a, **k: {})
    # Same chunk, rank 1 lexically — the case max() used to inflate to ~90.
    monkeypatch.setattr(
        svc.bm25, "search", lambda *a, **k: [(cid, 7.5, _hit_payload(hit))]
    )

    (r,) = svc.search("alpha", "test", mode="hybrid", top_k=1)
    assert set(r.matched_by) >= {"vector", "bm25"}
    expected = _cosine_to_pct(cos, model=corpus.embed_model)
    assert abs(r.relevance_pct - expected) < 1e-6, (
        f"lexical rank leaked into the display %: {r.relevance_pct} != {expected}"
    )
    assert r.relevance_pct < 90.0


def test_bm25_only_hit_still_gets_rank_relative_pct(svc, make_doc, monkeypatch):
    """The BM25-only fallback must survive — a keyword hit the dense channel
    never saw should not read as the raw RRF score (~1.6%)."""
    from vecgrep.backend.service import BM25_DISPLAY_TOP, _hit_payload, _id_for
    from vecgrep.backend.store.qdrant_store import StoredHit

    p = make_doc("doc.md", "alpha content here. beta content here.")
    svc.index(str(p), "test")

    text = "alpha content here"
    hit = StoredHit(
        score=0.9, chunk_text=text, source_id="s0", source_text=text,
        chunk_start=0, chunk_end=len(text), chunk_index=0, metadata={},
        corpus="test",
    )
    monkeypatch.setattr(svc.store, "search", lambda *a, **k: [])
    monkeypatch.setattr(svc.store, "get_many_by_id", lambda *a, **k: {})
    monkeypatch.setattr(
        svc.bm25, "search", lambda *a, **k: [(_id_for(hit), 7.5, _hit_payload(hit))]
    )

    (r,) = svc.search("alpha", "test", mode="hybrid", top_k=1)
    assert r.matched_by == ["bm25"]
    assert abs(r.relevance_pct - BM25_DISPLAY_TOP) < 1e-6
    # ...but it must announce that the dense channel never corroborated it.
    assert r.relevance_label == "lexical-only"
