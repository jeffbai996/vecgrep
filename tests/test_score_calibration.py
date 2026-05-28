"""Unified relevance-score calibration (item 2).

When reranking is on (the default), every hit's similarity_pct comes from ONE
formula: a calibrated sigmoid over the cross-encoder score. This is the single
canonical relevance number, computed identically regardless of which retriever
found the hit. These tests pin the calibration shape and the wiring, without
loading torch (the cross-encoder is monkeypatched).
"""
from __future__ import annotations

from vecgrep.backend import service as svc_mod
from vecgrep.backend.service import _rerank_to_pct


def test_rerank_pct_calibration_shape() -> None:
    # Noise floor (~0.51) reads well below 30%.
    assert _rerank_to_pct(0.51) < 30
    # Uncertain boundary at the center maps to 50%.
    assert abs(_rerank_to_pct(0.57) - 50.0) < 0.01
    # Strong hits (>=0.66) read above 70%.
    assert _rerank_to_pct(0.66) > 70
    # Monotonic.
    vals = [_rerank_to_pct(p) for p in (0.40, 0.50, 0.57, 0.65, 0.75)]
    assert vals == sorted(vals)
    # Bounded.
    assert _rerank_to_pct(0.0) >= 0.0
    assert _rerank_to_pct(1.0) <= 100.0


def test_rerank_sets_unified_display_pct(svc, make_doc, monkeypatch) -> None:
    """With rerank on, similarity_pct is derived from the rerank score via the
    one calibration — not the per-retriever cosine/BM25 mix."""
    p = make_doc(
        "d.md",
        "Alpha topic sentence. Beta topic sentence. Gamma topic sentence.",
    )
    svc.index(str(p), "c")

    # Stub the cross-encoder: assign a high score to the first candidate, low
    # to the rest, so we can predict the resulting display pcts exactly.
    def fake_rerank(query, candidates, model_name):
        scored = []
        for i, (_text, payload) in enumerate(candidates):
            score = 0.70 if i == 0 else 0.50
            scored.append((score, payload))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    monkeypatch.setattr(svc_mod, "rerank", fake_rerank, raising=False)
    # _apply_rerank imports rerank lazily from .rerank; patch there too.
    import vecgrep.backend.rerank as rr_mod
    monkeypatch.setattr(rr_mod, "rerank", fake_rerank, raising=False)

    hits = svc.search("alpha beta gamma", "c", top_k=5, rerank=True)
    assert hits
    top = hits[0]
    # 0.70 -> well above 70%; the display pct must match the rerank calibration,
    # not a cosine-derived number.
    assert abs(top.similarity_pct - _rerank_to_pct(0.70)) < 0.01
    assert top.similarity_pct > 70
    assert "rerank" in top.matched_by
    assert "rerank_score" in top.explain
