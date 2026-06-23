"""Vector noise floor: sub-noise vector hits are dropped before fusion."""
from __future__ import annotations

from vecgrep.backend.service import COSINE_FLOOR_MARGIN, _calibration_for, _cosine_floor
from vecgrep.backend.store.qdrant_store import StoredHit


def test_floor_is_center_minus_margin():
    for model in ("bge-m3", "nomic-embed-text", "mxbai-embed-large", None, "unknown"):
        center, _ = _calibration_for(model)
        assert abs(_cosine_floor(model) - max(0.0, center - COSINE_FLOOR_MARGIN)) < 1e-9


def test_floor_never_negative():
    # A huge margin disables the floor (clamped to 0, no filtering).
    import vecgrep.backend.service as S
    saved = S.COSINE_FLOOR_MARGIN
    try:
        S.COSINE_FLOOR_MARGIN = 5.0
        assert S._cosine_floor("bge-m3") == 0.0
    finally:
        S.COSINE_FLOOR_MARGIN = saved


def _hit(score: float, text: str, idx: int) -> StoredHit:
    return StoredHit(
        score=score, chunk_text=text, source_id=f"s{idx}", source_text=text,
        chunk_start=0, chunk_end=len(text), chunk_index=idx, metadata={},
        corpus="test",
    )


def test_subnoise_vector_hits_dropped_before_results(svc, make_doc, monkeypatch):
    p = make_doc("doc.md", "alpha content here. beta content here.")
    svc.index(str(p), "test")
    corpus = svc.list_corpora()[0]
    floor = _cosine_floor(corpus.embed_model)

    strong = _hit(min(1.0, floor + 0.2), "strong real hit", 0)
    junk = _hit(max(0.0, floor - 0.2), "subnoise junk hit", 1)
    # Return both regardless of the query vector; the floor must drop `junk`.
    monkeypatch.setattr(svc.store, "search", lambda *a, **k: [strong, junk])

    hits = svc.search("anything", "test", mode="vector", top_k=10)
    texts = " ".join(h.chunk for h in hits)
    assert "strong real hit" in texts
    assert "subnoise junk hit" not in texts, "a sub-floor vector hit leaked through"
