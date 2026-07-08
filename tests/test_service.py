"""Service-layer tests — the layer where every bug we've shipped lived.

Covers:
- index() round-trip + chunk_count accuracy across re-index
- incremental skip via content hash
- hybrid / vector / bm25 modes return ranked SearchResult
- filters narrow results post-retrieval
- explain payload populates per-retriever fields
- delete_corpus clears qdrant + bm25 + registry
"""
from __future__ import annotations

from tests.conftest import StubEmbed


def test_index_then_search_round_trip(svc, make_doc):
    p = make_doc(
        "doc.md",
        "Cats and dogs and fish. Birds in the sky. The cat is on the mat.",
    )
    docs, chunks, skipped = svc.index(str(p), "test")
    assert docs == 1
    assert chunks > 0
    assert skipped == 0

    hits = svc.search("cats", "test", top_k=2)
    assert hits, "search returned nothing"
    assert all(h.corpus == "test" for h in hits)
    assert all(h.source_id == str(p.resolve()) for h in hits)


def test_reindex_unchanged_is_idempotent(svc, make_doc):
    """Chunk count stays accurate when the same source is indexed twice."""
    p = make_doc("doc.md", "First sentence. Second sentence. Third sentence.")

    svc.index(str(p), "test")
    c1 = svc.list_corpora()[0].chunk_count

    docs, _, skipped = svc.index(str(p), "test")
    c2 = svc.list_corpora()[0].chunk_count

    assert docs == 0, "unchanged source should not be indexed again"
    assert skipped == 1
    assert c1 == c2, f"chunk_count drifted on reindex: {c1} -> {c2}"


def test_reindex_after_content_change(svc, make_doc):
    p = make_doc("doc.md", "Original content. Two sentences.")
    svc.index(str(p), "test")
    before = svc.list_corpora()[0].chunk_count

    p.write_text(
        "Replaced content. Different sentences. Now a third one. And a fourth."
    )
    docs, _, skipped = svc.index(str(p), "test")
    after = svc.list_corpora()[0].chunk_count

    assert docs == 1, "modified source should re-index"
    assert skipped == 0
    # Stronger or weaker depending on chunker, but it MUST update.
    assert after >= before  # new content has at least as many sentences


def test_force_reindexes_unchanged(svc, make_doc):
    p = make_doc("doc.md", "Same content. Same sentences.")
    svc.index(str(p), "test")
    pre = StubEmbed.calls
    docs, _, skipped = svc.index(str(p), "test", force=True)
    assert docs == 1, "--force should re-embed"
    assert skipped == 0
    assert StubEmbed.calls > pre


def test_chunk_count_self_heals_after_collection_wiped(svc, make_doc):
    """chunk_count is recounted from the vector store, so a collection wiped
    out-of-band (the real Qdrant-flap-on-reboot scenario) doesn't leave a
    permanently inflated count. Regression for the delta-accumulator bug."""
    from vecgrep.backend.service import _collection_for

    p = make_doc("doc.md", "First sentence. Second sentence. Third sentence.")
    svc.index(str(p), "test")
    coll = _collection_for("test")
    live = svc.store.count(coll)
    assert svc.list_corpora()[0].chunk_count == live

    # Simulate the flap: the vector collection vanishes but the registry keeps
    # its (now stale) chunk_count. The old delta math would compound this error.
    svc.store.drop_collection(coll)
    assert svc.store.count(coll) == 0

    # Re-index the still-present source. chunk_count must reflect reality, not
    # old_count - freed + added.
    docs, _, _ = svc.index(str(p), "test", force=True)
    assert docs == 1
    assert svc.list_corpora()[0].chunk_count == svc.store.count(coll)
    assert svc.list_corpora()[0].chunk_count > 0


def test_hybrid_search_includes_both_retrievers(svc, make_doc):
    p = make_doc(
        "doc.md",
        "The CPI inflation print was 2.8 percent. Federal Reserve held rates steady. "
        "Bonds rallied on the dovish surprise. Tech stocks responded.",
    )
    svc.index(str(p), "test")

    hits = svc.search("CPI inflation", "test", top_k=5, mode="hybrid")
    matched_sets = [set(h.matched_by) for h in hits]
    # At least one hit should have both retrievers; CPI is rare enough
    # that BM25 finds it precisely while the vector model also catches it.
    assert any({"vector", "bm25"} <= s for s in matched_sets), matched_sets


def test_hybrid_floats_keyword_match_above_vector_noise(svc, make_doc):
    """Regression for the "jackson problem":

    nomic-embed-text floors at ~70-75% similarity for any English query, so
    the vector retriever returns 50 noisy near-ties even when nothing
    matches semantically. With unweighted RRF, a genuine literal-keyword
    hit (BM25 rank 1) tied with vector noise (also rank 1) — and vector's
    larger candidate pool dominated.

    Fix: BM25_WEIGHT > 1 in fused RRF. The doc that actually contains the
    rare query token must rank #1 even when vector noise is loud.
    """
    rare = make_doc("rare.md", "Quartzite formations are rare in this region.")
    other = make_doc(
        "other.md",
        "Generic content about rocks, minerals, sediment, and geology in general.",
    )
    svc.index(str(rare), "test")
    svc.index(str(other), "test")

    hits = svc.search("quartzite", "test", top_k=5, mode="hybrid")
    assert hits, "no hits at all"
    top = hits[0]
    assert "rare.md" in top.source_id, (
        f"expected the doc containing 'quartzite' to rank #1, got "
        f"{top.source_id} matched_by={top.matched_by}"
    )
    # And the BM25-only hit should display as confident, not 1.6%-noise.
    assert top.similarity_pct >= 50.0, (
        f"BM25-only hit should display in 'real result' band, got {top.similarity_pct}%"
    )


def test_vector_only_mode_excludes_bm25(svc, make_doc):
    p = make_doc("doc.md", "Cats and dogs. Birds and fish. The end.")
    svc.index(str(p), "test")
    hits = svc.search("cats", "test", top_k=3, mode="vector")
    assert hits
    for h in hits:
        assert h.matched_by == ["vector"]


def test_bm25_only_mode_excludes_vector(svc, make_doc):
    p = make_doc("doc.md", "Quartzite formations rare. Sediment layers shift.")
    svc.index(str(p), "test")
    # Term must be present verbatim for BM25.
    hits = svc.search("quartzite", "test", top_k=3, mode="bm25")
    assert hits
    for h in hits:
        assert h.matched_by == ["bm25"]


def test_filter_source_glob(svc, make_doc):
    a = make_doc("aaa.md", "Apple banana cherry. Date eggplant fennel.")
    b = make_doc("bbb.md", "Apple banana grape. Different content here.")
    svc.index(str(a), "test")
    svc.index(str(b), "test")

    no_filter = svc.search("apple", "test", top_k=5)
    only_a = svc.search("apple", "test", top_k=5, filters=["source:*aaa.md"])

    assert len(no_filter) >= 2
    assert all("aaa.md" in h.source_id for h in only_a)


def test_filter_corpus_name(svc, make_doc):
    p = make_doc("doc.md", "Some content. More content here.")
    svc.index(str(p), "test")

    matched = svc.search("content", filters=["corpus:test"])
    not_matched = svc.search("content", filters=["corpus:nope"])
    assert matched
    assert not not_matched


def test_explain_populates_score_breakdown(svc, make_doc):
    p = make_doc("doc.md", "Federal Reserve holds rates. Inflation cools. Bonds rally.")
    svc.index(str(p), "test")

    hits = svc.search("inflation", "test", top_k=2, mode="hybrid", explain=True)
    assert hits
    for h in hits:
        assert "rrf" in h.explain
        # At least one of vector/bm25 must have placed it.
        assert "vector_cosine" in h.explain or "bm25_score" in h.explain


def test_delete_corpus_removes_all_state(svc, make_doc):
    p = make_doc("doc.md", "Some sentences. More sentences here.")
    svc.index(str(p), "test")
    assert svc.list_corpora()

    svc.delete_corpus("test")
    assert not svc.list_corpora()
    # BM25 pickle gone too
    assert not (svc.settings.home / "bm25" / "test.pkl").exists()


def test_delete_source_removes_only_that_source(svc, make_doc):
    a = make_doc("a.md", "Cats sit on mats. Cats are nice.")
    b = make_doc("b.md", "Dogs run in parks. Dogs bark loudly.")
    svc.index(str(a), "test")
    svc.index(str(b), "test")

    corpus = svc.list_corpora()[0]
    assert str(a.resolve()) in corpus.sources
    assert str(b.resolve()) in corpus.sources
    chunks_before = corpus.chunk_count

    svc.delete_source("test", str(a.resolve()))

    corpus = svc.list_corpora()[0]
    assert str(a.resolve()) not in corpus.sources
    assert str(b.resolve()) in corpus.sources
    assert corpus.doc_count == 1
    assert corpus.chunk_count < chunks_before
    # B is still searchable
    hits = svc.search("dogs", "test", top_k=2)
    assert hits and all(h.source_id == str(b.resolve()) for h in hits)
    # A is not
    hits = svc.search("cats", "test", top_k=2)
    assert not hits or all(h.source_id != str(a.resolve()) for h in hits)


# ============================================================================
# Cosine-to-percentage calibration tests
#
# The sigmoid-calibrated _cosine_to_pct should:
#   - put unrelated content (cos ~0.5) well below 30% (so users can ignore it)
#   - put genuine matches (cos ~0.75) well above 60%
#   - put strong matches (cos ~0.85+) above 85%
# ============================================================================


def test_cosine_to_pct_noise_below_30():
    from vecgrep.backend.service import _cosine_to_pct
    # Empirical: nomic-embed-text gives cos=0.50 for unrelated queries.
    assert _cosine_to_pct(0.50) < 30.0
    assert _cosine_to_pct(0.45) < 20.0
    # And outright nothing reads as basically zero.
    assert _cosine_to_pct(0.0) < 1.0


def test_cosine_to_pct_strong_match_above_85():
    from vecgrep.backend.service import _cosine_to_pct
    assert _cosine_to_pct(0.85) > 85.0
    assert _cosine_to_pct(0.92) > 95.0


def test_cosine_to_pct_signal_band_meaningful():
    """Display % should monotonically increase with cosine in the band users
    actually inspect. The old (cos+1)/2 mapping crushed signal into 70-90%;
    the new mapping spreads it across 30-95%."""
    from vecgrep.backend.service import _cosine_to_pct
    weak = _cosine_to_pct(0.65)
    mid = _cosine_to_pct(0.75)
    strong = _cosine_to_pct(0.85)
    assert weak < mid < strong
    # And the spread between weak and strong is at least 30 points
    # (vs the old map, where 0.65→82% and 0.85→92%, a 10-point spread).
    assert strong - weak > 30, f"expected wide spread, got {strong - weak:.1f}"


def test_cosine_to_pct_overrides_apply():
    """Allow per-call center/slope overrides for the tuning UI."""
    from vecgrep.backend.service import _cosine_to_pct
    # Lower the center to 0.5 — now cos=0.5 reads as 50%.
    assert abs(_cosine_to_pct(0.5, center=0.5) - 50.0) < 0.1
    # Steeper slope makes the mapping sharper.
    flat = _cosine_to_pct(0.70, slope=4)
    steep = _cosine_to_pct(0.70, slope=20)
    assert steep > flat, "steeper slope should rise faster past the center"


# ============================================================================
# Chunk-window expansion (powers the click-to-expand UI)
# ============================================================================


def test_get_chunk_window_returns_expanded_context(svc, make_doc):
    body = "Header. " + ("Filler sentence. " * 200) + "Target sentence here. " + ("Tail. " * 200)
    p = make_doc("expand.md", body)
    svc.index(str(p), "test")

    hits = svc.search("Target sentence here", "test", top_k=1)
    assert hits, "search returned nothing"
    cid = hits[0].chunk_id
    assert cid, "search hit should carry a chunk_id"

    # Default window slice — should be wider than the 400-char sidewindow
    # baked into _payload_to_result.
    wide = svc.get_chunk_window("test", cid, window=2000)
    assert wide is not None
    assert len(wide["before"]) + len(wide["after"]) > 800
    assert "Target sentence here" in wide["chunk"]
    assert wide["source_length"] == len(body)


def test_get_chunk_window_full_returns_whole_source(svc, make_doc):
    body = "Alpha. " * 50 + "TARGET. " + "Omega. " * 50
    p = make_doc("full.md", body)
    svc.index(str(p), "test")

    hits = svc.search("TARGET", "test", top_k=1)
    cid = hits[0].chunk_id
    full = svc.get_chunk_window("test", cid, window=-1)
    assert full is not None
    # The whole document should be reconstructible from before + chunk + after
    assert full["before"] + full["chunk"] + full["after"] == body


def test_get_chunk_window_missing_chunk_returns_none(svc, make_doc):
    p = make_doc("doc.md", "Just some content.")
    svc.index(str(p), "test")
    result = svc.get_chunk_window("test", "nonexistent-chunk-id", window=500)
    assert result is None
