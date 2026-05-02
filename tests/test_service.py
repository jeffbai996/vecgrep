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
