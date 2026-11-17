from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests import eval_harness as eh
from tests.conftest import StubEmbed
from vecgrep.backend.embed.base import EmbedBackend
from vecgrep.backend.embed.cache import CachedBackend
from vecgrep.backend.service import VecgrepService, _collection_for


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


def test_diagnose_flags_wiped_collection_and_reconcile_rebuilds(svc, make_doc):
    """diagnose() surfaces a corpus whose collection was wiped out-of-band, and
    reconcile(reindex=True) rebuilds it from the recorded source. This is the
    prevention path for the Qdrant-flap data loss."""
    from vecgrep.backend.service import _collection_for

    p = make_doc("doc.md", "Alpha sentence. Beta sentence. Gamma sentence.")
    svc.index(str(p), "test")
    assert svc.diagnose() == []  # healthy right after index

    # Wipe the collection but keep the registry row (the flap).
    svc.store.drop_collection(_collection_for("test"))
    issues = svc.diagnose()
    kinds = {i["kind"] for i in issues}
    assert "missing_collection" in kinds
    miss = next(i for i in issues if i["kind"] == "missing_collection")
    assert miss["fixable"] is True  # the source file still exists

    # Repair rebuilds from source; the corpus is searchable + reconciled again.
    svc.reconcile(reindex=True)
    assert svc.diagnose() == []
    assert svc.list_corpora()[0].chunk_count == svc.store.count(_collection_for("test"))
    assert svc.search("beta", "test", top_k=3)


def test_diagnose_flags_count_drift_and_reconcile_recounts(svc, make_doc):
    """A registry chunk_count that diverges from Qdrant is flagged and recounted
    without needing a re-embed."""
    p = make_doc("doc.md", "One. Two. Three. Four.")
    svc.index(str(p), "test")
    c = svc.list_corpora()[0]
    # Corrupt the registry count to simulate historical drift.
    c.chunk_count = 999
    svc.registry.upsert(c)

    issues = svc.diagnose()
    assert any(i["kind"] == "count_drift" for i in issues)

    svc.reconcile()  # recount only, no reindex flag needed
    from vecgrep.backend.service import _collection_for
    assert svc.list_corpora()[0].chunk_count == svc.store.count(_collection_for("test"))
    assert svc.diagnose() == []


def test_reconcile_rebuilds_missing_bm25_from_qdrant(svc, make_doc):
    """A vector-only recovery must restore exact-keyword retrieval too.

    Qdrant stores each chunk's text and payload, so rebuilding BM25 from those
    live points must not re-embed or re-ingest the original source files.
    """
    p = make_doc("doc.md", "Quartzite is the exact keyword this corpus needs.")
    svc.index(str(p), "test")
    svc.bm25.drop("test")

    assert any(i["kind"] == "missing_bm25" for i in svc.diagnose())

    actions = svc.reconcile()

    assert actions == [{"corpus": "test", "kind": "missing_bm25", "action": "rebuilt_bm25"}]
    assert svc.diagnose() == []
    hits = svc.search("quartzite", "test", top_k=3, mode="bm25")
    assert hits
    assert hits[0].matched_by == ["bm25"]
    # The redundant full source stays in Qdrant, not duplicated into every
    # rebuilt BM25 chunk; source expansion still fetches the canonical copy.
    assert "source_text" not in next(svc.bm25.iter_payloads("test"))
    source = svc.get_source("test", str(p.resolve()))
    assert source is not None
    assert source["text"] == p.read_text()


def test_reconcile_rebuilds_bm25_count_drift_from_qdrant(svc, make_doc):
    first = make_doc("first.md", "Quartzite belongs in the first document.")
    second = make_doc("second.md", "Feldspar belongs in the second document.")
    svc.index(str(first), "test")
    svc.index(str(second), "test")
    expected = svc.list_corpora()[0].chunk_count
    assert expected > 1

    payload = next(svc.bm25.iter_payloads("test"))
    cid, payload = svc.bm25.first_chunk_for_source("test", payload["source_id"])
    svc.bm25.replace("test", [
        (cid, payload["text"], payload),
    ])

    issue = next(i for i in svc.diagnose() if i["kind"] == "bm25_count_drift")
    assert issue["fixable"] is True
    assert issue["detail"] == f"vector store {expected} vs BM25 1"

    actions = svc.reconcile()
    assert actions == [{
        "corpus": "test",
        "kind": "bm25_count_drift",
        "action": "rebuilt_bm25",
    }]
    assert svc.bm25.count("test") == expected
    assert svc.diagnose() == []


def test_reconcile_can_be_scoped_to_one_corpus(svc, make_doc):
    """A targeted recovery must not repair unrelated live corpora.

    A broad doctor run once chose the active chats corpus before the broken CLI
    corpus.  The selection belongs in the service layer so every caller gets
    the same hard boundary, not merely a CLI-level filter.
    """
    from vecgrep.backend.service import _collection_for

    first = make_doc("first.md", "First corpus sentence. Another first sentence.")
    second = make_doc("second.md", "Second corpus sentence. Another second sentence.")
    svc.index(str(first), "first")
    svc.index(str(second), "second")

    for name in ("first", "second"):
        corpus = svc.registry.get(name)
        corpus.chunk_count = 999
        svc.registry.upsert(corpus)

    issues = svc.diagnose(corpora={"first"})
    assert [issue["corpus"] for issue in issues] == ["first"]

    svc.reconcile(corpora={"first"})
    assert svc.registry.get("first").chunk_count == svc.store.count(_collection_for("first"))
    assert svc.registry.get("second").chunk_count == 999


def test_recovery_reuses_partial_points_without_source_delete(svc, make_doc, monkeypatch):
    """A partial rebuild resumes by deterministic IDs, not repeated full scans.

    `source_id` is not indexed in Qdrant.  Deleting by that field before every
    recovered source turns a 2k-source repair into thousands of collection
    scans, even though the deterministic point IDs make a same-snapshot upsert
    idempotent already.
    """
    from vecgrep.backend.service import _collection_for

    first = make_doc("first.md", "First corpus sentence. Another first sentence.")
    second = make_doc("second.md", "Second corpus sentence. Another second sentence.")
    svc.index(str(first), "test")
    svc.index(str(second), "test")
    svc.store.delete_by_source(_collection_for("test"), str(second.resolve()))

    def source_delete_would_scan(*_args, **_kwargs):
        raise AssertionError("partial recovery must not delete by source")

    monkeypatch.setattr(svc.store, "delete_by_source", source_delete_would_scan)
    actions = svc.reconcile(reindex=True, corpora={"test"})

    assert actions == [{"corpus": "test", "kind": "count_drift", "action": "reindexed"}]
    assert svc.diagnose(corpora={"test"}) == []


def test_recovery_skips_a_complete_source_already_in_qdrant(svc, make_doc, monkeypatch):
    """Resume only source files whose deterministic points are incomplete."""
    from vecgrep.backend.service import _collection_for

    first = make_doc("first.md", "First corpus sentence. Another first sentence.")
    second = make_doc("second.md", "Second corpus sentence. Another second sentence.")
    svc.index(str(first), "test")
    svc.index(str(second), "test")
    svc.store.delete_by_source(_collection_for("test"), str(second.resolve()))

    upserted_sources = []
    original_upsert = svc.store.upsert

    def record_upsert(collection, ids, vectors, payloads):
        upserted_sources.extend(payload["source_id"] for payload in payloads)
        return original_upsert(collection, ids, vectors, payloads)

    monkeypatch.setattr(svc.store, "upsert", record_upsert)
    svc.reconcile(reindex=True, corpora={"test"})

    assert str(first.resolve()) not in upserted_sources
    assert str(second.resolve()) in upserted_sources


def test_recovery_persists_completed_progress(svc, make_doc, vg_home):
    """A watchdog can distinguish real recovery work from a stuck process."""
    from vecgrep.backend.service import _collection_for

    doc = make_doc("doc.md", "Recovery progress must be durable.")
    svc.index(str(doc), "test")
    svc.store.drop_collection(_collection_for("test"))

    svc.reconcile(reindex=True, corpora={"test"})

    progress = json.loads((vg_home / "recovery-progress.json").read_text())
    assert progress["corpus"] == "test"
    assert progress["source_done"] == progress["sources_total"] == 1
    assert progress["chunks_done"] == svc.store.count(_collection_for("test"))
    assert progress["active"] is False
    assert progress["last_progress_at"] > 0


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

    hits = svc.search("Target sentence here", "test", top_k=1, mode="bm25")
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


def test_unchanged_source_skips_chunker(svc, make_doc, monkeypatch):
    p = make_doc('unchanged.md', 'Existing evidence. Another sentence.')
    svc.index(str(p), 'test')

    class UnreachableChunker:
        def chunk_doc(self, doc):
            raise AssertionError('unchanged source reached chunker')

    monkeypatch.setattr(svc, 'chunker', lambda name: UnreachableChunker())
    assert svc.index(str(p), 'test') == (0, 0, 1)


def test_recovery_count_mismatch_is_not_skipped_by_matching_hash(svc, make_doc):
    p = make_doc('incomplete.md', 'Existing evidence. Another sentence.')
    _, count, _ = svc.index(str(p), 'test')
    docs, chunks, skipped = svc.index(str(p), 'test',
        resume_source_counts={str(p.resolve()): max(0, count - 1)})
    assert docs == 1 and chunks == count and skipped == 0

# ── test_tools_v12 ──

def _built(svc):
    eh.build_eval_corpus(svc)
    return svc


# ── related ──────────────────────────────────────────────────────────────────

def test_related_excludes_self_and_bounds(svc) -> None:
    svc = _built(svc)
    seed = svc.search("relay-service", "evalchat", top_k=1)[0]
    rel = svc.related(seed.chunk_id, "evalchat", top_k=5)
    assert rel, "neighbours exist in a 55-chunk corpus"
    assert len(rel) <= 5
    assert all(r.chunk_id != seed.chunk_id for r in rel)
    assert all(r.corpus == "evalchat" for r in rel)


def test_related_unknown_chunk_raises(svc) -> None:
    svc = _built(svc)
    import pytest

    from vecgrep.backend.service import CorpusError
    with pytest.raises(CorpusError):
        svc.related("nope-not-a-chunk", "evalchat")


# ── compare ──────────────────────────────────────────────────────────────────

def test_compare_two_windows(svc) -> None:
    """Fixtures span 2026-01-08 and 2026-01-15 — split the windows between
    them and both sides populate, with per-side source sets and the delta."""
    svc = _built(svc)
    out = svc.compare(
        "relay-service", "evalchat",
        a_after="2026-01-01", a_before="2026-01-12",
        b_after="2026-01-12", b_before="2026-01-31",
        top_k=8,
    )
    assert out["a"]["results"] and out["b"]["results"]
    a_srcs = set(out["a"]["sources"])
    b_srcs = set(out["b"]["sources"])
    assert a_srcs and b_srcs and a_srcs.isdisjoint(b_srcs)
    assert set(out["only_in_b"]) == b_srcs - a_srcs
    assert out["windows"]["a"]["after"] == "2026-01-01"


def test_compare_empty_window_is_explicit(svc) -> None:
    svc = _built(svc)
    out = svc.compare(
        "relay-service", "evalchat",
        a_after="2020-01-01", a_before="2020-02-01",   # nothing there
        b_after="2026-01-01", b_before="2026-01-31",
    )
    assert out["a"]["results"] == []
    assert out["b"]["results"]


# ── corpus_stats ─────────────────────────────────────────────────────────────

def test_corpus_stats_counts_and_span(svc) -> None:
    svc = _built(svc)
    st = svc.corpus_stats("evalchat")
    assert st["corpus"] == "evalchat"
    assert st["chunks"] == 55
    assert st["docs"] == 4
    assert st["date_span"]["first"].startswith("2026-01-")
    assert st["date_span"]["last"].startswith("2026-01-")
    # fixtures are 01-08 and 01-15 per channel — the days between are gaps
    assert st["gap_days"] >= 1
    assert st["sources"]  # per-source chunk counts present


def test_corpus_stats_unknown_corpus_raises(svc) -> None:
    import pytest

    from vecgrep.backend.service import CorpusError
    with pytest.raises(CorpusError):
        svc.corpus_stats("nope")


# ── summarize_corpus ─────────────────────────────────────────────────────────

def test_summarize_corpus_rollup(svc) -> None:
    svc = _built(svc)
    out = svc.summarize_corpus("evalchat", sample=10)
    assert out["corpus"] == "evalchat"
    assert out["chunks"] == 55
    # enrichment-powered speaker tally (fixtures have alice/bob/ops-bot)
    speakers = dict(out["top_speakers"])
    assert "alice" in speakers
    assert out["date_span"]["first"]
    # sampling is explicit, never silent
    assert out["sampled"] is True
    assert len(out["sample_chunks"]) == 10
    assert all(isinstance(c, dict) and c.get("text") for c in out["sample_chunks"])


def test_summarize_corpus_window_filters(svc) -> None:
    svc = _built(svc)
    out = svc.summarize_corpus("evalchat", after="2026-01-12", sample=10)
    # only the 01-15 docs → fewer chunks than the whole corpus
    assert 0 < out["chunks"] < 55

# ── test_orphaned_source ──

def _index(svc: VecgrepService, path: Path, corpus: str) -> None:
    svc.index(str(path), corpus)


def test_diagnose_reports_a_source_whose_file_is_gone(svc, make_doc):
    live = make_doc("live.md", "the fund GPs are Alice and Bob")
    dead = make_doc("dead.md", "the fund GPs are Carol and Dave")
    _index(svc, live, "fixture")
    _index(svc, dead, "fixture")

    dead.unlink()

    issues = svc.diagnose(corpora={"fixture"})
    orphans = [i for i in issues if i["kind"] == "orphaned_source"]
    assert len(orphans) == 1, f"expected the deleted source to be reported, got {issues}"
    assert orphans[0]["source_id"] == str(dead)
    assert orphans[0]["fixable"] is True


def test_reconcile_purges_the_orphan_from_both_backends(svc, make_doc):
    live = make_doc("live.md", "the fund GPs are Alice and Bob")
    dead = make_doc("dead.md", "the fund GPs are Carol and Dave")
    _index(svc, live, "fixture")
    _index(svc, dead, "fixture")
    dead.unlink()

    svc.reconcile(corpora={"fixture"})

    # Vector side: no points left for the dead source.
    collection = "vecgrep__fixture"
    assert svc.store.source_counts(collection).get(str(dead), 0) == 0

    # BM25 side: this is the half-purge trap. A keyword query that the dead
    # doc would answer must not return it.
    hits = svc.bm25.search("fixture", "Carol Dave fund GPs", 10)
    dead_hits = [h for h in hits if h[2].get("source_id") == str(dead)]
    assert not dead_hits, f"BM25 still serves the deleted source: {dead_hits}"

    # The living source is untouched — a purge must not be a corpus wipe.
    assert svc.store.source_counts(collection).get(str(live), 0) > 0


def test_reconcile_leaves_a_present_source_alone(svc, make_doc):
    keep = make_doc("keep.md", "the fund GPs are Alice and Bob")
    _index(svc, keep, "fixture")

    svc.reconcile(corpora={"fixture"})

    collection = "vecgrep__fixture"
    assert svc.store.source_counts(collection).get(str(keep), 0) > 0
    corpus = svc.registry.get("fixture")
    assert str(keep) in corpus.sources


def test_url_sources_are_never_treated_as_orphans(svc, make_doc):
    """A URL can't be stat'd; treating an unreachable one as deleted would
    purge real data on a network blip."""
    doc = make_doc("page.md", "the fund GPs are Alice and Bob")
    _index(svc, doc, "fixture")
    corpus = svc.registry.get("fixture")
    corpus.sources.append("https://example.com/gone")
    svc.registry.upsert(corpus)

    issues = svc.diagnose(corpora={"fixture"})
    orphans = [i for i in issues if i["kind"] == "orphaned_source"]
    assert not orphans, f"a URL source must never be reported as orphaned: {orphans}"

# ── test_migrate ──

class _M1(EmbedBackend):
    name = "stub"
    model = "m1"
    dim = 8

    def embed(self, texts):
        return [[(len(t) % 7 + i) / 10 for i in range(self.dim)] for t in texts]


class _M2(EmbedBackend):
    name = "ollama"
    model = "m2"
    dim = 8

    def embed(self, texts):
        return [[(len(t) % 5 + i) / 10 for i in range(self.dim)] for t in texts]


@pytest.mark.parametrize("lexical_backend", ["pickle", "sqlite"])
def test_migrate_swaps_backend_and_keeps_data(svc, make_doc, monkeypatch, lexical_backend):
    if lexical_backend == "sqlite":
        from vecgrep.backend.store.bm25_sqlite import BM25SqliteStore

        svc.bm25 = BM25SqliteStore(svc.settings.home / "bm25")
    p = make_doc("doc.md", "Cats and dogs and fish. Three sentences here.")
    svc.index(str(p), "test")
    pre_hits = svc.search("cats", "test", top_k=5)
    assert pre_hits

    # Patch the factory so migrate() resolves to M2 regardless of env.
    import vecgrep.backend.service as s_mod

    monkeypatch.setattr(s_mod, "get_embed_backend", lambda settings, prefer=None: _M2())

    out = svc.migrate_corpus("test", to_backend="ollama", to_model="m2")
    assert out.embed_backend == "ollama"
    assert out.embed_model == "m2"
    assert out.dim == 8

    # Force the search backend to M2 (cache cleared because the old M1 entry
    # is now stale for this corpus).
    svc._backend_cache.clear()
    svc._backend_cache["auto"] = CachedBackend(_M2(), svc._embed_cache)
    svc._backend_cache["ollama"] = svc._backend_cache["auto"]

    post_hits = svc.search("cats", "test", top_k=5)
    assert post_hits, "search after migrate should still return hits"
    # Payload's `corpus` field must be the final name, not the temp.
    for h in post_hits:
        assert h.corpus == "test"


def test_migrate_no_op_raises(svc, make_doc):
    p = make_doc("doc.md", "noop test content here")
    svc.index(str(p), "test")
    import pytest

    from vecgrep.backend.store import CorpusError

    # Asking to migrate to the same backend+model the corpus is already on
    # should refuse rather than silently churn.
    corpus = svc.list_corpora()[0]
    with pytest.raises(CorpusError):
        svc.migrate_corpus(
            "test",
            to_backend=corpus.embed_backend,
            to_model=corpus.embed_model,
        )

# ── test_corpus_datatype ──

def _dt(svc, name: str) -> str:
    info = svc.store.client.get_collection(_collection_for(name))
    return str(info.config.params.vectors.datatype).lower()


def test_new_corpus_defaults_to_float32(svc, make_doc):
    svc.index(str(make_doc("a.md", "hello world")), "c1")
    assert svc.registry.get("c1").datatype == "float32"


def test_migrate_converts_the_collection_and_persists_the_choice(svc, make_doc):
    svc.index(str(make_doc("a.md", "alpha beta gamma")), "c1")
    before = svc.store.client.count(_collection_for("c1"), exact=True).count
    assert before > 0

    svc.migrate_datatype("c1", "float16")

    assert svc.registry.get("c1").datatype == "float16"
    assert "float16" in _dt(svc, "c1")
    assert svc.store.client.count(_collection_for("c1"), exact=True).count == before


def test_a_reindex_after_migrating_keeps_float16(svc, make_doc):
    """The whole point. An index that recreates the collection must not
    quietly hand the corpus back to float32."""
    svc.index(str(make_doc("a.md", "alpha beta gamma")), "c1")
    svc.migrate_datatype("c1", "float16")
    svc.index(str(make_doc("b.md", "delta epsilon")), "c1")
    assert svc.registry.get("c1").datatype == "float16"
    assert "float16" in _dt(svc, "c1")


def test_migrating_to_the_current_datatype_is_a_no_op(svc, make_doc):
    svc.index(str(make_doc("a.md", "alpha")), "c1")
    n = svc.store.client.count(_collection_for("c1"), exact=True).count
    svc.migrate_datatype("c1", "float32")
    assert svc.store.client.count(_collection_for("c1"), exact=True).count == n


def test_an_unknown_datatype_is_refused(svc, make_doc):
    import pytest

    svc.index(str(make_doc("a.md", "alpha")), "c1")
    with pytest.raises(ValueError):
        svc.migrate_datatype("c1", "bfloat16")
    assert svc.registry.get("c1").datatype == "float32"


def test_quantization_persists_across_datatype_migration_and_reindex(
    svc, make_doc, monkeypatch
):
    from qdrant_client import models as qm

    svc.index(str(make_doc("a.md", "alpha beta gamma")), "c1")
    updated = svc.set_quantization("c1", "int8")
    assert updated.quantization == "int8"
    assert svc.registry.get("c1").quantization == "int8"

    created_quant: dict[str, object] = {}
    orig_create = svc.store.client.create_collection

    def spy_create(collection_name: str, **kwargs):
        created_quant[collection_name] = kwargs.get("quantization_config")
        return orig_create(collection_name=collection_name, **kwargs)

    monkeypatch.setattr(svc.store.client, "create_collection", spy_create)
    svc.migrate_datatype("c1", "float16")
    live_col = _collection_for("c1")
    info = svc.store.client.get_collection(live_col)
    assert info.config.params.vectors.datatype == qm.Datatype.FLOAT16
    qcfg = created_quant.get(live_col)
    assert isinstance(qcfg, qm.ScalarQuantization)
    assert qcfg.scalar.always_ram is True
    assert svc.registry.get("c1").quantization == "int8"

    svc.index(str(make_doc("b.md", "delta epsilon")), "c1")
    assert svc.registry.get("c1").quantization == "int8"
