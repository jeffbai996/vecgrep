"""Regression tests for the 2026-10-28 vecgrep scaling audit (Phases 1-3)."""
from __future__ import annotations

from vecgrep.backend.mutation_journal import MutationJournal
from vecgrep.backend.service import (
    SearchResult,
    _collection_for,
)
from vecgrep.backend.store.bm25_sqlite import BM25SqliteStore


def test_rerank_pool_orders_by_fused_score_before_display_pct(svc, monkeypatch):
    """BM25-only hits cap similarity_pct at 39.0, so sorting the pre-rerank
    pool by similarity_pct alone dropped exact identifier hits behind weak
    vector-only hits (~42%) whenever len(candidates) > RERANK_POOL_MAX."""
    import vecgrep.backend.rerank as rerank_mod
    import vecgrep.backend.service as svc_mod

    monkeypatch.setattr(svc_mod, "RERANK_POOL_MAX", 2)
    seen_chunks: list[str] = []

    def fake_rerank(query, pairs, model_name):
        seen_chunks.extend(text for text, _ in pairs)
        return [(0.9 if "exact_symbol" in text else 0.2, payload) for text, payload in pairs]

    monkeypatch.setattr(rerank_mod, "rerank", fake_rerank)

    candidates = [
        SearchResult(
            score=0.010,
            similarity_pct=45.0,
            chunk="weak vector hit 1",
            chunk_start=0,
            chunk_end=10,
            context_before="",
            context_after="",
            source_id="/a.py",
            corpus="code",
            metadata={},
            chunk_id="c1",
            matched_by=["vector"],
        ),
        SearchResult(
            score=0.009,
            similarity_pct=44.0,
            chunk="weak vector hit 2",
            chunk_start=0,
            chunk_end=10,
            context_before="",
            context_after="",
            source_id="/b.py",
            corpus="code",
            metadata={},
            chunk_id="c2",
            matched_by=["vector"],
        ),
        SearchResult(
            score=0.045,
            similarity_pct=39.0,
            chunk="def exact_symbol(): pass",
            chunk_start=0,
            chunk_end=24,
            context_before="",
            context_after="",
            source_id="/c.py",
            corpus="code",
            metadata={},
            chunk_id="c3",
            matched_by=["bm25"],
        ),
    ]

    out = svc._apply_rerank("exact_symbol", candidates, top_k=2, model_name="fake")
    assert "def exact_symbol(): pass" in seen_chunks
    assert out[0].chunk == "def exact_symbol(): pass"


def test_filter_pushdown_finds_matches_beyond_candidate_pool(svc, make_doc, monkeypatch):
    """A channel:/source: filter must not return [] just because the matching
    source sits below the top CANDIDATE_POOL unfiltered chunks."""
    import vecgrep.backend.service as svc_mod

    monkeypatch.setattr(svc_mod, "CANDIDATE_POOL", 3)
    for i in range(6):
        p = make_doc(
            f"noise_{i}.md",
            f"---\nchannel: general\n---\nalpha shared keyword in noise {i}.",
        )
        svc.index(str(p), "chats")

    target = make_doc(
        "target_ops.md",
        "---\nchannel: ops\n---\nalpha shared keyword in ops channel.",
    )
    svc.index(str(target), "chats")

    vec_hits = svc.search(
        "alpha shared keyword", "chats", top_k=3, mode="vector", filters=["channel:ops"]
    )
    assert len(vec_hits) == 1
    assert vec_hits[0].source_id == str(target.resolve())

    bm25_hits = svc.search(
        "alpha shared keyword",
        "chats",
        top_k=3,
        mode="bm25",
        filters=["source:*target_ops.md"],
    )
    assert len(bm25_hits) == 1
    assert bm25_hits[0].source_id == str(target.resolve())


def test_multi_paragraph_turn_propagates_speaker_to_body_chunks(svc, make_doc):
    """When a long transcript turn splits across multiple sentence-window
    chunks, continuation chunks without a `**speaker** · HH:MM` line still
    inherit the enclosing turn's speaker."""
    para = " ".join(
        f"Sentence number {i} discusses deployment throughput and memory limits."
        for i in range(30)
    )
    doc = make_doc("chat.md", f"**Alice** · 14:20\n{para}\n")
    svc.index(str(doc), "chats")

    points = dict(svc.store.iter_payloads(_collection_for("chats")))
    assert len(points) >= 2
    for payload in points.values():
        assert payload["metadata"].get("speakers") == ["Alice"]

    hits = svc.search(
        "Sentence number 28 discusses deployment",
        "chats",
        filters=["speaker:alice"],
    )
    assert hits


def test_mutation_journal_advance_phase_uses_sidecar(tmp_path):
    journal = MutationJournal(tmp_path / "mutations")
    big_record = {
        "version": 1,
        "corpus": "chats",
        "operation": "index_source",
        "phase": "prepared",
        "source_id": "/tmp/chat.md",
        "old_points": [{"id": f"p{i}", "vector": [0.1] * 8, "payload": {}} for i in range(50)],
    }
    journal.write(big_record)
    main_mtime = journal.path("chats").stat().st_mtime_ns

    journal.advance_phase("chats", "qdrant_done")
    assert journal.phase_path("chats").exists()
    assert journal.path("chats").stat().st_mtime_ns == main_mtime
    read_back = journal.read("chats")
    assert read_back is not None
    assert read_back["phase"] == "qdrant_done"
    assert len(read_back["old_points"]) == 50

    journal.finish("chats")
    assert not journal.has("chats")
    assert not journal.phase_path("chats").exists()


def test_incremental_tail_append_embeds_only_new_chunks_and_strips_bm25_source_text(
    svc, make_doc, stub_backend, monkeypatch
):
    lines = [
        f"Paragraph {i} has several sentences about service scaling. "
        f"Detail {i} covers indexing and query latency."
        for i in range(25)
    ]
    doc = make_doc("growing.md", "\n\n".join(lines))
    svc.index(str(doc), "notes")

    initial_chunks = svc.store.count(_collection_for("notes"))
    assert initial_chunks >= 3

    embedded_batches: list[list[str]] = []
    real_embed = stub_backend.embed

    def spy_embed(texts: list[str]) -> list[list[float]]:
        embedded_batches.append(list(texts))
        return real_embed(texts)

    monkeypatch.setattr(stub_backend, "embed", spy_embed)
    monkeypatch.setattr(
        svc.store,
        "delete_by_source",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("incremental append must not full-scan delete_by_source")
        ),
    )

    lines.append(
        "Paragraph 999 introduces brand new tail content about quantized vectors. "
        "It sits at the very end of the growing transcript."
    )
    doc.write_text("\n\n".join(lines))
    svc.index(str(doc), "notes")

    new_total = svc.store.count(_collection_for("notes"))
    assert new_total >= initial_chunks
    total_embedded = sum(len(b) for b in embedded_batches)
    assert total_embedded < new_total, (
        f"expected incremental embed of tail delta only, got {total_embedded}/{new_total}"
    )

    for p in svc.bm25.iter_payloads("notes"):
        assert "source_text" not in p

    hits = svc.search("quantized vectors", "notes")
    assert hits


def test_bm25_sqlite_two_stage_and_deferred_payload_and_count_cache(tmp_path):
    store = BM25SqliteStore(tmp_path / "bm25")
    try:
        ids = [f"id-{i}" for i in range(10)]
        texts = [
            f"common_term noise document {i}" if i < 9 else "common_term rare_needle exact hit"
            for i in range(10)
        ]
        payloads = [
            {
                "corpus": "c",
                "source_id": f"/src/{i}.md",
                "chunk_index": 0,
                "text": texts[i],
                "source_text": "SHOULD_BE_STRIPPED",
            }
            for i in range(10)
        ]
        store.upsert("c", ids, texts, payloads)

        assert store.count("c") == 10
        assert store.count("c") == 10  # hits stat+data_version cache
        assert "source_text" not in (store.get_by_id("c", "id-0") or {})

        hits = store.search("c", "common_term rare_needle", top_k=3)
        assert hits
        assert hits[0][0] == "id-9"
        assert "source_text" not in hits[0][2]

        filtered = store.search(
            "c", "common_term", top_k=5, source_ids={"/src/2.md"}
        )
        assert [cid for cid, _, _ in filtered] == ["id-2"]

        store.delete_ids("c", ["id-0", "id-1"])
        assert store.count("c") == 8
    finally:
        store.close_all()


def test_stats_filterable_fields_and_browse_tail_use_explorer_catalog(
    svc, make_doc, monkeypatch
):
    d1 = make_doc(
        "day1.md",
        "---\nchannel: ops\n---\nSaved: 2026-06-01T00:00:00+00:00\n\n"
        "**Alice** · 09:00\nDay one first message.\n",
    )
    d2 = make_doc(
        "day2.md",
        "---\nchannel: ops\n---\nSaved: 2026-06-02T00:00:00+00:00\n\n"
        "**Bob** · 10:00\nDay two first message.\n"
        "**Bob** · 10:05\nDay two newest message.\n",
    )
    svc.index(str(d1), "chats")
    svc.index(str(d2), "chats")

    monkeypatch.setattr(
        svc.bm25,
        "iter_payloads",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("stats/filterable_fields/browse must use explorer.db, not iter_payloads")
        ),
    )

    st = svc.corpus_stats("chats")
    assert st["docs"] == 2
    assert st["chunks"] >= 2
    assert st["days_covered"] == 2

    schema = svc.filterable_fields("chats")
    assert schema["has_doc_timestamp"] is True
    assert "ops" in schema["filters"]["meta"]["keys"].get("channel", [])

    fetched_sources: list[str] = []
    real_source_text = svc._source_text_for

    def spy_source_text(corpus_name: str, source_id: str) -> str:
        fetched_sources.append(source_id)
        return real_source_text(corpus_name, source_id)

    monkeypatch.setattr(svc, "_source_text_for", spy_source_text)
    groups = svc.browse("chats", channel="ops", tail=1)
    assert len(groups) == 1
    assert len(groups[0]["events"]) == 1
    assert groups[0]["events"][0]["text"] == "Day two newest message."
    assert fetched_sources == [str(d2.resolve())]
