"""source_text is stored once per source, not once per chunk.

Every chunk used to carry a byte-identical copy of its entire source
document (~8.3 GB across the live corpora, 91%+ redundant). The full text
now lives only on the chunk_index==0 point; per-chunk context that the hot
search path needs is precomputed into bounded fields at write time.

These tests pin the contract in both directions: storage really shrinks,
and no reader loses fidelity.
"""
from __future__ import annotations

import pytest

from vecgrep.backend.service import _chunk_id, _collection_for


# A doc long enough to chunk several ways, with newlines at known spots so
# line anchors are checkable, and speaker lines so parse_events has input.
# Each message ends in a period: the sentence-window chunker splits on
# terminal punctuation, so without it the whole file is a single chunk and
# these tests would prove nothing.
def _msg(i: int) -> str:
    return (
        f"message number {i} about {'alpha' if i % 2 else 'beta'} and some "
        f"filler text to give the chunker something to bite on."
    )


# Archiver speaker-header format (`**name** · HH:MM`) -- what parse_events
# actually matches.
TRANSCRIPT = "\n".join(
    f"**speaker{i % 3}** · 10:{i:02d}\n{_msg(i)}" for i in range(60)
)


def _index(svc, make_doc, name="doc.md", text=TRANSCRIPT, corpus="dedup"):
    p = make_doc(name, text)
    svc.index(str(p), corpus)
    return str(p)


def _points(svc, corpus="dedup"):
    """All stored payloads for a corpus, keyed by point id."""
    return dict(svc.store.iter_payloads(_collection_for(corpus)))


# ---------------------------------------------------------------- storage


def test_source_text_stored_once_per_source(svc, make_doc) -> None:
    src = _index(svc, make_doc)
    payloads = _points(svc)
    assert len(payloads) > 1, "need a multi-chunk doc for this to mean anything"

    carriers = [p for p in payloads.values() if p.get("source_text")]
    assert len(carriers) == 1, (
        f"expected exactly one point to carry source_text, got {len(carriers)}"
    )
    assert carriers[0]["chunk_index"] == 0
    assert carriers[0]["source_text"] == TRANSCRIPT
    assert carriers[0]["source_id"] == src


def test_source_text_carrier_is_addressable_by_deterministic_id(svc, make_doc) -> None:
    """The whole-doc readers find the carrier by recomputing its uuid5 id --
    no filter, no payload index, O(1)."""
    src = _index(svc, make_doc)
    pid = _chunk_id("dedup", src, 0)
    payload = svc.store.get_by_id(_collection_for("dedup"), pid)
    assert payload is not None
    assert payload["source_text"] == TRANSCRIPT


def test_every_chunk_keeps_bounded_context_fields(svc, make_doc) -> None:
    _index(svc, make_doc)
    for pid, p in _points(svc).items():
        assert "source_length" in p, f"{pid} lost source_length"
        assert p["source_length"] == len(TRANSCRIPT)
        assert len(p.get("context_before", "")) <= 400
        assert len(p.get("context_after", "")) <= 400
        if p["chunk_end"] > p["chunk_start"]:
            assert p.get("line_start") is not None
            assert p.get("line_end") is not None


def test_multiple_sources_each_get_one_carrier(svc, make_doc) -> None:
    _index(svc, make_doc, "a.md", TRANSCRIPT)
    _index(svc, make_doc, "b.md", TRANSCRIPT.replace("message", "note"))
    by_source: dict[str, int] = {}
    for p in _points(svc).values():
        if p.get("source_text"):
            by_source[p["source_id"]] = by_source.get(p["source_id"], 0) + 1
    assert len(by_source) == 2
    assert set(by_source.values()) == {1}


# ------------------------------------------------- per-chunk fidelity


def test_search_context_identical_to_full_text_slicing(svc, make_doc) -> None:
    """context_before/after must be byte-identical to what the old
    source_text[chunk_start-400:chunk_start] slicing produced."""
    _index(svc, make_doc)
    for p in _points(svc).values():
        cs, ce = p["chunk_start"], p["chunk_end"]
        assert p["context_before"] == TRANSCRIPT[max(0, cs - 400):cs]
        assert p["context_after"] == TRANSCRIPT[ce:ce + 400]


def test_line_anchors_match_full_text_computation(svc, make_doc) -> None:
    """1-based inclusive line range, counted at the last char INSIDE the
    span -- a chunk ending exactly on a newline must not claim the next
    line (the edge case called out at service.py:2683)."""
    _index(svc, make_doc)
    for p in _points(svc).values():
        cs, ce = p["chunk_start"], p["chunk_end"]
        if ce <= cs:
            continue
        assert p["line_start"] == TRANSCRIPT.count("\n", 0, cs) + 1
        assert p["line_end"] == TRANSCRIPT.count("\n", 0, max(cs, ce - 1)) + 1


@pytest.mark.parametrize("end_offset", [0, 1])
def test_line_end_excludes_trailing_newline(end_offset: int) -> None:
    """Direct unit check on the anchor helper, independent of chunking."""
    from vecgrep.backend.service import _line_anchors

    text = "one\ntwo\nthree\n"
    # chunk covering exactly "one\n" (ends on the newline) is still line 1.
    start, end = _line_anchors(text, 0, 4 - end_offset)
    assert (start, end) == (1, 1)


def test_search_results_still_carry_context(svc, make_doc) -> None:
    _index(svc, make_doc)
    results = svc.search("alpha filler text", "dedup", top_k=5)
    assert results
    got_context = [r for r in results if r.context_before or r.context_after]
    assert got_context, "search results lost their context windows"
    for r in results:
        assert r.chunk


# --------------------------------------------------- whole-doc readers


def test_get_source_returns_whole_document(svc, make_doc) -> None:
    src = _index(svc, make_doc)
    got = svc.get_source("dedup", src)
    assert got is not None
    assert got["text"] == TRANSCRIPT, "get_source must not be truncated"
    assert len(got["events"]) == 60, "all speaker lines must parse"


def test_browse_tail_sees_whole_document(svc, make_doc) -> None:
    """tail=N is 'the last N messages of the document'. A truncated window
    would silently return the last N of a 4KB slice instead."""
    src = _index(svc, make_doc)
    groups = svc.browse("dedup", source_path="*doc.md", tail=5)
    events = [e for g in groups for e in g["events"]]
    assert len(events) == 5
    assert [e["text"] for e in events] == [_msg(i) for i in range(55, 60)]


def test_browse_full_document_event_count(svc, make_doc) -> None:
    _index(svc, make_doc)
    groups = svc.browse("dedup", source_path="*doc.md")
    events = [e for g in groups for e in g["events"]]
    assert len(events) == 60


def test_build_timeline_slice_matches_absolute_offsets(svc, make_doc) -> None:
    _index(svc, make_doc)
    groups = svc.timeline("alpha filler", "dedup")
    assert groups
    for g in groups:
        lo, hi = g["slice_start"], g["slice_end"]
        assert 0 <= lo < hi <= len(TRANSCRIPT)
        # events in the group must actually come from that span
        parsed = [e["text"] for e in g["events"]]
        span = TRANSCRIPT[lo:hi]
        for t in parsed:
            assert t in span


def test_get_chunk_window_bounded_and_full(svc, make_doc) -> None:
    src = _index(svc, make_doc)
    payloads = _points(svc)
    # pick a chunk in the middle so both sides have room
    pid = sorted(payloads, key=lambda k: payloads[k]["chunk_index"])[
        len(payloads) // 2
    ]
    p = payloads[pid]
    cs, ce = p["chunk_start"], p["chunk_end"]

    w = svc.get_chunk_window("dedup", pid, 300)
    assert w is not None
    assert w["source_length"] == len(TRANSCRIPT)
    assert w["before"] == TRANSCRIPT[max(0, cs - 300):cs]
    assert w["after"] == TRANSCRIPT[ce:min(len(TRANSCRIPT), ce + 300)]

    full = svc.get_chunk_window("dedup", pid, -1)
    assert full is not None
    assert full["before"] == TRANSCRIPT[:cs]
    assert full["after"] == TRANSCRIPT[ce:]
    # The stored chunk text is whitespace-normalized by the chunker, so it is
    # not byte-equal to the span it came from. The offsets still have to
    # reconstruct the document exactly.
    assert full["before"] + TRANSCRIPT[cs:ce] + full["after"] == TRANSCRIPT


def test_get_chunk_window_beyond_stored_context(svc, make_doc) -> None:
    """A window wider than the stored 400-char context must still be exact --
    it has to reach the chunk-0 carrier, not silently clamp."""
    _index(svc, make_doc)
    payloads = _points(svc)
    pid = sorted(payloads, key=lambda k: payloads[k]["chunk_index"])[
        len(payloads) // 2
    ]
    p = payloads[pid]
    cs = p["chunk_start"]
    w = svc.get_chunk_window("dedup", pid, 2000)
    assert w is not None
    assert w["before"] == TRANSCRIPT[max(0, cs - 2000):cs]
    assert len(w["before"]) > 400, "fixture must exercise the >400 path"


# ------------------------------------------- pre-v1.1 payload compatibility


def test_legacy_payload_still_renders_context() -> None:
    """A store written before v1.1 has source_text on every chunk and none
    of the derived fields. The search path must fall back to slicing it.

    Absence has to be None, not "": an empty context_before is legitimate
    for the first chunk of a document, so coercing absent->"" silently
    disables the fallback and every hit renders with no context.
    """
    from vecgrep.backend.service import _payload_to_result

    doc = "alpha beta gamma\ndelta epsilon zeta\neta theta"
    legacy = {
        "corpus": "c",
        "source_id": "/tmp/x.md",
        "source_text": doc,
        "chunk_index": 1,
        "chunk_start": 17,
        "chunk_end": 35,
        "text": doc[17:35],
    }
    r = _payload_to_result(legacy, score=1.0, pct=99.0, matched_by=["vector"])
    assert r.context_before == doc[:17]
    assert r.context_after == doc[35:]
    assert (r.line_start, r.line_end) == (2, 2)


def test_stored_hit_without_context_reports_absence_not_empty() -> None:
    """StoredHit must distinguish 'not stored' from 'stored and empty', or
    _payload_to_result cannot tell whether to fall back."""
    from vecgrep.backend.service import _hit_payload
    from vecgrep.backend.store.qdrant_store import StoredHit

    hit = StoredHit(
        score=1.0, chunk_text="x", source_id="/tmp/x.md", source_text="",
        chunk_start=0, chunk_end=1, chunk_index=3, metadata={}, corpus="c",
    )
    payload = _hit_payload(hit)
    assert payload["context_before"] is None
    assert payload["context_after"] is None


def test_first_chunk_empty_context_is_preserved(svc, make_doc) -> None:
    """chunk 0 genuinely has no text before it -- that empty string must
    survive as an empty string, not trigger a legacy re-derive."""
    _index(svc, make_doc)
    first = [p for p in _points(svc).values() if p["chunk_index"] == 0][0]
    assert first["context_before"] == ""
    r = svc.search("message number 0 about beta", "dedup", top_k=5)
    assert r


# ------------------------------------------------------------ bm25 path


def test_bm25_payload_without_source_text_still_yields_context(svc, make_doc) -> None:
    """BM25-mode results build context from the stored per-chunk fields;
    they must not depend on source_text being present."""
    _index(svc, make_doc)
    results = svc.search("alpha filler text", "dedup", top_k=5, mode="bm25")
    assert results
    assert any(r.context_before or r.context_after for r in results)


def test_recovery_rebuild_preserves_source_hashes(svc, make_doc) -> None:
    """Recovery hashes each source from its full text. With one carrier per
    source that still has to resolve -- a hash over a truncated doc would
    make every source look changed and force a full re-index."""
    src = _index(svc, make_doc)
    import hashlib

    expected = hashlib.sha256(TRANSCRIPT.encode("utf-8")).hexdigest()
    svc._recover_corpus_locked("dedup")
    corpus = svc.registry.get("dedup")
    assert corpus.source_hashes.get(src) == expected


# ------------------------------------------------------------ migration


def test_migration_strips_and_backfills(svc, make_doc) -> None:
    """Given a collection in the OLD shape (source_text on every point, no
    context fields), migration must land it in the new shape with fields
    identical to what a fresh index produces."""
    from vecgrep.backend.migrate_payload import migrate_collection

    _index(svc, make_doc)
    collection = _collection_for("dedup")
    fresh = _points(svc)

    # Rewind to the old shape.
    for pid, p in fresh.items():
        svc.store.set_payload(
            collection, pid,
            {"source_text": TRANSCRIPT},
        )
        svc.store.clear_payload_keys(
            collection, pid,
            ["context_before", "context_after", "line_start",
             "line_end", "source_length"],
        )

    stats = migrate_collection(svc.store, collection)
    assert stats["stripped"] == len(fresh) - 1
    assert stats["backfilled"] == len(fresh)

    after = _points(svc)
    for pid, p in after.items():
        old = fresh[pid]
        assert p.get("context_before", "") == old["context_before"]
        assert p.get("context_after", "") == old["context_after"]
        assert p.get("line_start") == old.get("line_start")
        assert p.get("line_end") == old.get("line_end")
        assert p["source_length"] == old["source_length"]
        if p["chunk_index"] == 0:
            assert p["source_text"] == TRANSCRIPT
        else:
            assert not p.get("source_text")


def test_migration_idempotent(svc, make_doc) -> None:
    from vecgrep.backend.migrate_payload import migrate_collection

    _index(svc, make_doc)
    collection = _collection_for("dedup")
    migrate_collection(svc.store, collection)
    before = _points(svc)
    stats = migrate_collection(svc.store, collection)
    assert stats["stripped"] == 0
    assert stats["backfilled"] == 0
    assert _points(svc) == before


def test_migration_leaves_readers_working(svc, make_doc) -> None:
    from vecgrep.backend.migrate_payload import migrate_collection

    src = _index(svc, make_doc)
    migrate_collection(svc.store, _collection_for("dedup"))
    assert svc.get_source("dedup", src)["text"] == TRANSCRIPT
    events = [e for g in svc.browse("dedup", source_path="*doc.md", tail=5)
              for e in g["events"]]
    assert len(events) == 5
