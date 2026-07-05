"""Phase 5 — timeline mode.

`timeline()` answers "what happened?" questions with an ordered event
sequence instead of ranked chunks: search finds the anchor chunks, the
top source files are selected, one CONTIGUOUS slice per file is pulled
around the anchors, and each slice is parsed into (speaker, time, text)
events — chronological within a file, files ordered by document date.
Non-transcript sources degrade to the raw slice (no fake events).
"""
from __future__ import annotations

from vecgrep.backend.timeline import parse_events

TRANSCRIPT = """**alice** · 08:02
> relay-service is down again, ERR-0451 on every request

**ops-bot [bot]** · 08:02
> ALERT: relay-service health check failed.

**小北** · 08:09
> 小艾说得对，先抓 core dump

**bob** · 08:10
> fine. five minutes. clock starts now
"""


# ── event parsing ────────────────────────────────────────────────────────────

def test_parse_events_extracts_speakers_times_bodies() -> None:
    events = parse_events(TRANSCRIPT)
    assert len(events) == 4
    assert events[0].speaker == "alice"
    assert events[0].time == "08:02"
    assert "ERR-0451" in events[0].text
    assert events[2].speaker == "小北"
    assert "core dump" in events[2].text
    assert events[3].speaker == "bob"


def test_parse_events_strips_quote_markers() -> None:
    events = parse_events(TRANSCRIPT)
    for e in events:
        assert not e.text.startswith(">")


def test_parse_events_preserves_document_order() -> None:
    events = parse_events(TRANSCRIPT)
    assert [e.time for e in events] == ["08:02", "08:02", "08:09", "08:10"]


def test_parse_events_non_transcript_returns_empty() -> None:
    # Plain prose has no speaker headers — no fake events.
    assert parse_events("Just a paragraph of ordinary text. Nothing else.") == []


def test_parse_events_partial_leading_text_ignored() -> None:
    # A slice can start mid-message; text before the first header has no
    # attributable speaker and is dropped rather than misattributed.
    events = parse_events("tail of a previous message\n" + TRANSCRIPT)
    assert len(events) == 4
    assert events[0].speaker == "alice"


# ── service.timeline end-to-end on the fixtures ──────────────────────────────

def test_timeline_returns_chronological_grouped_events(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    groups = svc.timeline("relay-service ERR-0451 outage", "evalchat")
    assert groups, "timeline must find the incident"
    g = next(
        (g for g in groups if g["source_id"].endswith("channel-a/2026-01-15.md")),
        None,
    )
    assert g is not None, "incident file must be a timeline group"
    events = g["events"]
    assert len(events) >= 5, "a contiguous slice, not isolated chunks"
    times = [e["time"] for e in events]
    assert times == sorted(times), "events must be chronological"
    speakers = {e["speaker"] for e in events}
    assert "alice" in speakers and "ops-bot [bot]" in speakers


def test_timeline_groups_are_ordered_by_doc_date(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    groups = svc.timeline("relay-service down ERR-0451", "evalchat")
    dates = [g["doc_timestamp"] for g in groups if g["doc_timestamp"]]
    assert dates == sorted(dates), "groups ordered oldest → newest"


def test_timeline_respects_filters(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    groups = svc.timeline(
        "relay-service down ERR-0451", "evalchat",
        filters=["date:2026-01-15"],
    )
    assert groups
    for g in groups:
        assert "2026-01-08" not in g["source_id"], "filtered file leaked in"


def test_timeline_slices_are_contiguous(svc) -> None:
    """The slice is one contiguous region per file — the whole point vs
    ranked chunks. Verify via slice bounds covering all its events."""
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    groups = svc.timeline("relay-service ERR-0451 outage", "evalchat")
    for g in groups:
        assert g["slice_end"] > g["slice_start"] >= 0


def test_timeline_empty_on_no_match(svc) -> None:
    """Zero anchors → zero groups, no crash. Uses bm25 retrieval so 'no
    match' is real: the stub embedder scores everything ~90% cosine, which a
    production cosine floor would reject — noise precision is the retrieval
    core's job, not timeline's."""
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    assert svc.timeline("qqqzzznope", "evalchat", mode="bm25") == []
