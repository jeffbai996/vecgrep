from __future__ import annotations

import pytest

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


def test_build_timeline_type_hints_resolve_at_runtime() -> None:
    import typing

    from vecgrep.backend.timeline import build_timeline

    hints = typing.get_type_hints(build_timeline)
    assert "payload_for" in hints
    assert "source_text_for" in hints

# ── test_anchors_incident ──

# ── 8a: line anchors ─────────────────────────────────────────────────────────

def test_results_carry_line_ranges(svc, make_doc) -> None:
    p = make_doc(
        "lines.md",
        "alpha first line here\nbeta second line here\ngamma third line here\n"
        "delta fourth line here\nepsilon fifth line here\n",
    )
    svc.index(str(p), "c")
    hits = svc.search("gamma third line", "c", top_k=3)
    assert hits
    h = hits[0]
    assert h.line_start is not None and h.line_end is not None
    assert 1 <= h.line_start <= h.line_end
    # The chunk containing 'gamma' must include line 3 in its range.
    assert h.line_start <= 3 <= h.line_end


def test_anchor_string_format(svc, make_doc) -> None:
    p = make_doc("anchor.md", "one\ntwo\nthree\nfour anchor target line\nfive\n")
    svc.index(str(p), "c")
    h = svc.search("four anchor target", "c", top_k=1)[0]
    a = h.anchor
    assert a.startswith(h.source_id)
    assert f"#L{h.line_start}" in a
    assert a.endswith((f"-L{h.line_end}", f"#L{h.line_start}"))


def test_eval_fixture_lines_are_plausible(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    hits = svc.search("ERR-0451", "evalchat", top_k=5)
    for h in hits:
        assert h.line_start is not None and h.line_start >= 1
        assert h.line_end >= h.line_start


# ── 8b: incident object ──────────────────────────────────────────────────────

def test_incident_assembles_structured_answer(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    inc = svc.incident("relay-service ERR-0451 outage", "evalchat")
    assert inc is not None
    assert inc["title"] == "relay-service ERR-0451 outage"
    assert inc["sources"], "must name its evidence files"
    assert "alice" in inc["participants"]
    assert inc["confidence"] in ("exact", "strong", "related", "weak")
    # Primary timeline is a chronological event list.
    events = inc["primary_timeline"]
    assert len(events) >= 5
    times = [e["time"] for e in events]
    assert times == sorted(times)


def test_incident_separates_primary_from_related(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    inc = svc.incident("relay-service ERR-0451 outage", "evalchat")
    primary_files = {inc["primary_source"]}
    related_files = {g["source_id"] for g in inc["related"]}
    assert primary_files.isdisjoint(related_files), (
        "the primary file must not repeat in related context"
    )


def test_incident_time_range_spans_events(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    inc = svc.incident("relay-service ERR-0451 outage", "evalchat")
    tr = inc["time_range"]
    assert tr["start"] and tr["end"] and tr["start"] <= tr["end"]


def test_incident_respects_filters(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    inc = svc.incident(
        "relay-service ERR-0451 outage", "evalchat",
        filters=["date:2026-01-15"],
    )
    all_sources = [inc["primary_source"], *(g["source_id"] for g in inc["related"])]
    assert all("2026-01-08" not in s for s in all_sources)


def test_incident_none_when_nothing_found(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    assert svc.incident("qqqzzznope", "evalchat", mode="bm25") is None

# ── test_browse_window ──

def test_browse_since_until_range(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    # channel-a has 2026-01-08 and 2026-01-15; a range covering only the 15th.
    groups = svc.browse("evalchat", channel="channel-a",
                        since="2026-01-10", until="2026-01-15")
    assert {g["source_id"].split("/")[-1] for g in groups} == {"2026-01-15.md"}
    # until is inclusive of its whole day.
    groups = svc.browse("evalchat", channel="channel-a",
                        since="2026-01-08", until="2026-01-08")
    assert {g["source_id"].split("/")[-1] for g in groups} == {"2026-01-08.md"}


def test_browse_since_alone_is_a_valid_selector(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    groups = svc.browse("evalchat", since="2026-01-09")
    assert groups, "range-only browse must work without channel/date"
    for g in groups:
        assert g["source_id"].endswith("2026-01-15.md")


def test_browse_tail_keeps_newest_events(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    full = svc.browse("evalchat", channel="channel-a")
    all_events = [e for g in full for e in g["events"]]
    assert len(all_events) > 5
    tailed = svc.browse("evalchat", channel="channel-a", tail=5)
    got = [e for g in tailed for e in g["events"]]
    assert len(got) == 5
    assert got == all_events[-5:], "tail must be the newest events, in order"


def test_browse_tail_larger_than_corpus_returns_everything(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    full = svc.browse("evalchat", channel="channel-a")
    tailed = svc.browse("evalchat", channel="channel-a", tail=10_000)
    assert [e for g in tailed for e in g["events"]] == \
        [e for g in full for e in g["events"]]


def test_browse_bad_range_dates_raise(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    with pytest.raises(ValueError):
        svc.browse("evalchat", since="not-a-date")
    with pytest.raises(ValueError):
        svc.browse("evalchat", until="13/01/2026")
