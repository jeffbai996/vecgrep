"""Phase 3 — date / path / time filters as hard constraints.

The caller (an AI assistant) passes explicit filters; vecgrep pushes them as
hard constraints so a "this morning" question stops dragging in last week's
lore. No natural-language inference here — that's the assistant's job.

Forms added to the existing filter grammar (source:/corpus:/meta.K=V):

    date:YYYY-MM-DD      — doc_timestamp inside that UTC day
    after:<iso>          — doc_timestamp >= <iso> (date or datetime)
    before:<iso>         — doc_timestamp <  <iso>
    channel:<name>       — metadata channel match (quote-tolerant)
    source_path:<glob>   — fnmatch on source_id (alias of source:)

Hard-constraint semantics: chunks with NO doc_timestamp fail any time filter
(if you asked for a date, undated evidence doesn't qualify), and an
unparseable time value matches NOTHING — failing closed makes a typo'd date
visible as zero results instead of silently leaking the old lore back in.
"""
from __future__ import annotations

from datetime import datetime, timezone

from vecgrep.backend.service import SearchResult, _passes_filters


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp()


def _hit(
    doc_timestamp: float | None,
    source_id: str = "channel-a/2026-01-15.md",
    metadata: dict | None = None,
) -> SearchResult:
    return SearchResult(
        score=0.5,
        similarity_pct=50.0,
        chunk="text",
        chunk_start=0,
        chunk_end=100,
        context_before="",
        context_after="",
        source_id=source_id,
        corpus="c",
        metadata=metadata or {},
        chunk_id="cid",
        matched_by=["vector"],
        doc_timestamp=doc_timestamp,
    )


# ── date: ────────────────────────────────────────────────────────────────────

def test_date_filter_keeps_same_day() -> None:
    r = _hit(_ts("2026-01-15T08:30"))
    assert _passes_filters(r, ["date:2026-01-15"])


def test_date_filter_drops_other_days() -> None:
    assert not _passes_filters(_hit(_ts("2026-01-08T14:00")), ["date:2026-01-15"])
    assert not _passes_filters(_hit(_ts("2026-01-16T00:00")), ["date:2026-01-15"])


def test_date_filter_day_boundaries_utc() -> None:
    assert _passes_filters(_hit(_ts("2026-01-15T00:00")), ["date:2026-01-15"])
    assert _passes_filters(_hit(_ts("2026-01-15T23:59:59")), ["date:2026-01-15"])


def test_date_filter_excludes_undated() -> None:
    # Hard constraint: undated evidence doesn't qualify for a dated ask.
    assert not _passes_filters(_hit(None), ["date:2026-01-15"])


def test_unparseable_date_fails_closed() -> None:
    # A typo'd date must surface as 0 results, not silently leak everything.
    assert not _passes_filters(_hit(_ts("2026-01-15T08:00")), ["date:jan-15th"])


# ── after: / before: ─────────────────────────────────────────────────────────

def test_after_inclusive_before_exclusive() -> None:
    r = _hit(_ts("2026-01-15T08:00"))
    assert _passes_filters(r, ["after:2026-01-15"])          # 00:00 <= 08:00
    assert _passes_filters(r, ["after:2026-01-15T08:00"])    # boundary in
    assert not _passes_filters(r, ["before:2026-01-15T08:00"])  # boundary out
    assert _passes_filters(r, ["before:2026-01-15T08:01"])


def test_after_and_before_combine_to_window() -> None:
    filters = ["after:2026-01-15T06:00", "before:2026-01-15T12:00"]
    assert _passes_filters(_hit(_ts("2026-01-15T08:00")), filters)
    assert not _passes_filters(_hit(_ts("2026-01-15T14:00")), filters)
    assert not _passes_filters(_hit(_ts("2026-01-08T08:00")), filters)


def test_time_filters_exclude_undated() -> None:
    assert not _passes_filters(_hit(None), ["after:2026-01-01"])
    assert not _passes_filters(_hit(None), ["before:2027-01-01"])


# ── channel: ─────────────────────────────────────────────────────────────────

def test_channel_filter_matches_metadata() -> None:
    r = _hit(_ts("2026-01-15T08:00"), metadata={"channel": "channel-b"})
    assert _passes_filters(r, ["channel:channel-b"])
    assert not _passes_filters(r, ["channel:channel-a"])


def test_channel_filter_tolerates_quoted_metadata() -> None:
    # Real archiver frontmatter is `channel: "cl-9"` — the raw value keeps its
    # quotes through the tiny frontmatter parser. The filter must still match.
    r = _hit(_ts("2026-01-15T08:00"), metadata={"channel": '"channel-b"'})
    assert _passes_filters(r, ["channel:channel-b"])


def test_channel_filter_missing_metadata_fails() -> None:
    assert not _passes_filters(_hit(_ts("2026-01-15T08:00")), ["channel:channel-b"])


# ── source_path: ─────────────────────────────────────────────────────────────

def test_source_path_glob() -> None:
    r = _hit(_ts("2026-01-15T08:00"), source_id="/data/channel-a/2026-01-15.md")
    assert _passes_filters(r, ["source_path:*channel-a/2026-01-15.md"])
    assert not _passes_filters(r, ["source_path:*channel-b*"])


# ── end-to-end on the eval fixtures: the over-eagerness fix ─────────────────

def test_date_filter_excludes_week_old_lore(svc) -> None:
    """The golden 'incident_today' trap: same vocabulary a week apart. With
    date:2026-01-15 the 2026-01-08 file must vanish entirely."""
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    hits = svc.search(
        "why did relay-service fail today?", "evalchat",
        top_k=10, filters=["date:2026-01-15"],
    )
    assert hits, "must still find the incident"
    assert all("2026-01-08" not in h.source_id for h in hits), "old lore leaked"


def test_channel_filter_isolates_postmortem(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    hits = svc.search(
        "relay-service ERR-0451", "evalchat",
        top_k=10, filters=["channel:channel-b"],
    )
    assert hits
    assert all("channel-b" in h.source_id for h in hits)


def test_filters_compose_with_budget(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    full, stubs = svc.search_budgeted(
        "relay-service", "evalchat", filters=["date:2026-01-15"],
    )
    for r in full:
        assert "2026-01-08" not in r.source_id
    for s in stubs:
        assert "2026-01-08" not in s.source_id
