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


# ── relative time forms (F-4) ────────────────────────────────────────────────
# after:7d / after:24h / before:yesterday / date:today — resolved against an
# injected `now` so tests are deterministic. ISO forms unchanged.

_NOW = _ts("2026-07-05T12:00")  # fixed reference clock for every test below


def test_after_relative_days() -> None:
    r = _hit(_ts("2026-07-01T12:00"))          # 4 days old
    assert _passes_filters(r, ["after:7d"], now=_NOW)
    assert not _passes_filters(r, ["after:3d"], now=_NOW)


def test_after_relative_hours_and_weeks() -> None:
    r = _hit(_ts("2026-07-05T09:00"))          # 3h old
    assert _passes_filters(r, ["after:24h"], now=_NOW)
    assert not _passes_filters(r, ["after:2h"], now=_NOW)
    old = _hit(_ts("2026-06-15T12:00"))        # ~3 weeks old
    assert _passes_filters(old, ["after:4w"], now=_NOW)
    assert not _passes_filters(old, ["after:2w"], now=_NOW)


def test_date_today_and_before_yesterday() -> None:
    today = _hit(_ts("2026-07-05T03:00"))
    yesterday = _hit(_ts("2026-07-04T15:00"))
    assert _passes_filters(today, ["date:today"], now=_NOW)
    assert not _passes_filters(yesterday, ["date:today"], now=_NOW)
    assert _passes_filters(yesterday, ["date:yesterday"], now=_NOW)
    assert _passes_filters(yesterday, ["before:today"], now=_NOW)
    assert not _passes_filters(today, ["before:today"], now=_NOW)


def test_relative_garbage_fails_closed() -> None:
    r = _hit(_ts("2026-07-05T03:00"))
    assert not _passes_filters(r, ["after:7x"], now=_NOW)
    assert not _passes_filters(r, ["after:someday"], now=_NOW)


def test_iso_forms_unchanged_with_now_param() -> None:
    r = _hit(_ts("2026-01-15T08:00"))
    assert _passes_filters(r, ["after:2026-01-15"], now=_NOW)


# ── negation (F-5) ───────────────────────────────────────────────────────────

def test_negated_corpus_excludes() -> None:
    r = _hit(_ts("2026-07-05T03:00"))          # corpus "c" per _hit
    assert not _passes_filters(r, ["-corpus:c"])
    assert _passes_filters(r, ["-corpus:scratch"])


def test_negated_source_glob_excludes() -> None:
    r = _hit(_ts("2026-07-05T03:00"))          # source_id "chan/2026-*.md"
    assert not _passes_filters(r, [f"-source:{r.source_id}"])
    assert _passes_filters(r, ["-source:*/other/*"])


def test_negated_channel_excludes() -> None:
    r = _hit(_ts("2026-07-05T03:00"))
    r.metadata["channel"] = "cl-3"
    assert not _passes_filters(r, ["-channel:cl-3"])
    assert _passes_filters(r, ["-channel:cl-6"])


def test_negated_time_filter_with_garbage_still_fails_closed() -> None:
    """-after:<typo> must NOT invert into match-everything — a typo'd filter
    yields zero results in either polarity."""
    r = _hit(_ts("2026-07-05T03:00"))
    assert not _passes_filters(r, ["-after:notadate"], now=_NOW)


# ── speaker: / bot: / has: filters (F-1..F-3, chunk-level) ──────────────────
# Chunk-LEVEL semantics on purpose: a chunk containing any line by NAME passes
# speaker:NAME. Not per-line attribution — that's documented, not oversold.

def _enriched(speakers=(), bot=False, code=False, table=False, link=False):
    r = _hit(_ts("2026-07-05T03:00"))
    r.metadata.update({
        "speakers": list(speakers), "has_bot_speaker": bot,
        "has_code": code, "has_table": table, "has_link": link,
    })
    return r


def test_speaker_filter_case_insensitive_membership() -> None:
    r = _enriched(speakers=["alice", "helper_bot [bot]"])
    assert _passes_filters(r, ["speaker:alice"])
    assert _passes_filters(r, ["speaker:ALICE"])
    assert not _passes_filters(r, ["speaker:mac"])


def test_speaker_filter_tolerates_bot_suffix() -> None:
    """speaker:helper_bot matches the archiver's 'helper_bot [bot]'
    — nobody types the suffix by hand."""
    r = _enriched(speakers=["helper_bot [bot]"])
    assert _passes_filters(r, ["speaker:helper_bot"])
    assert _passes_filters(r, ["author:helper_bot"])  # alias


def test_speaker_filter_fails_closed_on_unenriched_chunk() -> None:
    """Pre-enrichment chunks (no speakers key) fail a speaker: filter — hard
    constraint, same discipline as the time filters."""
    r = _hit(_ts("2026-07-05T03:00"))  # no enrichment keys at all
    assert not _passes_filters(r, ["speaker:alice"])


def test_bot_filter() -> None:
    assert _passes_filters(_enriched(bot=True), ["bot:true"])
    assert not _passes_filters(_enriched(bot=False), ["bot:true"])
    assert _passes_filters(_enriched(bot=False), ["bot:false"])
    assert not _passes_filters(_enriched(bot=True), ["bot:false"])


def test_has_filters() -> None:
    assert _passes_filters(_enriched(code=True), ["has:code"])
    assert not _passes_filters(_enriched(code=False), ["has:code"])
    assert _passes_filters(_enriched(table=True), ["has:table"])
    assert _passes_filters(_enriched(link=True), ["has:link"])
    # unknown shape name fails closed (recognized form, bad value)
    assert not _passes_filters(_enriched(code=True), ["has:gifs"])


def test_negated_speaker_and_has() -> None:
    r = _enriched(speakers=["alice"], code=True)
    assert not _passes_filters(r, ["-speaker:alice"])
    assert not _passes_filters(r, ["-has:code"])
    assert _passes_filters(r, ["-speaker:mac", "-has:table"])
