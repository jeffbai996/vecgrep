from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from vecgrep.backend.service import (
    DEFAULT_MODE,
    SEARCH_MODES,
    SearchResult,
    _passes_filters,
    _validate_search_mode,
)
from vecgrep.backend.store import CorpusError
from vecgrep.backend.store.corpora import (
    CorpusError as CorpusError_search_mode_validation,
)


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

# ── test_filter_schema ──

def test_reports_source_and_corpus_always(svc, make_doc):
    p = make_doc("d.md", "plain content here. another sentence follows.")
    svc.index(str(p), "c")
    schema = svc.filterable_fields("c")
    assert schema["corpus"] == "c"
    assert "source" in schema["filters"]
    assert "corpus" in schema["filters"]
    assert schema["filters"]["source"]["form"] == "source:GLOB"


def test_discovers_meta_keys_and_values(svc, make_doc):
    """Markdown adapter records adapter+path metadata; those keys show up."""
    p = make_doc("d.md", "content. more content.")
    svc.index(str(p), "c")
    schema = svc.filterable_fields("c")
    meta_keys = schema["filters"]["meta"]["keys"]
    # The markdown adapter writes {"adapter": "markdown", "path": ...}.
    assert "adapter" in meta_keys
    assert "markdown" in meta_keys["adapter"]


def test_reports_doc_timestamp_presence(svc, make_doc):
    dated = make_doc("d.md", "Saved: 2026-05-18T00:00:00+00:00\n\nbody content.")
    svc.index(str(dated), "c")
    schema = svc.filterable_fields("c")
    assert schema["has_doc_timestamp"] is True


def test_unknown_corpus_raises(svc):
    with pytest.raises(CorpusError):
        svc.filterable_fields("nonexistent")


def test_schema_is_json_serializable(svc, make_doc):
    """get_corpus serializes this to JSON — sets must be converted to lists."""
    p = make_doc("d.md", "content here.")
    svc.index(str(p), "c")
    schema = svc.filterable_fields("c")
    json.dumps(schema)  # must not raise (no raw sets)

# ── test_search_mode_validation ──

def test_known_modes_are_the_literal_members() -> None:
    assert SEARCH_MODES == {"hybrid", "vector", "bm25"}
    assert DEFAULT_MODE in SEARCH_MODES


@pytest.mark.parametrize("mode", sorted(SEARCH_MODES))
def test_known_modes_pass(mode: str) -> None:
    _validate_search_mode(mode)  # must not raise


@pytest.mark.parametrize("mode", ["dense", "semantic", "embedding", "HYBRID", "", "  "])
def test_unknown_modes_raise(mode: str) -> None:
    with pytest.raises(CorpusError_search_mode_validation) as e:
        _validate_search_mode(mode)
    # The message must name the valid set — an agent that guessed "dense"
    # needs to learn what to guess next without reading the source.
    for known in SEARCH_MODES:
        assert known in str(e.value)


def test_search_rejects_unknown_mode_before_touching_a_corpus(svc, make_doc) -> None:
    p = make_doc("doc.md", "alpha content here. beta content here.")
    svc.index(str(p), "test")

    with pytest.raises(CorpusError_search_mode_validation):
        svc.search("alpha", "test", mode="dense")


def test_search_rejects_unknown_mode_with_no_corpora(svc) -> None:
    """Validation is at the entry point, so it fires even on an empty install
    — the old code returned [] here for a *valid* reason and [] for an invalid
    mode, which is exactly the ambiguity being removed."""
    with pytest.raises(CorpusError_search_mode_validation):
        svc.search("anything", mode="dense")


def test_valid_mode_still_searches(svc, make_doc) -> None:
    p = make_doc("doc.md", "alpha content here. beta content here.")
    svc.index(str(p), "test")
    assert svc.search("alpha", "test", mode="bm25")

# ── test_write_metadata_readpath ──

class _TestSearchResult(SearchResult):
    text: str


def _result(text="t", metadata=None, corpus="c", source_id="s"):
    r = _TestSearchResult.__new__(_TestSearchResult)
    r.corpus = corpus
    r.source_id = source_id
    r.text = text
    r.metadata = metadata or {}
    r.score = 1.0
    r.matched_by = ["vector"]
    r.explain = {}
    return r


# --- default status:active filtering (the one real behavior change) ---

class TestDefaultActiveFilter:
    def test_active_chunk_passes_default(self):
        r = _result(metadata={"status": "active"})
        assert _passes_filters(r, ["meta.status=active"]) is True

    def test_superseded_chunk_blocked_by_active_filter(self):
        r = _result(metadata={"status": "superseded"})
        assert _passes_filters(r, ["meta.status=active"]) is False

    def test_legacy_chunk_without_status_treated_as_active(self):
        # Back-compat: a chunk indexed before the schema has no status key.
        # It must NOT vanish under a default status:active filter.
        r = _result(metadata={})
        assert _passes_filters(r, ["meta.status=active"], default_active=True) is True

    def test_legacy_chunk_blocked_only_under_strict_match(self):
        # Without the back-compat default, a literal meta.status=active match
        # excludes a chunk that has no status (the pre-existing strict behavior).
        r = _result(metadata={})
        assert _passes_filters(r, ["meta.status=active"]) is False


# --- source_kind / tier filters (already meta.* keys; pin they work) ---

class TestSchemaFieldFilters:
    def test_filter_by_source_kind(self):
        decision = _result(metadata={"source_kind": "decision"})
        journal = _result(metadata={"source_kind": "journal"})
        assert _passes_filters(decision, ["meta.source_kind=decision"]) is True
        assert _passes_filters(journal, ["meta.source_kind=decision"]) is False

    def test_filter_by_tier(self):
        protected = _result(metadata={"tier": "protected"})
        normal = _result(metadata={"tier": "normal"})
        assert _passes_filters(protected, ["meta.tier=protected"]) is True
        assert _passes_filters(normal, ["meta.tier=protected"]) is False

    def test_filters_and_together(self):
        r = _result(metadata={"status": "active", "tier": "protected"})
        assert _passes_filters(r, ["meta.status=active", "meta.tier=protected"]) is True
        assert _passes_filters(r, ["meta.status=active", "meta.tier=normal"]) is False
