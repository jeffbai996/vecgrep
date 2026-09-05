"""Source labels — stamp search hits with a caller-defined origin label.

Chat archives freeze whatever display name a bot had when each message
posted, and display names drift; the stable identity anchor is the source
PATH (a channel directory, a per-host archive root). The map is
DEPLOYMENT-SPECIFIC data and lives outside the repo — default
`$VECGREP_HOME/source_labels.json`, overridable via
`VECGREP_SOURCE_LABELS_FILE`. vecgrep ships only a generic example.
No map = exact no-op.

Map format (flat, ordered): {"*/transcripts/team-a/*": "alice@host-1", ...}
— key is an fnmatch glob over source_id, value is the label stamped onto
matching hits. First matching glob wins.
"""
from __future__ import annotations

import json
from pathlib import Path

from vecgrep.backend.source_labels import (
    apply_labels,
    label_for,
    load_source_labels,
    load_source_labels_cached,
)

MAP = {
    "*/transcripts/team-a/*": "alice@host-1",
    "*/transcripts/*": "shared",
}


def _result(source_id: str):
    from vecgrep.backend.service import SearchResult

    return SearchResult(
        score=1.0,
        similarity_pct=90.0,
        chunk="x",
        chunk_start=0,
        chunk_end=1,
        context_before="",
        context_after="",
        source_id=source_id,
        corpus="chats",
        metadata={},
        chunk_id="c1",
        matched_by=["bm25"],
    )


# ── loading ──────────────────────────────────────────────────────────────────

def test_load_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_source_labels(tmp_path / "nope.json") == {}


def test_load_valid_map_preserves_order(tmp_path: Path) -> None:
    p = tmp_path / "source_labels.json"
    p.write_text(json.dumps(MAP), encoding="utf-8")
    m = load_source_labels(p)
    assert list(m.items()) == list(MAP.items())


def test_load_corrupt_file_is_empty_not_crash(tmp_path: Path) -> None:
    p = tmp_path / "source_labels.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_source_labels(p) == {}


def test_load_rejects_non_string_entries(tmp_path: Path) -> None:
    p = tmp_path / "source_labels.json"
    p.write_text(
        json.dumps({"*/ok/*": "kept", "*/bad/*": 7, "*/worse/*": ["list"]}),
        encoding="utf-8",
    )
    assert load_source_labels(p) == {"*/ok/*": "kept"}


def test_cached_load_tracks_mtime(tmp_path: Path) -> None:
    import os

    p = tmp_path / "source_labels.json"
    p.write_text(json.dumps({"*/a/*": "one"}), encoding="utf-8")
    assert load_source_labels_cached(p) == {"*/a/*": "one"}
    p.write_text(json.dumps({"*/a/*": "two"}), encoding="utf-8")
    os.utime(p, (0, 12345))  # force a distinct mtime on coarse filesystems
    assert load_source_labels_cached(p) == {"*/a/*": "two"}


# ── matching ─────────────────────────────────────────────────────────────────

def test_first_matching_glob_wins() -> None:
    sid = "/data/transcripts/team-a/2026-08-28.md"
    assert label_for(sid, MAP) == "alice@host-1"


def test_fallback_glob_matches_when_specific_does_not() -> None:
    sid = "/data/transcripts/team-b/2026-08-28.md"
    assert label_for(sid, MAP) == "shared"


def test_no_match_is_none() -> None:
    assert label_for("/somewhere/else.md", MAP) is None


def test_empty_map_is_none() -> None:
    assert label_for("/data/transcripts/team-a/x.md", {}) is None


# ── application to results ───────────────────────────────────────────────────

def test_apply_labels_stamps_matching_hits() -> None:
    hits = [_result("/data/transcripts/team-a/x.md"), _result("/elsewhere/y.md")]
    apply_labels(hits, MAP)
    assert hits[0].source_label == "alice@host-1"
    assert hits[1].source_label is None


def test_apply_labels_empty_map_is_noop() -> None:
    hits = [_result("/data/transcripts/team-a/x.md")]
    apply_labels(hits, {})
    assert hits[0].source_label is None
