"""`vecgrep watch` must not index a file a live session is still writing.

Incident 2026-07-27: a session recorder APPENDS to its transcript file every
turn an agent takes. Each append genuinely changes the content,
so the byte-identical guard (test_cli_watch_unchanged_skip) passes it — and
the watcher re-embeds the ENTIRE 100-150 chunk file, taking 5-9 minutes, by
which time the file has changed again. It never converges. The perpetual
qdrant write volume kept its optimizer threads in D state, churned the page
cache, and pushed the box to 93% swap and ~23% sustained IO stall.

Contract: a file whose mtime is inside the quiet period is DEFERRED, not
indexed — parked in a pending set and swept in once the file has gone quiet.
Deferred is not dropped: the last write must always end up indexed, even
though no filesystem event fires after the session stops.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from vecgrep.cli import main as m


def _aged(path: Path, seconds_ago: float) -> str:
    past = time.time() - seconds_ago
    os.utime(path, (past, past))
    return str(path)


def test_freshly_written_file_is_not_quiet(tmp_path: Path) -> None:
    f = tmp_path / "live.md"
    f.write_text("a live session is appending here", encoding="utf-8")
    assert m._watch_is_quiet(str(f), quiet_period=300) is False, (
        "a file modified seconds ago belongs to a live session — indexing it "
        "now is the re-index treadmill")


def test_old_file_is_quiet(tmp_path: Path) -> None:
    f = tmp_path / "settled.md"
    f.write_text("session over", encoding="utf-8")
    assert m._watch_is_quiet(_aged(f, 20 * 60), quiet_period=300) is True


def test_zero_quiet_period_disables_the_guard(tmp_path: Path) -> None:
    f = tmp_path / "live.md"
    f.write_text("fresh", encoding="utf-8")
    assert m._watch_is_quiet(str(f), quiet_period=0) is True


def test_missing_file_counts_as_quiet(tmp_path: Path) -> None:
    """Can't stat it → let the downstream path handle (and report) it rather
    than parking it in pending forever."""
    assert m._watch_is_quiet(str(tmp_path / "gone.md"), quiet_period=300) is True


def test_deferred_file_becomes_due_after_going_quiet(tmp_path: Path) -> None:
    f = tmp_path / "transcript.md"
    f.write_text("turn 1", encoding="utf-8")

    m._WATCH_PENDING.clear()
    m._WATCH_PENDING.add(str(f))
    assert m._watch_due_pending(quiet_period=300) == [], (
        "still hot — must stay parked")
    assert str(f) in m._WATCH_PENDING

    _aged(f, 20 * 60)
    assert m._watch_due_pending(quiet_period=300) == [str(f)], (
        "quiet now — the deferred write must surface for indexing")
    assert str(f) not in m._WATCH_PENDING, "due files leave the pending set"


def test_deleted_file_leaves_the_pending_set(tmp_path: Path) -> None:
    """A deferred file that gets deleted must not resurface for indexing."""
    f = tmp_path / "doomed.md"
    f.write_text("x", encoding="utf-8")
    m._WATCH_PENDING.clear()
    m._WATCH_PENDING.add(str(f))
    f.unlink()
    m._WATCH_PENDING.discard(str(f))
    assert m._watch_due_pending(quiet_period=300) == []
