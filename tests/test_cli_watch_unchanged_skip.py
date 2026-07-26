"""`vecgrep watch` must not re-index a file whose content hasn't changed.

Incident 2026-07-26: a watch-mode indexer sat in a re-index loop —
the same "2 change(s)" every ~15s, 501 iterations in 6 hours, never
converging. It peaked at 1.4GB RSS and pushed 2.0GB into swap. Because the
unit sets MemoryHigh (not just MemoryMax), systemd THROTTLED and swapped it
instead of killing it, so one wedged process swapped continuously while
everything else fought for the disk: load 143 on 12 cores, 140MB free, and
the whole VM became unreachable.

Root cause: the watcher fired on filesystem events and dispatched a full
index call per event. The content-hash check that decides "nothing to do"
lived on the far side of that call (server-side, or inside the service), so
an unchanged file still cost a full HTTP round-trip plus re-read every time
its mtime was touched. A recorder that rewrites files with identical content
therefore produced unbounded work forever.

Contract: hash first, dispatch second. An event on a byte-identical file
must cost nothing beyond a local re-hash.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from vecgrep.cli import main as m


def test_unchanged_file_is_not_dispatched(tmp_path: Path) -> None:
    """Second event on identical content must not call the index path."""
    f = tmp_path / "a.md"
    f.write_text("hello world", encoding="utf-8")

    calls: list[str] = []
    with patch.object(m, "_do_index", side_effect=lambda s, *a, **k: calls.append(s)):
        assert m._watch_should_index(str(f)) is True
        m._do_index(str(f), "c", "auto", force=False)
        # Same bytes, new event (mtime touched, content identical).
        assert m._watch_should_index(str(f)) is False, (
            "an unchanged file must be skipped BEFORE the index dispatch — "
            "that skip is what stops the runaway re-index loop")
    assert calls == [str(f)], calls


def test_changed_content_still_indexes(tmp_path: Path) -> None:
    """A real edit must still get through — the guard is a skip, not a block."""
    f = tmp_path / "a.md"
    f.write_text("v1", encoding="utf-8")
    assert m._watch_should_index(str(f)) is True
    f.write_text("v2 — genuinely different", encoding="utf-8")
    assert m._watch_should_index(str(f)) is True, "content changed → must index"


def test_unreadable_file_defaults_to_indexing(tmp_path: Path) -> None:
    """If we can't hash it, fall through to the old behaviour rather than
    silently dropping a real change."""
    assert m._watch_should_index(str(tmp_path / "does-not-exist.md")) is True
