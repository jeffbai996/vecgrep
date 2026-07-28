"""A transient backend failure must not kill the watcher.

Incident 2026-07-27: an embed backend under load raised httpx.ReadTimeout
during the watcher's initial pass. Only ClickException was caught, so the
process died, systemd restarted it, and the initial pass began again from
zero — under the same load that caused the timeout. The service had
NRestarts=90 from an earlier round of exactly this. A daemon that re-does
its most expensive work on every transient failure is its own treadmill.

Contract: per-file indexing never raises past the guard (except
KeyboardInterrupt); a failure is logged and reported so the loop — and the
box — keeps breathing.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import click
import pytest

from vecgrep.cli import main as m


def _file(tmp_path: Path) -> str:
    f = tmp_path / "a.md"
    f.write_text("content", encoding="utf-8")
    return str(f)


def test_transient_backend_error_is_swallowed_and_reported(tmp_path: Path) -> None:
    p = _file(tmp_path)
    m._WATCH_SEEN_HASHES.clear()
    with patch.object(m, "_do_index", side_effect=TimeoutError("backend slow")):
        assert m._watch_index_resilient(p, "c", "auto", None) is False


def test_click_exception_is_swallowed_too(tmp_path: Path) -> None:
    p = _file(tmp_path)
    m._WATCH_SEEN_HASHES.clear()
    with patch.object(m, "_do_index",
                      side_effect=click.ClickException("bad file")):
        assert m._watch_index_resilient(p, "c", "auto", None) is False


def test_keyboard_interrupt_still_propagates(tmp_path: Path) -> None:
    """Ctrl+C must stop the watcher — the guard is for failures, not intent."""
    p = _file(tmp_path)
    m._WATCH_SEEN_HASHES.clear()
    with patch.object(m, "_do_index", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            m._watch_index_resilient(p, "c", "auto", None)


def test_successful_index_reports_clean(tmp_path: Path) -> None:
    p = _file(tmp_path)
    m._WATCH_SEEN_HASHES.clear()
    with patch.object(m, "_do_index"):
        assert m._watch_index_resilient(p, "c", "auto", None) is True


def test_a_failed_file_is_retried_on_its_next_event(tmp_path: Path) -> None:
    """The failure path must not poison the seen-hash cache: the file's next
    event has identical content, and a hash recorded before a FAILED dispatch
    would make the skip-guard drop it forever."""
    p = _file(tmp_path)
    m._WATCH_SEEN_HASHES.clear()
    with patch.object(m, "_do_index", side_effect=TimeoutError("slow")):
        m._watch_index_resilient(p, "c", "auto", None)
    calls: list[str] = []
    with patch.object(m, "_do_index",
                      side_effect=lambda s, *a, **k: calls.append(s)):
        m._watch_index_resilient(p, "c", "auto", None)
    assert calls == [p], (
        "same content after a failed dispatch must still be indexed — "
        "the hash guard may only remember SUCCESSFUL dispatches")
