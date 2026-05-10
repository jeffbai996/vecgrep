"""Tests for `vecgrep search --watch`.

Watch mode re-runs the same query at a fixed interval and prints a
diff of source_ids vs the previous iteration. Useful while indexing
to confirm new chunks are actually showing up in search results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from vecgrep.cli.main import cli


def _hit(sid: str, pct: float = 90.0) -> dict[str, Any]:
    return {
        "similarity_pct": pct,
        "chunk": f"chunk for {sid}",
        "context_before": "",
        "context_after": "",
        "source_id": sid,
        "corpus": "test",
        "metadata": {},
        "matched_by": ["V"],
        "explain": {},
    }


def test_watch_diff_shows_new_and_removed_sources(vg_home: Path) -> None:
    """Two iterations: first returns {a, b}, second returns {b, c}.
    Output should mark 'c' as new and 'a' as gone."""
    runner = CliRunner()
    iterations = [
        {"hits": [_hit("a"), _hit("b")]},
        {"hits": [_hit("b"), _hit("c")]},
    ]
    call_count = {"n": 0}

    def fake_post(path: str, payload: dict) -> Any:
        i = call_count["n"]
        call_count["n"] += 1
        if i >= len(iterations):
            raise KeyboardInterrupt
        return iterations[i]

    with (
        patch("vecgrep.cli.main._api_alive", return_value=True),
        patch("vecgrep.cli.main._post", side_effect=fake_post),
        patch("vecgrep.cli.main.time.sleep", return_value=None),
    ):
        result = runner.invoke(cli, ["search", "x", "--corpus", "test", "--watch", "--interval", "0"])
    assert result.exit_code == 0, result.output
    # 'c' arrived in iteration 2; 'a' disappeared. Diff lines start with '+ ' and '- '.
    iter2 = result.output.split("iteration 2")[-1]
    assert "+" in iter2 and "test" in iter2 and " c" in iter2, iter2
    assert "-" in iter2 and "test" in iter2 and " a" in iter2, iter2


def test_watch_first_iteration_treats_all_as_new(vg_home: Path) -> None:
    """The first pass has no baseline, so every hit is 'new'. The diff
    block should NOT call existing hits 'gone' on iteration 1."""
    runner = CliRunner()
    iterations = [{"hits": [_hit("a"), _hit("b")]}]
    call_count = {"n": 0}

    def fake_post(path: str, payload: dict) -> Any:
        i = call_count["n"]
        call_count["n"] += 1
        if i >= len(iterations):
            raise KeyboardInterrupt
        return iterations[i]

    with (
        patch("vecgrep.cli.main._api_alive", return_value=True),
        patch("vecgrep.cli.main._post", side_effect=fake_post),
        patch("vecgrep.cli.main.time.sleep", return_value=None),
    ):
        result = runner.invoke(cli, ["search", "x", "--corpus", "test", "--watch", "--interval", "0"])
    assert result.exit_code == 0, result.output
    # We can't pin format precisely but iteration 1 should not contain '-' diff lines
    # for any of {a, b}. (Diff lines only appear from iteration 2 onward.)
    iter1_section = result.output.split("iteration 2")[0] if "iteration 2" in result.output else result.output
    assert "- a" not in iter1_section
    assert "- b" not in iter1_section


def test_watch_keyboard_interrupt_exits_cleanly(vg_home: Path) -> None:
    """Ctrl-C during watch should exit 0, not raise. Also ensures we don't
    crash if the very first iteration is interrupted."""
    runner = CliRunner()

    def fake_post(*a, **kw) -> Any:
        raise KeyboardInterrupt

    with (
        patch("vecgrep.cli.main._api_alive", return_value=True),
        patch("vecgrep.cli.main._post", side_effect=fake_post),
        patch("vecgrep.cli.main.time.sleep", return_value=None),
    ):
        result = runner.invoke(cli, ["search", "x", "--corpus", "test", "--watch", "--interval", "0"])
    assert result.exit_code == 0, result.output
