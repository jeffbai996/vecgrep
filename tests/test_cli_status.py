"""Tests for `vecgrep status`.

The command is meant to be the first thing you run when something
looks off — daemon up/down, version, auth state, corpus list with
chunk counts. Tests exercise both the daemon-up and daemon-down paths
since the CLI flips between API and in-process service depending on
which is alive.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from vecgrep import __version__
from vecgrep.cli.main import cli


def _fake_corpus_dict(name: str, docs: int, chunks: int, updated: float = 0.0) -> dict[str, Any]:
    return {
        "name": name,
        "embed_backend": "ollama",
        "embed_model": "nomic-embed-text",
        "dim": 768,
        "chunker": "sentence_window",
        "doc_count": docs,
        "chunk_count": chunks,
        "sources": [],
        "created_at": updated,
        "updated_at": updated,
    }


def test_status_daemon_up_human_output(vg_home: Path) -> None:
    """When the daemon is reachable, status hits /api/corpora and renders
    a human block including 'daemon: up' and the version."""
    runner = CliRunner()
    fake_corpora = [
        _fake_corpus_dict("chats", 3, 8200),
        _fake_corpus_dict("repos", 42, 4787),
    ]
    with (
        patch("vecgrep.cli.main._api_alive", return_value=True),
        patch("vecgrep.cli.main._get", return_value=fake_corpora),
    ):
        result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output
    assert "daemon" in result.output and "up" in result.output
    assert "chats" in result.output
    assert "repos" in result.output
    assert "8200" in result.output
    assert "4787" in result.output


def test_status_daemon_down_falls_back_to_in_process(vg_home: Path, svc, make_doc) -> None:
    """When the daemon is dead, status synthesizes its view from a local
    VecgrepService. No HTTP call should happen."""
    p = make_doc("a.txt", "hello world. small corpus for the status test.")
    svc.index(str(p), "fixture", "sentence_window")

    runner = CliRunner()
    with (
        patch("vecgrep.cli.main._api_alive", return_value=False),
        patch("vecgrep.cli.main._get") as mock_get,
        patch("vecgrep.cli.main.VecgrepService", return_value=svc),
    ):
        result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0, result.output
    mock_get.assert_not_called()
    assert "daemon" in result.output
    assert "down" in result.output
    assert "fixture" in result.output


def test_status_json_output_shape(vg_home: Path) -> None:
    """--json emits a structured object: top-level keys version, daemon,
    auth, home, corpora (list)."""
    runner = CliRunner()
    fake_corpora = [_fake_corpus_dict("solo", 1, 5)]
    with (
        patch("vecgrep.cli.main._api_alive", return_value=True),
        patch("vecgrep.cli.main._get", return_value=fake_corpora),
    ):
        result = runner.invoke(cli, ["status", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["version"] == __version__
    assert payload["daemon"]["alive"] is True
    assert "auth" in payload
    assert "home" in payload
    assert isinstance(payload["corpora"], list)
    assert payload["corpora"][0]["name"] == "solo"
    assert payload["corpora"][0]["chunk_count"] == 5
    assert payload["totals"]["chunks"] == 5
    assert payload["totals"]["docs"] == 1


def test_status_auth_field_reflects_token_setting(vg_home: Path, monkeypatch) -> None:
    """When VECGREP_API_TOKEN is set, status reports auth: enabled."""
    monkeypatch.setenv("VECGREP_API_TOKEN", "deadbeef")
    # Force settings reload so the env var wins.
    from vecgrep.backend import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "_settings", None)

    runner = CliRunner()
    with (
        patch("vecgrep.cli.main._api_alive", return_value=True),
        patch("vecgrep.cli.main._get", return_value=[]),
    ):
        result = runner.invoke(cli, ["status", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["auth"]["enabled"] is True


def test_status_empty_corpora_renders_cleanly(vg_home: Path) -> None:
    """No corpora → still prints the header block and a 'no corpora' line,
    not a crash or empty table."""
    runner = CliRunner()
    with (
        patch("vecgrep.cli.main._api_alive", return_value=True),
        patch("vecgrep.cli.main._get", return_value=[]),
    ):
        result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0, result.output
    assert "no corpora" in result.output.lower()


def test_status_totals_in_human_output(vg_home: Path) -> None:
    """The text table footer should include summed docs and chunks across
    all corpora — that's the whole point of running status."""
    runner = CliRunner()
    fake_corpora = [
        _fake_corpus_dict("a", 10, 100),
        _fake_corpus_dict("b", 20, 250),
    ]
    with (
        patch("vecgrep.cli.main._api_alive", return_value=True),
        patch("vecgrep.cli.main._get", return_value=fake_corpora),
    ):
        result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0, result.output
    # Totals row: 30 docs, 350 chunks.
    assert "30" in result.output
    assert "350" in result.output
