"""MCP propose path — the injection-safe write surface.

claude.ai (an agent that ingests untrusted content) can PROPOSE but never write
directly: propose_write/propose_edit create an inert pending proposal; only a
human `vecgrep confirm` turns it into a write. These tests pin: propose writes
nothing, the proposal persists, and a subsequent CLI confirm performs the write.
"""
from __future__ import annotations

import glob
import json
import os

import pytest

pytest.importorskip("mcp")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("VECGREP_HOME", str(tmp_path / "vg"))
    import vecgrep.backend.config as cfg
    monkeypatch.setattr(cfg, "_settings", None)
    return tmp_path


def test_propose_writes_nothing_but_persists(home):
    from vecgrep.mcp import server as S
    r = json.loads(S._run_propose("notes", "a proposed fact", source_kind="fact"))
    assert r["proposal_id"] and r["doc_id"] == "notes-001"
    assert r["is_edit"] is False
    h = os.environ["VECGREP_HOME"]
    # no doc written
    assert not glob.glob(f"{h}/write/notes/notes-001.md")
    # but a pending proposal exists
    assert glob.glob(f"{h}/write/_pending/*.json")


def test_proposed_entry_origin_is_bot_suggested(home):
    from vecgrep.mcp import server as S
    S._run_propose("notes", "x")
    h = os.environ["VECGREP_HOME"]
    pend = json.loads(open(glob.glob(f"{h}/write/_pending/*.json")[0]).read())
    assert pend["meta"]["origin"] == "bot-suggested"  # the wall: bots propose


def test_cli_confirm_turns_proposal_into_write(home):
    from click.testing import CliRunner
    from vecgrep.cli.main import cli
    from vecgrep.mcp import server as S

    r = json.loads(S._run_propose("notes", "confirm me into existence"))
    pid = r["proposal_id"]
    runner = CliRunner()
    # pending lists it
    p = runner.invoke(cli, ["pending"])
    assert pid in p.output
    # confirm writes it
    c = runner.invoke(cli, ["confirm", pid])
    assert c.exit_code == 0, c.output
    h = os.environ["VECGREP_HOME"]
    f = glob.glob(f"{h}/write/notes/notes-001.md")
    assert f and "confirm me into existence" in open(f[0]).read()
    # proposal consumed — confirming again fails
    c2 = runner.invoke(cli, ["confirm", pid])
    assert c2.exit_code != 0


def test_pending_empty_when_none(home):
    from click.testing import CliRunner
    from vecgrep.cli.main import cli
    r = CliRunner().invoke(cli, ["pending"])
    assert r.exit_code == 0
    assert "No pending" in r.output
