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
    # The propose path is default-deny on corpus now; these mechanics tests use
    # the "notes" corpus, so allow it here as if an operator had widened the set.
    monkeypatch.setenv("VECGREP_PROPOSE_ALLOWED_CORPORA", "notes")
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


def test_default_propose_corpus_is_claude_ai(home):
    """An agent that doesn't name a corpus proposes into 'claude-ai' — a
    dedicated corpus for agent contributions, kept apart from human/ingested
    corpora."""
    from vecgrep.mcp import server as S
    assert S.DEFAULT_PROPOSE_CORPUS == "claude-ai"
    r = json.loads(S._run_propose(S.DEFAULT_PROPOSE_CORPUS, "an agent fact"))
    assert r["corpus"] == "claude-ai"
    assert r["doc_id"].startswith("claudeai-")  # slug prefix, sanitized


def test_discard_removes_proposal_without_writing(home):
    from click.testing import CliRunner
    from vecgrep.cli.main import cli
    from vecgrep.mcp import server as S

    r = json.loads(S._run_propose("notes", "discard me"))
    pid = r["proposal_id"]
    runner = CliRunner()
    d = runner.invoke(cli, ["discard", pid])
    assert d.exit_code == 0, d.output
    assert "discarded" in d.output
    h = os.environ["VECGREP_HOME"]
    # nothing written, pending cleared
    assert not glob.glob(f"{h}/write/notes/notes-001.md")
    assert not glob.glob(f"{h}/write/_pending/*.json")
    # discarding again fails (already gone)
    d2 = runner.invoke(cli, ["discard", pid])
    assert d2.exit_code != 0


def test_propose_hook_fires_with_payload(home, tmp_path, monkeypatch):
    """VECGREP_PROPOSE_HOOK runs with the proposal JSON on stdin so a deployment
    can post a Discord card / notification out-of-band."""
    from vecgrep.mcp import server as S
    out = tmp_path / "hook_saw.json"
    # A tiny shell hook that copies stdin to a file we can inspect.
    monkeypatch.setenv("VECGREP_PROPOSE_HOOK", f"cat > {out}")
    S._run_propose("notes", "card me", source_kind="fact")
    assert out.exists(), "hook should have run"
    seen = json.loads(out.read_text())
    assert seen["proposal_id"].startswith("prop-notes-001-")
    assert seen["corpus"] == "notes"
    assert "card me" in seen["preview"]


def test_propose_hook_failure_does_not_break_propose(home, monkeypatch):
    """A missing/failing hook must never break the propose — the proposal is
    already stored; notification is best-effort."""
    from vecgrep.mcp import server as S
    monkeypatch.setenv("VECGREP_PROPOSE_HOOK", "this-command-does-not-exist-xyz")
    r = json.loads(S._run_propose("notes", "still works"))
    assert r["proposal_id"]  # propose succeeded despite the broken hook
    assert glob.glob(f"{os.environ['VECGREP_HOME']}/write/_pending/*.json")


# --- WS3: corpus allowlist (default-deny) + size cap at the tool boundary ---

def test_propose_to_disallowed_corpus_is_rejected(home):
    """The wall: an agent cannot propose into a corpus outside the allowlist, so
    a confirm can never publish agent content into a shared/human corpus. The
    `home` fixture allows only 'notes' (+ the always-allowed default)."""
    from vecgrep.mcp import server as S
    r = json.loads(S._run_propose("squad-shared", "leak attempt"))
    assert "error" in r and "not agent-writable" in r["error"]
    # nothing landed on disk -- no pending proposal, no corpus dir
    h = os.environ["VECGREP_HOME"]
    assert not glob.glob(f"{h}/write/_pending/*.json")
    assert not glob.glob(f"{h}/write/squad-shared")


def test_default_propose_corpus_is_always_allowed(home, monkeypatch):
    # Even with an unrelated allowlist, the dedicated default corpus stays open.
    monkeypatch.setenv("VECGREP_PROPOSE_ALLOWED_CORPORA", "something-else")
    from vecgrep.mcp import server as S
    r = json.loads(S._run_propose(S.DEFAULT_PROPOSE_CORPUS, "an agent fact"))
    assert r["proposal_id"] and "error" not in r


def test_allowlist_widening_via_env(home, monkeypatch):
    monkeypatch.setenv("VECGREP_PROPOSE_ALLOWED_CORPORA", "team-notes,research")
    from vecgrep.mcp import server as S
    assert json.loads(S._run_propose("team-notes", "x"))["proposal_id"]
    assert json.loads(S._run_propose("research", "y"))["proposal_id"]
    assert "error" in json.loads(S._run_propose("notes", "z"))  # not widened here


def test_propose_edit_into_disallowed_corpus_is_rejected(home):
    # propose_edit infers the corpus from the doc-id prefix; that inferred corpus
    # must also clear the allowlist.
    from vecgrep.mcp import server as S
    r = json.loads(S._run_propose("squad-shared", "edit", edit_id="squad-shared-007"))
    assert "error" in r and "not agent-writable" in r["error"]


def test_oversized_content_is_rejected(home, monkeypatch):
    from vecgrep.mcp import server as S
    # patch the resolved cap on the module (auto-restored at test end) rather
    # than reloading the module.
    monkeypatch.setattr(S, "MAX_PROPOSAL_CONTENT_BYTES", 100)
    r = json.loads(S._run_propose("notes", "x" * 500))
    assert "error" in r and "proposal cap" in r["error"]
    assert not glob.glob(f"{os.environ['VECGREP_HOME']}/write/_pending/*.json")


def test_full_preview_not_truncated_at_500(home):
    # The old 500-char preview hid the body of any real entry. A ~1.5k entry must
    # come back in full so a human can actually review what they'd confirm.
    from vecgrep.mcp import server as S
    body = "para. " * 300  # ~1800 chars, well over the old 500 cap
    r = json.loads(S._run_propose("notes", body))
    assert "(truncated)" not in r["preview"]
    assert r["preview"].count("para.") > 250
