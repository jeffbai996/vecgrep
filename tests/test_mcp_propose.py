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
    import re as _re
    assert r["proposal_id"] and _re.match(r"^notes-\d+$", r["doc_id"])
    assert r["is_edit"] is False
    h = os.environ["VECGREP_HOME"]
    # no doc written
    assert not glob.glob(f"{h}/write/notes/notes-*.md")
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
    f = glob.glob(f"{h}/write/notes/notes-*.md")
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
    assert not glob.glob(f"{h}/write/notes/notes-*.md")
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
    import re as _re
    assert _re.match(r"^prop-notes-\d+-", seen["proposal_id"])
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


# --- propose_delete: the inert, human-confirmed removal path (Jeff 2026-06-25) ---

def _seed_doc(corpus, doc_id, body):
    """Write a doc straight to the corpus write-dir (no embed). Lets the delete
    tests target an existing file without a confirm→index round-trip — which
    would open the embedded Qdrant a second time in-process and trip its
    single-writer lock (a test-harness limitation, not a path users hit)."""
    h = os.environ["VECGREP_HOME"]
    d = os.path.join(h, "write", corpus)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{doc_id}.md")
    with open(p, "w") as f:
        f.write(f"---\nid: {doc_id}\ncorpus: {corpus}\n---\n\n{body}\n")
    return p


def test_propose_delete_removes_nothing_but_persists(home):
    from vecgrep.mcp import server as S
    _seed_doc("notes", "notes-001", "delete me later")
    h = os.environ["VECGREP_HOME"]

    r = json.loads(S._run_propose_delete("notes", "notes-001"))
    assert r["proposal_id"] and r["is_delete"] is True
    assert r["doc_id"] == "notes-001"
    # The proposal is inert: the doc is STILL on disk, only a pending exists.
    assert glob.glob(f"{h}/write/notes/notes-001.md")
    assert glob.glob(f"{h}/write/_pending/*.json")
    # Preview shows what WILL be removed.
    assert "delete me later" in r["preview"] and "DELETE" in r["preview"]


def test_propose_delete_gated_by_allowlist(home):
    from vecgrep.mcp import server as S
    r = json.loads(S._run_propose_delete("squad-shared", "squad-shared-007"))
    assert "error" in r and "not agent-writable" in r["error"]
    assert not glob.glob(f"{os.environ['VECGREP_HOME']}/write/_pending/*.json")


def test_propose_delete_nonexistent_target_rejected(home):
    from vecgrep.mcp import server as S
    r = json.loads(S._run_propose_delete("notes", "notes-999"))
    assert "error" in r and "does not exist" in r["error"]
    assert not glob.glob(f"{os.environ['VECGREP_HOME']}/write/_pending/*.json")


def test_propose_delete_rejects_path_escape(home):
    # delete_id must be a plain prefix-NNN; a traversal id can't escape the corpus.
    from vecgrep.mcp import server as S
    r = json.loads(S._run_propose_delete("notes", "../../etc/passwd"))
    assert "error" in r  # rejected by the id validator before any fs touch


def test_confirm_delete_removes_file_and_pending(home):
    from click.testing import CliRunner
    from vecgrep.cli.main import cli
    from vecgrep.mcp import server as S
    _seed_doc("notes", "notes-001", "doomed entry")
    h = os.environ["VECGREP_HOME"]

    r = json.loads(S._run_propose_delete("notes", "notes-001"))
    pid = r["proposal_id"]
    runner = CliRunner()
    # pending labels it as a delete
    p = runner.invoke(cli, ["pending"])
    assert pid in p.output and "[delete]" in p.output
    # confirm performs the removal (delete_source no-ops cleanly on an un-indexed
    # seed doc — the confirm swallows the de-index warning and removes the file)
    c = runner.invoke(cli, ["confirm", pid])
    assert c.exit_code == 0, c.output
    assert "deleted" in c.output
    # file gone, pending consumed
    assert not glob.glob(f"{h}/write/notes/notes-001.md")
    assert not glob.glob(f"{h}/write/_pending/*.json")
    # confirming again fails (proposal consumed)
    c2 = runner.invoke(cli, ["confirm", pid])
    assert c2.exit_code != 0


def test_discard_cancels_a_delete_proposal(home):
    from click.testing import CliRunner
    from vecgrep.cli.main import cli
    from vecgrep.mcp import server as S
    _seed_doc("notes", "notes-001", "keep me actually")
    r = json.loads(S._run_propose_delete("notes", "notes-001"))
    runner = CliRunner()
    d = runner.invoke(cli, ["discard", r["proposal_id"]])
    assert d.exit_code == 0
    h = os.environ["VECGREP_HOME"]
    # the doc survives — a discarded delete leaves the entry intact
    assert glob.glob(f"{h}/write/notes/notes-001.md")


# --- propose_edit PATCH mode: surgical str-replace (Jeff 2026-07-07) ---
# propose_edit gains an old_str/new_str patch mode: instead of re-sending the
# whole body to fix one line, patch loads the current body, does a strict
# single-match str-replace, and routes the result through the SAME proposal
# path. Uniqueness is a hard error (mirrors the str_replace contract) — 0 or
# >1 matches never silently mis-edit. content and old_str are exclusive modes.

def test_propose_edit_patch_single_match(home):
    from vecgrep.mcp import server as S
    _seed_doc("notes", "notes-001", "the launch is July 25, ship it")
    r = json.loads(S._run_propose(
        "notes", None, edit_id="notes-001",
        old_str="July 25", new_str="July 17"))
    assert "error" not in r, r
    assert r["is_edit"] is True and r["doc_id"] == "notes-001"
    # the pending proposal carries the patched body, only the one token changed
    h = os.environ["VECGREP_HOME"]
    pend = json.loads(open(glob.glob(f"{h}/write/_pending/*.json")[0]).read())
    assert "July 17" in pend["rendered"] and "July 25" not in pend["rendered"]
    assert "the launch is" in pend["rendered"] and "ship it" in pend["rendered"]


def test_propose_edit_patch_old_str_not_found(home):
    from vecgrep.mcp import server as S
    _seed_doc("notes", "notes-001", "nothing to see here")
    r = json.loads(S._run_propose(
        "notes", None, edit_id="notes-001",
        old_str="July 25", new_str="July 17"))
    assert "error" in r and "not found" in r["error"]
    assert not glob.glob(f"{os.environ['VECGREP_HOME']}/write/_pending/*.json")


def test_propose_edit_patch_old_str_not_unique(home):
    from vecgrep.mcp import server as S
    _seed_doc("notes", "notes-001", "date date — pick the right date")
    r = json.loads(S._run_propose(
        "notes", None, edit_id="notes-001",
        old_str="date", new_str="day"))
    assert "error" in r and "not unique" in r["error"] and "3 match" in r["error"]
    assert not glob.glob(f"{os.environ['VECGREP_HOME']}/write/_pending/*.json")


def test_propose_edit_patch_respects_corpus_deny(home):
    # patch mode must hit the SAME default-deny gate as a full edit — no bypass.
    from vecgrep.mcp import server as S
    r = json.loads(S._run_propose(
        "squad-shared", None, edit_id="squad-shared-007",
        old_str="a", new_str="b"))
    assert "error" in r and "not agent-writable" in r["error"]
    assert not glob.glob(f"{os.environ['VECGREP_HOME']}/write/_pending/*.json")


def test_propose_edit_rejects_content_and_old_str_together(home):
    from vecgrep.mcp import server as S
    _seed_doc("notes", "notes-001", "some body")
    r = json.loads(S._run_propose(
        "notes", "a full new body", edit_id="notes-001",
        old_str="some", new_str="any"))
    assert "error" in r  # ambiguous: can't both overwrite AND patch
    assert not glob.glob(f"{os.environ['VECGREP_HOME']}/write/_pending/*.json")


def test_propose_edit_patch_missing_target_doc(home):
    from vecgrep.mcp import server as S  # patch of a non-existent doc → clean error
    r = json.loads(S._run_propose(
        "notes", None, edit_id="notes-404",
        old_str="x", new_str="y"))
    assert "error" in r
    assert not glob.glob(f"{os.environ['VECGREP_HOME']}/write/_pending/*.json")


def test_propose_edit_patch_confirm_round_trip(home):
    from click.testing import CliRunner
    from vecgrep.cli.main import cli
    from vecgrep.mcp import server as S
    _seed_doc("notes", "notes-001", "release date is July 25 for sure")
    r = json.loads(S._run_propose(
        "notes", None, edit_id="notes-001",
        old_str="July 25", new_str="July 17"))
    pid = r["proposal_id"]
    c = CliRunner().invoke(cli, ["confirm", pid])
    assert c.exit_code == 0, c.output
    h = os.environ["VECGREP_HOME"]
    body = open(f"{h}/write/notes/notes-001.md").read()
    assert "July 17" in body and "July 25" not in body
    # frontmatter (the id line) survived the body-only patch
    assert "id: notes-001" in body
