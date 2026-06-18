"""Write-tool phase 2 — propose + preview. The load-bearing invariant: it
WRITES NOTHING. It assigns the next id, renders the file, and (for updates)
diffs against the existing version, but never touches disk or the index.
"""
from __future__ import annotations

import pytest

from vecgrep.backend.write import proposal as P


def test_propose_assigns_first_id(tmp_path):
    pr = P.propose("note", "first note", tmp_path)
    assert pr.doc_id == "note-001"


def test_propose_assigns_next_sequential_id(tmp_path):
    # Existing files in the corpus dir → next id continues the sequence.
    (tmp_path / "note-001.md").write_text("---\nid: note-001\n---\nx")
    (tmp_path / "note-003.md").write_text("---\nid: note-003\n---\ny")
    pr = P.propose("note", "another", tmp_path)
    assert pr.doc_id == "note-004"  # max(1,3)+1


def test_propose_writes_nothing(tmp_path):
    before = set(tmp_path.iterdir())
    P.propose("note", "should not persist", tmp_path)
    after = set(tmp_path.iterdir())
    assert before == after  # no file created


def test_propose_returns_target_path(tmp_path):
    pr = P.propose("note", "x", tmp_path)
    assert pr.target_path == str(tmp_path / "note-001.md")


def test_rendered_doc_carries_schema_frontmatter(tmp_path):
    pr = P.propose("note", "body text here", tmp_path,
                   meta={"origin": "human", "source_kind": "decision", "tags": ["a", "b"]})
    r = pr.rendered
    assert "id: note-001" in r
    assert "status: active" in r
    assert "origin: human" in r
    assert "source_kind: decision" in r
    assert "version: 1" in r
    assert "tags: [a, b]" in r
    assert "body text here" in r


def test_origin_defaults_to_bot_suggested(tmp_path):
    pr = P.propose("note", "x", tmp_path)
    assert pr.meta["origin"] == "bot-suggested"
    assert "origin: bot-suggested" in pr.rendered


def test_proposal_has_stable_id_for_confirm(tmp_path):
    pr = P.propose("note", "x", tmp_path)
    assert pr.proposal_id  # a confirm step must be able to cite this
    assert pr.is_update is False
    assert pr.diff == ""


# --- updates: new version + diff, still no write ---

def test_update_bumps_version_and_diffs(tmp_path):
    (tmp_path / "note-007.md").write_text(
        "---\nid: note-007\nversion: 1\nstatus: active\n---\nold body\n")
    pr = P.propose("note", "new body", tmp_path, update_id="note-007")
    assert pr.is_update is True
    assert pr.doc_id == "note-007"
    assert "version: 2" in pr.rendered
    assert pr.diff  # non-empty unified diff
    assert "old body" in pr.diff and "new body" in pr.diff
    # still wrote nothing
    assert (tmp_path / "note-007.md").read_text().splitlines()[-1] == "old body"


# --- validation: bad metadata is caught at propose, before any write path ---

def test_bad_origin_rejected(tmp_path):
    with pytest.raises(P.ProposalError):
        P.propose("note", "x", tmp_path, meta={"origin": "the-internet"})


def test_bad_source_kind_rejected(tmp_path):
    with pytest.raises(P.ProposalError):
        P.propose("note", "x", tmp_path, meta={"source_kind": "vibes"})


def test_empty_content_rejected(tmp_path):
    with pytest.raises(P.ProposalError):
        P.propose("note", "   ", tmp_path)
