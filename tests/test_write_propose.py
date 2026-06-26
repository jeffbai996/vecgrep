"""Write-tool phase 2 — propose + preview. The load-bearing invariant: it
WRITES NOTHING. It assigns the next id, renders the file, and (for updates)
diffs against the existing version, but never touches disk or the index.
"""
from __future__ import annotations

import re

import pytest

from vecgrep.backend.write import proposal as P

_ID = re.compile(r"^note-\d+$")  # <prefix>-<epoch> (timestamp, not a serial)


def test_propose_assigns_timestamp_id(tmp_path):
    pr = P.propose("note", "first note", tmp_path)
    assert _ID.match(pr.doc_id), f"expected note-<epoch>, got {pr.doc_id!r}"


def test_propose_ids_are_unique(tmp_path):
    # Ids are timestamp-based, not a running serial — back-to-back proposes (even
    # within the same second) must still get DISTINCT ids so neither overwrites
    # the other on confirm.
    ids = {P.propose("note", f"entry {i}", tmp_path).doc_id for i in range(5)}
    assert len(ids) == 5, f"collision: {ids}"
    assert all(_ID.match(i) for i in ids)


def test_propose_writes_nothing(tmp_path):
    before = set(tmp_path.iterdir())
    P.propose("note", "should not persist", tmp_path)
    after = set(tmp_path.iterdir())
    assert before == after  # no file created


def test_propose_returns_target_path(tmp_path):
    pr = P.propose("note", "x", tmp_path)
    assert pr.target_path == str(tmp_path / f"{pr.doc_id}.md")


def test_rendered_doc_carries_schema_frontmatter(tmp_path):
    pr = P.propose("note", "body text here", tmp_path,
                   meta={"origin": "human", "source_kind": "decision", "tags": ["a", "b"]})
    r = pr.rendered
    assert re.search(r"id: note-\d+", r)
    assert "status: active" in r
    assert "origin: human" in r
    assert "source_kind: decision" in r
    assert "tags: [a, b]" in r
    assert "body text here" in r


def test_title_auto_derived_from_first_line(tmp_path):
    # No explicit title → derived from the first content line (markdown '#'
    # stripped) and persisted, a scannable handle vs the opaque timestamp id.
    pr = P.propose("note", "# Victor grad-date correction\nbody…", tmp_path,
                   meta={"origin": "human"})
    assert "title: Victor grad-date correction" in pr.rendered


def test_explicit_title_wins(tmp_path):
    pr = P.propose("note", "ignored first line", tmp_path,
                   meta={"origin": "human", "title": "My Chosen Title"})
    assert "title: My Chosen Title" in pr.rendered


def test_origin_defaults_to_bot_suggested(tmp_path):
    pr = P.propose("note", "x", tmp_path)
    assert pr.meta["origin"] == "bot-suggested"
    assert "origin: bot-suggested" in pr.rendered


def test_proposal_has_stable_id_for_confirm(tmp_path):
    pr = P.propose("note", "x", tmp_path)
    assert pr.proposal_id  # a confirm step must be able to cite this
    assert pr.is_edit is False


# --- edits: target an existing id, overwrite on confirm, still no write here ---

def test_edit_targets_existing_id_writes_nothing(tmp_path):
    (tmp_path / "note-007.md").write_text(
        "---\nid: note-007\nstatus: active\n---\nold body\n")
    pr = P.propose("note", "new body", tmp_path, edit_id="note-007")
    assert pr.is_edit is True
    assert pr.doc_id == "note-007"
    assert pr.target_path == str(tmp_path / "note-007.md")
    assert "new body" in pr.rendered
    # propose writes nothing — the file is untouched until confirm.
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


def test_memory_and_todo_source_kinds_accepted(tmp_path):
    # memory + todo were added to SOURCE_KINDS so an agent can propose those
    # record types (routed to the real store by an operator write-through).
    for kind in ("memory", "todo", "journal"):
        pr = P.propose("note", f"a {kind} entry", tmp_path,
                       meta={"origin": "human", "source_kind": kind})
        assert f"source_kind: {kind}" in pr.rendered
