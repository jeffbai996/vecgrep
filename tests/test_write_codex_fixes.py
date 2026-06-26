"""Regression tests for the three Codex review findings on the write tool.

Each test pins a specific vulnerability Codex flagged so it can't silently come
back:

  P1  path traversal — an edit_id with `..`/separators must be rejected before
      it can resolve a target outside the corpus dir.
  P1  frontmatter — a confirmed write-tool doc's status/tier/source_kind must
      land in Document.metadata at index time, or search filters silently fail.
  P2  proposal_id collision — two un-confirmed proposals on the same corpus must
      get distinct ids, or the second overwrites the first in the store and a
      human could confirm id X but write content Y.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from vecgrep.backend.write.proposal import ProposalError, propose


# --- P1: path traversal in edit_id -----------------------------------------

@pytest.mark.parametrize(
    "bad",
    [
        "../outside",
        "../../etc/passwd",
        "notes/../../escape",
        "notes-001/../../x",
        "/abs/path",
        "notes-001\nnotes-002",
        "..",
    ],
)
def test_edit_id_with_traversal_is_rejected(tmp_path, bad):
    with pytest.raises(ProposalError):
        propose("notes", "x", tmp_path, edit_id=bad)


def test_edit_id_valid_doc_id_is_accepted(tmp_path):
    # A well-formed doc id must still work (don't over-reject).
    p = propose("notes", "x", tmp_path, edit_id="notes-007")
    assert p.doc_id == "notes-007"
    # And the resolved target stays inside the corpus dir.
    assert Path(p.target_path).resolve().parent == tmp_path.resolve()


def test_new_entry_target_stays_under_corpus_dir(tmp_path):
    p = propose("notes", "x", tmp_path)
    assert Path(p.target_path).resolve().parent == tmp_path.resolve()


# --- P2: proposal_id uniqueness --------------------------------------------

def test_two_new_proposals_get_distinct_ids(tmp_path):
    # Doc ids are nanosecond timestamps now, so two un-confirmed proposals get
    # DISTINCT doc_ids (no shared target) AND distinct proposal_ids — neither can
    # clobber the other on confirm.
    import re
    a = propose("notes", "first version", tmp_path)
    b = propose("notes", "second version", tmp_path)
    assert a.doc_id != b.doc_id
    assert re.match(r"^notes-\d+$", a.doc_id) and re.match(r"^notes-\d+$", b.doc_id)
    assert a.proposal_id != b.proposal_id


def test_proposal_id_includes_doc_id_for_readability(tmp_path):
    import re
    p = propose("notes", "x", tmp_path)
    assert re.match(r"^prop-notes-\d+-", p.proposal_id)


# --- P1: frontmatter parsed into metadata at index time --------------------

def test_markdown_frontmatter_lifts_into_metadata():
    from vecgrep.backend.ingestion.adapters.markdown import parse_frontmatter

    doc = (
        "---\n"
        "id: notes-001\n"
        "status: active\n"
        "tier: protected\n"
        "source_kind: decision\n"
        "tags: [alpha, beta]\n"
        "---\n\n"
        "the body\n"
    )
    fm = parse_frontmatter(doc)
    assert fm["status"] == "active"
    assert fm["tier"] == "protected"
    assert fm["source_kind"] == "decision"
    assert fm["tags"] == ["alpha", "beta"]


def test_markdown_without_frontmatter_yields_empty():
    from vecgrep.backend.ingestion.adapters.markdown import parse_frontmatter

    assert parse_frontmatter("# just a heading\n\nbody") == {}
    assert parse_frontmatter("--- not really\nfrontmatter") == {}


def test_adapter_metadata_carries_frontmatter(tmp_path):
    from vecgrep.backend.ingestion.adapters.markdown import MarkdownAdapter

    f = tmp_path / "notes-001.md"
    f.write_text(
        "---\nstatus: superseded\ntier: normal\nsource_kind: fact\n---\n\nx\n"
    )
    [doc] = list(MarkdownAdapter().load(str(f)))
    assert doc.metadata["status"] == "superseded"
    assert doc.metadata["source_kind"] == "fact"
    # adapter/path stay authoritative and aren't overridable by frontmatter.
    assert doc.metadata["adapter"] == "markdown"
    assert doc.metadata["path"] == str(f.resolve())


def test_frontmatter_filter_works_end_to_end(svc, tmp_path):
    """The integration Codex's P1 was really about: index a write-tool doc,
    then a meta.tier filter at search time must actually match it (it didn't
    before — the frontmatter never reached metadata)."""
    f = tmp_path / "notes-001.md"
    f.write_text(
        "---\nid: notes-001\nstatus: active\ntier: protected\n"
        "source_kind: decision\n---\n\nThe quick brown fox decision.\n"
    )
    svc.index(str(f), "wcorpus")
    # meta.tier=protected should match; meta.tier=normal should not.
    hit = svc.search("quick brown fox", corpus_name="wcorpus", top_k=5,
                     filters=["meta.tier=protected"])
    assert any("notes-001" in (r.source_id or "") for r in hit)
    miss = svc.search("quick brown fox", corpus_name="wcorpus", top_k=5,
                      filters=["meta.tier=normal"])
    assert not any("notes-001" in (r.source_id or "") for r in miss)
