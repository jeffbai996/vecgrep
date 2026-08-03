"""propose_merge: N docs -> 1 canonical-in-place edit + absorb-deletes.

Canonical rule: doc_ids[0] survives (is_edit on it), doc_ids[1:] are absorbed
(deleted at confirm). merged_from provenance lands in the canonical doc's
frontmatter. Nothing is written at propose time.
"""
from pathlib import Path

import pytest

from vecgrep.backend.write import proposal as P


def _seed(corpus_dir: Path, doc_id: str, body: str, tier: str = "normal") -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / f"{doc_id}.md").write_text(
        f"---\nid: {doc_id}\ntier: {tier}\n---\n\n{body}\n")


def test_merge_requires_at_least_two_ids(tmp_path):
    _seed(tmp_path, "note-1", "a")
    with pytest.raises(P.ProposalError):
        P.propose_merge("note", ["note-1"], "merged", tmp_path)


def test_merge_rejects_duplicate_ids(tmp_path):
    _seed(tmp_path, "note-1", "a")
    with pytest.raises(P.ProposalError):
        P.propose_merge("note", ["note-1", "note-1"], "merged", tmp_path)


def test_merge_rejects_missing_source(tmp_path):
    _seed(tmp_path, "note-1", "a")
    with pytest.raises(P.ProposalError):
        P.propose_merge("note", ["note-1", "note-404"], "merged", tmp_path)


def test_merge_builds_canonical_in_place_proposal(tmp_path):
    _seed(tmp_path, "note-1", "a")
    _seed(tmp_path, "note-2", "b")
    _seed(tmp_path, "note-3", "c")
    pr = P.propose_merge("note", ["note-1", "note-2", "note-3"],
                         "a+b+c merged", tmp_path)
    assert pr.doc_id == "note-1"
    assert pr.is_edit is True
    assert pr.is_delete is False
    assert pr.merge_absorbs == ["note-2", "note-3"]
    assert pr.meta["merged_from"] == ["note-2", "note-3"]
    assert "merged_from: [note-2, note-3]" in pr.rendered
    assert "a+b+c merged" in pr.rendered
    # Nothing written/removed at propose time.
    assert (tmp_path / "note-2.md").exists()


def test_merge_writes_nothing(tmp_path):
    _seed(tmp_path, "note-1", "a")
    _seed(tmp_path, "note-2", "b")
    before = sorted(p.name for p in tmp_path.iterdir())
    P.propose_merge("note", ["note-1", "note-2"], "merged", tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_proposal_backcompat_without_merge_absorbs():
    # On-disk pending proposals predating the field must still load.
    d = dict(proposal_id="p1", doc_id="note-1", corpus="note",
             target_path="/tmp/x.md", rendered="x", is_edit=False, meta={})
    pr = P.Proposal(**d)
    assert pr.merge_absorbs == []
