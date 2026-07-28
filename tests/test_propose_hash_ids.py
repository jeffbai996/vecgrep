"""Edit/delete proposals must accept hash-suffixed doc ids (task-a1b2c), not
just numeric ones — mirror corpora of hash-id upstream stores use them."""
from __future__ import annotations

from vecgrep.backend.write import proposal as P


def _seed(corpus_dir, doc_id: str) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / f"{doc_id}.md").write_text(f"---\nid: {doc_id}\n---\n\nbody\n")


def test_edit_accepts_hash_id(tmp_path):
    _seed(tmp_path, "task-a1b2c")
    pr = P.propose("task", "new body", tmp_path, edit_id="task-a1b2c")
    assert pr.doc_id == "task-a1b2c" and pr.is_edit


def test_delete_accepts_hash_id(tmp_path):
    _seed(tmp_path, "task-a1b2c")
    pr = P.propose_delete("task", "task-a1b2c", tmp_path)
    assert pr.doc_id == "task-a1b2c" and pr.is_delete


def test_traversal_still_rejected(tmp_path):
    import pytest
    _seed(tmp_path, "task-a1b2c")
    for bad in ("../evil", "task-a.b", "task-a/b", "task-A1"):
        with pytest.raises(P.ProposalError):
            P.propose("task", "x", tmp_path, edit_id=bad)
