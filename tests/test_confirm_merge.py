"""confirm() on a merge: canonical written+embedded BEFORE absorbed deletes;
a mid-way failure leaves absorbed docs intact; protected sources escalate."""
from pathlib import Path

import pytest

from vecgrep.backend.write import confirm as C
from vecgrep.backend.write import proposal as P


class FakeSvc:
    def __init__(self, fail_index: bool = False):
        self.calls = []
        self.fail_index = fail_index

    def index(self, path, corpus):
        self.calls.append(("index", path))
        if self.fail_index:
            raise RuntimeError("embed down")

    def delete_source(self, corpus, path):
        self.calls.append(("delete_source", path))

    def search(self, probe, corpus_name=None, top_k=10):
        class R:  # the canonical id is always "found"
            source_id = "note-1"
        return [R()]


def _seed(d: Path, doc_id: str, body: str, tier: str = "normal") -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{doc_id}.md").write_text(
        f"---\nid: {doc_id}\ntier: {tier}\n---\n\n{body}\n")


def _merge_setup(tmp_path, tier2="normal"):
    corpus_dir = tmp_path / "c"
    _seed(corpus_dir, "note-1", "a")
    _seed(corpus_dir, "note-2", "b", tier=tier2)
    store = C.ProposalStore(tmp_path / "pending")
    pr = P.propose_merge("note", ["note-1", "note-2"], "merged body", corpus_dir)
    store.put(pr)
    return corpus_dir, store, pr


def test_merge_writes_canonical_then_deletes_absorbed(tmp_path):
    corpus_dir, store, pr = _merge_setup(tmp_path)
    svc = FakeSvc()
    res = C.confirm(pr.proposal_id, store, svc, "note", corpus_dir,
                    confirmed_by="jeff")
    assert res.ok
    assert "merged body" in (corpus_dir / "note-1.md").read_text()
    assert "merged_from: [note-2]" in (corpus_dir / "note-1.md").read_text()
    assert not (corpus_dir / "note-2.md").exists()
    assert "note-2" in res.message
    # Ordering: the canonical index() precedes the absorbed delete_source().
    ops = [op for op, _ in svc.calls]
    assert ops.index("index") < ops.index("delete_source")
    assert store.get(pr.proposal_id) is None  # consumed


def test_merge_embed_failure_leaves_absorbed_intact(tmp_path):
    corpus_dir, store, pr = _merge_setup(tmp_path)
    res = C.confirm(pr.proposal_id, store, FakeSvc(fail_index=True), "note",
                    corpus_dir, confirmed_by="jeff")
    assert not res.ok
    assert (corpus_dir / "note-2.md").exists()  # nothing absorbed was deleted


def test_merge_protected_absorbed_requires_ack(tmp_path):
    corpus_dir, store, pr = _merge_setup(tmp_path, tier2="protected")
    with pytest.raises(C.ConfirmError):
        C.confirm(pr.proposal_id, store, FakeSvc(), "note", corpus_dir,
                  confirmed_by="jeff")
    # Ack = canonical id unlocks it.
    res = C.confirm(pr.proposal_id, store, FakeSvc(), "note", corpus_dir,
                    confirmed_by="jeff", protected_ack="note-1")
    assert res.ok
