"""Write-tool phase 3 — confirm gate + append-only write + re-embed + verify.

The core, now actually writing. A proposal is persisted by propose(); confirm()
looks it up by id (a stale/unknown id is rejected — the gate), writes the file
APPEND-ONLY (never overwrites an existing path), re-embeds just that file, and
verifies the chunk is retrievable (flags if not — no silent write-but-
unsearchable). Identity-of-confirmer enforcement (the human-only wall) is a
later phase; this phase proves the pipeline + the no-overwrite + verify guards.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from vecgrep.backend.write import proposal as P
# Import the submodule explicitly — the package __init__ re-exports a `confirm`
# FUNCTION, which would shadow the module name under `from ... import confirm`.
import vecgrep.backend.write.confirm as C


@pytest.fixture
def corpus_dir(tmp_path):
    d = tmp_path / "notes"
    d.mkdir()
    return d


@pytest.fixture
def store(tmp_path):
    return C.ProposalStore(tmp_path / "pending")


class _FakeService:
    """Stand-in for VecgrepService: records index()/delete_source() calls, fakes
    search verify."""
    def __init__(self, found=True):
        self.indexed = []
        self.deleted = []
        self._found = found

    def index(self, source, corpus_name, **kw):
        self.indexed.append((source, corpus_name))
        return (1, 1, 0)

    def delete_source(self, corpus_name, source_id):
        self.deleted.append((corpus_name, source_id))

    def search(self, query, corpus_name=None, top_k=5, **kw):
        if not self._found:
            return []
        r = type("R", (), {})()
        r.source_id = self._last_written  # set by the test via attribute
        return [r]


def test_confirm_requires_known_proposal(corpus_dir, store):
    svc = _FakeService()
    with pytest.raises(C.ConfirmError):
        C.confirm("prop-does-not-exist", store, svc, "notes", corpus_dir, confirmed_by="owner")


def test_confirm_writes_the_file(corpus_dir, store):
    pr = P.propose("notes", "a real note", corpus_dir, meta={"origin": "human"})
    store.put(pr)
    svc = _FakeService()
    svc._last_written = pr.target_path
    res = C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir, confirmed_by="owner")
    written = Path(pr.target_path)
    assert written.exists()
    assert "a real note" in written.read_text()
    assert res.ok is True


def test_new_write_refuses_to_clobber_existing_id(corpus_dir, store):
    # A NEW entry (not an edit) must never silently overwrite an existing doc id.
    pr = P.propose("notes", "first", corpus_dir, meta={"origin": "human"})
    store.put(pr)
    Path(pr.target_path).write_text("PRE-EXISTING — must not be lost")
    svc = _FakeService()
    with pytest.raises(C.ConfirmError):
        C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir, confirmed_by="owner")
    assert "PRE-EXISTING" in Path(pr.target_path).read_text()


def test_edit_overwrites_in_place(corpus_dir, store):
    # An EDIT targets an existing id on purpose and overwrites it (simple, no
    # versioning). Seed a doc, then edit it.
    (corpus_dir / "notes-001.md").write_text("---\nid: notes-001\nstatus: active\n---\nold\n")
    pr = P.propose("notes", "new content", corpus_dir, edit_id="notes-001",
                   meta={"origin": "human"})
    store.put(pr)
    svc = _FakeService(); svc._last_written = pr.target_path
    res = C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir, confirmed_by="owner")
    assert res.ok is True
    body = (corpus_dir / "notes-001.md").read_text()
    assert "new content" in body and "old" not in body


def test_edit_of_missing_target_rejected(corpus_dir, store):
    pr = P.propose("notes", "x", corpus_dir, edit_id="notes-999", meta={"origin": "human"})
    store.put(pr)
    svc = _FakeService()
    with pytest.raises(C.ConfirmError):
        C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir, confirmed_by="owner")


def test_confirm_reembeds_the_written_file(corpus_dir, store):
    pr = P.propose("notes", "embed me", corpus_dir, meta={"origin": "human"})
    store.put(pr)
    svc = _FakeService()
    svc._last_written = pr.target_path
    C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir, confirmed_by="owner")
    assert svc.indexed == [(pr.target_path, "notes")]  # re-embedded exactly the new file


def test_confirm_flags_when_verify_fails(corpus_dir, store):
    # Written but not retrievable → result flags it, does not silently succeed.
    pr = P.propose("notes", "ghost note", corpus_dir, meta={"origin": "human"})
    store.put(pr)
    svc = _FakeService(found=False)
    res = C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir, confirmed_by="owner")
    assert res.ok is False
    assert "verif" in res.message.lower()
    assert Path(pr.target_path).exists()  # the file was still written


def test_confirm_consumes_the_proposal(corpus_dir, store):
    # After a successful confirm the proposal is gone — a replayed confirm fails.
    pr = P.propose("notes", "once", corpus_dir, meta={"origin": "human"})
    store.put(pr)
    svc = _FakeService()
    svc._last_written = pr.target_path
    C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir, confirmed_by="owner")
    with pytest.raises(C.ConfirmError):
        C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir, confirmed_by="owner")


def test_proposal_store_roundtrip(store, corpus_dir):
    pr = P.propose("notes", "persist me", corpus_dir)
    store.put(pr)
    got = store.get(pr.proposal_id)
    assert got is not None
    assert got.doc_id == pr.doc_id
    assert got.rendered == pr.rendered


# --- delete proposals: confirm removes the file + de-indexes (Jeff 2026-06-25) ---

def test_confirm_delete_removes_file_and_deindexes(corpus_dir, store):
    # Seed a doc, propose its delete, confirm → file gone + delete_source called.
    pr_w = P.propose("notes", "to be deleted", corpus_dir, meta={"origin": "human"})
    store.put(pr_w)
    svc = _FakeService()
    svc._last_written = pr_w.target_path
    C.confirm(pr_w.proposal_id, store, svc, "notes", corpus_dir, confirmed_by="owner")
    assert Path(pr_w.target_path).exists()

    pr_d = P.propose_delete("notes", pr_w.doc_id, corpus_dir, meta={"origin": "human"})
    store.put(pr_d)
    res = C.confirm(pr_d.proposal_id, store, svc, "notes", corpus_dir,
                    confirmed_by="owner")
    assert res.ok and "deleted" in res.message
    assert not Path(pr_w.target_path).exists()          # file removed
    assert (("notes", pr_w.target_path) in svc.deleted)  # embeddings removed
    assert store.get(pr_d.proposal_id) is None           # proposal consumed


def test_confirm_delete_requires_human_confirmer(corpus_dir, store):
    # The wall holds for deletes too: no confirmer identity → no delete.
    (corpus_dir / "notes-001.md").write_text("---\nid: notes-001\n---\n\nx")
    pr = P.propose_delete("notes", "notes-001", corpus_dir, meta={"origin": "human"})
    store.put(pr)
    with pytest.raises(C.ConfirmError):
        C.confirm(pr.proposal_id, store, _FakeService(), "notes", corpus_dir,
                  confirmed_by="")  # no human → rejected, file untouched
    assert (corpus_dir / "notes-001.md").exists()


def test_confirm_delete_protected_needs_ack(corpus_dir, store):
    # A protected doc can't be deleted without re-stating its exact id.
    (corpus_dir / "notes-001.md").write_text(
        "---\nid: notes-001\ntier: protected\n---\n\nimportant")
    pr = P.propose_delete("notes", "notes-001", corpus_dir, meta={"origin": "human"})
    store.put(pr)
    svc = _FakeService()
    with pytest.raises(C.ConfirmError):
        C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir,
                  confirmed_by="owner", protected_ack="wrong")
    assert (corpus_dir / "notes-001.md").exists()  # still there
    # correct ack → deletes
    res = C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir,
                    confirmed_by="owner", protected_ack="notes-001")
    assert res.ok and not (corpus_dir / "notes-001.md").exists()


def test_confirm_delete_deindex_failure_still_removes_file(corpus_dir, store):
    # If de-indexing raises, the file is STILL removed and the warning surfaced
    # (orphaned vectors are recoverable by a reindex; an undeleted file isn't the
    # user's intent).
    (corpus_dir / "notes-001.md").write_text("---\nid: notes-001\n---\n\nx")
    pr = P.propose_delete("notes", "notes-001", corpus_dir, meta={"origin": "human"})
    store.put(pr)

    class _Boom(_FakeService):
        def delete_source(self, corpus_name, source_id):
            raise RuntimeError("qdrant down")

    res = C.confirm(pr.proposal_id, store, _Boom(), "notes", corpus_dir,
                    confirmed_by="owner")
    assert res.ok and "de-index warning" in res.message
    assert not (corpus_dir / "notes-001.md").exists()


def test_proposal_store_roundtrips_is_delete(store, corpus_dir):
    # The new is_delete flag must survive the JSON round-trip in the store.
    (corpus_dir / "notes-001.md").write_text("---\nid: notes-001\n---\n\nx")
    pr = P.propose_delete("notes", "notes-001", corpus_dir)
    store.put(pr)
    got = store.get(pr.proposal_id)
    assert got is not None and got.is_delete is True
