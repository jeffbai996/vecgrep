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
    """Stand-in for VecgrepService: records index() calls, fakes search verify."""
    def __init__(self, found=True):
        self.indexed = []
        self._found = found

    def index(self, source, corpus_name, **kw):
        self.indexed.append((source, corpus_name))
        return (1, 1, 0)

    def search(self, query, corpus_name=None, top_k=5, **kw):
        if not self._found:
            return []
        r = type("R", (), {})()
        r.source_id = self._last_written  # set by the test via attribute
        return [r]


def test_confirm_requires_known_proposal(corpus_dir, store):
    svc = _FakeService()
    with pytest.raises(C.ConfirmError):
        C.confirm("prop-does-not-exist", store, svc, "notes", corpus_dir)


def test_confirm_writes_the_file(corpus_dir, store):
    pr = P.propose("notes", "a real note", corpus_dir, meta={"origin": "human"})
    store.put(pr)
    svc = _FakeService()
    svc._last_written = pr.target_path
    res = C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir)
    written = Path(pr.target_path)
    assert written.exists()
    assert "a real note" in written.read_text()
    assert res.ok is True


def test_confirm_is_append_only_never_overwrites(corpus_dir, store):
    # A file already at the target path must NOT be clobbered — confirm refuses
    # to overwrite (append-only is structural; updates go through versioning).
    pr = P.propose("notes", "first", corpus_dir, meta={"origin": "human"})
    store.put(pr)
    Path(pr.target_path).write_text("PRE-EXISTING — must not be lost")
    svc = _FakeService()
    with pytest.raises(C.ConfirmError):
        C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir)
    assert "PRE-EXISTING" in Path(pr.target_path).read_text()


def test_confirm_reembeds_the_written_file(corpus_dir, store):
    pr = P.propose("notes", "embed me", corpus_dir, meta={"origin": "human"})
    store.put(pr)
    svc = _FakeService()
    svc._last_written = pr.target_path
    C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir)
    assert svc.indexed == [(pr.target_path, "notes")]  # re-embedded exactly the new file


def test_confirm_flags_when_verify_fails(corpus_dir, store):
    # Written but not retrievable → result flags it, does not silently succeed.
    pr = P.propose("notes", "ghost note", corpus_dir, meta={"origin": "human"})
    store.put(pr)
    svc = _FakeService(found=False)
    res = C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir)
    assert res.ok is False
    assert "verif" in res.message.lower()
    assert Path(pr.target_path).exists()  # the file was still written


def test_confirm_consumes_the_proposal(corpus_dir, store):
    # After a successful confirm the proposal is gone — a replayed confirm fails.
    pr = P.propose("notes", "once", corpus_dir, meta={"origin": "human"})
    store.put(pr)
    svc = _FakeService()
    svc._last_written = pr.target_path
    C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir)
    with pytest.raises(C.ConfirmError):
        C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir)


def test_proposal_store_roundtrip(store, corpus_dir):
    pr = P.propose("notes", "persist me", corpus_dir)
    store.put(pr)
    got = store.get(pr.proposal_id)
    assert got is not None
    assert got.doc_id == pr.doc_id
    assert got.rendered == pr.rendered
