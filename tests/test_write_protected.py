"""Write-tool phase 6 — protected tier.

Safety-critical docs (identity, IBKR facts, medical, the "staying in Canada"
decision) are tier: protected. Writing or EDITING one needs stronger
confirmation than a normal write: on top of the human confirm (the wall), the
confirmer must re-state the exact doc id (proving deliberate intent, not a
fat-finger or a slipped-through bot suggestion). A protected doc can't be
silently changed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from vecgrep.backend.write import proposal as P
import vecgrep.backend.write.confirm as C


@pytest.fixture
def corpus_dir(tmp_path):
    d = tmp_path / "notes"; d.mkdir(); return d


@pytest.fixture
def store(tmp_path):
    return C.ProposalStore(tmp_path / "pending")


class _FakeService:
    def __init__(self): self.indexed = []
    def index(self, s, c, **k): self.indexed.append(s); return (1, 1, 0)
    def search(self, q, corpus_name=None, top_k=5, **k):
        r = type("R", (), {})(); r.source_id = self._live; return [r]


OWNER = "owner-123"


def _put_protected_proposal(corpus_dir, store):
    pr = P.propose("notes", "safety-critical fact", corpus_dir,
                   meta={"origin": "human", "tier": "protected"})
    store.put(pr)
    return pr


def test_protected_write_rejected_without_id_ack(corpus_dir, store):
    pr = _put_protected_proposal(corpus_dir, store)
    svc = _FakeService(); svc._live = pr.target_path
    with pytest.raises(C.ConfirmError):
        C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir,
                  confirmed_by=OWNER)  # no protected_ack


def test_protected_write_rejected_with_wrong_id_ack(corpus_dir, store):
    pr = _put_protected_proposal(corpus_dir, store)
    svc = _FakeService(); svc._live = pr.target_path
    with pytest.raises(C.ConfirmError):
        C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir,
                  confirmed_by=OWNER, protected_ack="wrong-id")


def test_protected_write_succeeds_with_correct_id_ack(corpus_dir, store):
    pr = _put_protected_proposal(corpus_dir, store)
    svc = _FakeService(); svc._live = pr.target_path
    res = C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir,
                    confirmed_by=OWNER, protected_ack=pr.doc_id)
    assert res.ok is True


def test_normal_tier_needs_no_ack(corpus_dir, store):
    pr = P.propose("notes", "ordinary note", corpus_dir, meta={"origin": "human"})
    store.put(pr)
    svc = _FakeService(); svc._live = pr.target_path
    res = C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir,
                    confirmed_by=OWNER)  # no ack required for normal
    assert res.ok is True


def test_editing_an_on_disk_protected_doc_requires_ack(corpus_dir, store):
    # The protection follows the EXISTING doc: even if the edit proposal doesn't
    # re-declare tier:protected, editing a doc that is protected ON DISK needs
    # the ack — a bot can't strip protection by omitting the tier.
    (corpus_dir / "notes-001.md").write_text(
        "---\nid: notes-001\nstatus: active\ntier: protected\n---\nidentity fact\n")
    pr = P.propose("notes", "sneaky change", corpus_dir, edit_id="notes-001",
                   meta={"origin": "bot-suggested"})  # tier omitted on purpose
    store.put(pr)
    svc = _FakeService(); svc._live = pr.target_path
    with pytest.raises(C.ConfirmError):
        C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir,
                  confirmed_by=OWNER)  # no ack → blocked because target is protected


def test_protected_edit_succeeds_with_human_and_ack(corpus_dir, store):
    (corpus_dir / "notes-001.md").write_text(
        "---\nid: notes-001\nstatus: active\ntier: protected\n---\nidentity fact\n")
    pr = P.propose("notes", "legit correction", corpus_dir, edit_id="notes-001",
                   meta={"origin": "human", "tier": "protected"})
    store.put(pr)
    svc = _FakeService(); svc._live = pr.target_path
    res = C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir,
                    confirmed_by=OWNER, protected_ack="notes-001")
    assert res.ok is True
