"""Write-tool phase 5 — THE WALL (anti-poisoning). Adversarial.

The load-bearing safety property: ingested/retrieved content can NEVER become a
write, and a bot can PROPOSE but never AUTHORIZE. Provenance must trace to a
human. These tests are written as ATTACKS — each describes something a poisoned
chunk / prompt-injection / rogue bot would try, and asserts it is BLOCKED. If
any of these ever passes-through, the wall is breached.
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


# ─────────── ATTACK 1: ingested content trying to become a write ───────────

def test_ingested_origin_is_rejected_at_propose(corpus_dir):
    # A chunk a bot READ (origin traces to retrieval) must not be proposable.
    with pytest.raises(P.ProposalError):
        P.propose("notes", "poisoned payload", corpus_dir,
                  meta={"origin": "ingested"})


def test_retrieved_origin_is_rejected_at_propose(corpus_dir):
    with pytest.raises(P.ProposalError):
        P.propose("notes", "x", corpus_dir, meta={"origin": "retrieved"})


def test_only_human_or_bot_suggested_origins_allowed(corpus_dir):
    # The ONLY two legal provenances. Anything else (the attack surface) fails.
    for bad in ["tool", "chunk", "web", "auto", "system", ""]:
        with pytest.raises(P.ProposalError):
            P.propose("notes", "x", corpus_dir, meta={"origin": bad})
    # the two legal ones construct fine
    assert P.propose("notes", "x", corpus_dir, meta={"origin": "human"})
    assert P.propose("notes", "x", corpus_dir, meta={"origin": "bot-suggested"})


# ─────────── ATTACK 2: a bot authorizing its own write ───────────

def test_bot_suggested_cannot_confirm_without_human(corpus_dir, store):
    # A bot proposes (fine) then tries to confirm with no human → BLOCKED.
    pr = P.propose("notes", "bot wants this saved", corpus_dir,
                   meta={"origin": "bot-suggested"})
    store.put(pr)
    svc = _FakeService()
    with pytest.raises(C.ConfirmError):
        C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir,
                  confirmed_by=None)


def test_confirm_requires_a_human_identity(corpus_dir, store):
    pr = P.propose("notes", "x", corpus_dir, meta={"origin": "human"})
    store.put(pr)
    svc = _FakeService()
    for bad in [None, "", "   "]:
        store.put(pr)  # re-put (confirm consumes it)
        with pytest.raises(C.ConfirmError):
            C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir,
                      confirmed_by=bad)


# ─────────── happy path: human-authorized write still works ───────────

def test_human_confirmed_write_succeeds(corpus_dir, store):
    pr = P.propose("notes", "a fact the human stated", corpus_dir,
                   meta={"origin": "human"})
    store.put(pr)
    svc = _FakeService(); svc._live = pr.target_path
    res = C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir,
                    confirmed_by=OWNER)
    assert res.ok is True


def test_bot_suggested_write_succeeds_WITH_human_confirm(corpus_dir, store):
    # The legit path: a bot proposes, a HUMAN confirms → allowed.
    pr = P.propose("notes", "bot noticed this", corpus_dir,
                   meta={"origin": "bot-suggested"})
    store.put(pr)
    svc = _FakeService(); svc._live = pr.target_path
    res = C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir,
                    confirmed_by=OWNER)
    assert res.ok is True


def test_confirmer_identity_recorded_in_written_doc(corpus_dir, store):
    pr = P.propose("notes", "audit me", corpus_dir, meta={"origin": "human"})
    store.put(pr)
    svc = _FakeService(); svc._live = pr.target_path
    C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir, confirmed_by=OWNER)
    written = (corpus_dir / f"{pr.doc_id}.md").read_text()
    assert f"confirmed_by: {OWNER}" in written  # provenance recorded on disk
