"""Write-tool phase 4 — versioning / supersede (correctable, not destructible).

An edit never overwrites: confirming an update ARCHIVES the old content to a
frozen {doc_id}-vN.md snapshot (status: superseded) and advances {doc_id}.md to
the new version. Retrieval defaults to status:active so the live doc is current
truth, while every prior version stays on disk for the audit trail. The point:
you can correct a memory AND prove what it used to say and when it changed.
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
    def index(self, source, corpus_name, **kw): self.indexed.append(source); return (1, 1, 0)
    def search(self, query, corpus_name=None, top_k=5, **kw):
        r = type("R", (), {})(); r.source_id = self._live; return [r]


def _seed_v1(corpus_dir, store, svc, body="original truth"):
    pr = P.propose("notes", body, corpus_dir, meta={"origin": "human"})
    store.put(pr); svc._live = pr.target_path
    C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir)
    return pr.doc_id


def test_update_archives_old_version_as_superseded(corpus_dir, store):
    svc = _FakeService()
    doc_id = _seed_v1(corpus_dir, store, svc, "original truth")

    pr2 = P.propose("notes", "corrected truth", corpus_dir, update_id=doc_id,
                    meta={"origin": "human"})
    store.put(pr2); svc._live = pr2.target_path
    res = C.confirm(pr2.proposal_id, store, svc, "notes", corpus_dir)
    assert res.ok

    archived = corpus_dir / f"{doc_id}-v1.md"
    assert archived.exists(), "old version must be frozen to an archive file"
    assert "original truth" in archived.read_text()        # audit trail intact
    assert "status: superseded" in archived.read_text()    # marked superseded


def test_update_advances_live_file_to_new_version(corpus_dir, store):
    svc = _FakeService()
    doc_id = _seed_v1(corpus_dir, store, svc, "original truth")
    pr2 = P.propose("notes", "corrected truth", corpus_dir, update_id=doc_id,
                    meta={"origin": "human"})
    store.put(pr2); svc._live = pr2.target_path
    C.confirm(pr2.proposal_id, store, svc, "notes", corpus_dir)

    live = (corpus_dir / f"{doc_id}.md").read_text()
    assert "corrected truth" in live
    assert "original truth" not in live
    assert "version: 2" in live
    assert "status: active" in live


def test_supersede_pointers_set_both_ways(corpus_dir, store):
    svc = _FakeService()
    doc_id = _seed_v1(corpus_dir, store, svc)
    pr2 = P.propose("notes", "v2 body", corpus_dir, update_id=doc_id,
                    meta={"origin": "human"})
    store.put(pr2); svc._live = pr2.target_path
    C.confirm(pr2.proposal_id, store, svc, "notes", corpus_dir)

    live = (corpus_dir / f"{doc_id}.md").read_text()
    archived = (corpus_dir / f"{doc_id}-v1.md").read_text()
    assert "supersedes:" in live                       # v2 points back
    assert "superseded_by:" in archived                # v1 points forward


def test_original_content_never_destroyed(corpus_dir, store):
    svc = _FakeService()
    doc_id = _seed_v1(corpus_dir, store, svc, "the precious original")
    pr2 = P.propose("notes", "overwrite attempt", corpus_dir, update_id=doc_id,
                    meta={"origin": "human"})
    store.put(pr2); svc._live = pr2.target_path
    C.confirm(pr2.proposal_id, store, svc, "notes", corpus_dir)
    # The original text survives somewhere on disk (the archive).
    all_text = "\n".join(p.read_text() for p in corpus_dir.glob("*.md"))
    assert "the precious original" in all_text


def test_re_embeds_both_archived_and_live(corpus_dir, store):
    # Both the now-superseded archive and the new live file get re-embedded so
    # the index reflects the status flip (archive → superseded, won't surface).
    svc = _FakeService()
    doc_id = _seed_v1(corpus_dir, store, svc)
    pr2 = P.propose("notes", "v2", corpus_dir, update_id=doc_id, meta={"origin": "human"})
    store.put(pr2); svc._live = pr2.target_path
    svc.indexed.clear()
    C.confirm(pr2.proposal_id, store, svc, "notes", corpus_dir)
    indexed = " ".join(svc.indexed)
    assert f"{doc_id}.md" in indexed          # live re-embedded
    assert f"{doc_id}-v1.md" in indexed       # archive re-embedded (status flip)


def test_second_update_archives_v2_not_clobber_v1(corpus_dir, store):
    svc = _FakeService()
    doc_id = _seed_v1(corpus_dir, store, svc, "v1 body")
    for body, ver in [("v2 body", 2), ("v3 body", 3)]:
        pr = P.propose("notes", body, corpus_dir, update_id=doc_id, meta={"origin": "human"})
        store.put(pr); svc._live = pr.target_path
        C.confirm(pr.proposal_id, store, svc, "notes", corpus_dir)
    # v1 and v2 archives both exist; live is v3.
    assert (corpus_dir / f"{doc_id}-v1.md").exists()
    assert (corpus_dir / f"{doc_id}-v2.md").exists()
    assert "v3 body" in (corpus_dir / f"{doc_id}.md").read_text()
    assert "v1 body" in (corpus_dir / f"{doc_id}-v1.md").read_text()
