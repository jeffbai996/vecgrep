"""Confirmed edits, merges and deletes reach a doc under any registered spelling.

Source ids are the resolved file path at index time. When a corpus directory
is later reached through a symlink (VECGREP_HOME moved to another disk and
linked back), the registry keeps the pre-move spelling while propose() targets
the resolved path. Confirming an edit then registered the file a second time
and left the old chunks searchable beside the new ones; a confirmed delete
removed the file but none of its chunks. Retracted text stayed retrievable.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import vecgrep.backend.write.confirm as C
from vecgrep.backend import config as cfg_mod
from vecgrep.backend.embed.cache import CachedBackend
from vecgrep.backend.service import VecgrepService, _collection_for
from vecgrep.backend.write import proposal as P

CORPUS = "notes"
RETRACTED = "zanzibar"
DOCS = {
    "notes-001": "# Meeting date\n\nThe meeting was on July 25 at the zanzibar office.",
    "notes-002": "# Budget\n\nThe quarterly budget review covers hardware spend.",
}


def _legacy_corpus(svc: VecgrepService, tmp_path: Path) -> Path:
    """Index DOCS, then move their directory and link the old path to it."""
    home = tmp_path / "home"
    notes = home / CORPUS
    notes.mkdir(parents=True)
    for doc_id, body in DOCS.items():
        path = notes / f"{doc_id}.md"
        path.write_text(P.render_doc(doc_id, body, {
            "origin": "human", "status": "active", "tier": "normal",
            "corpus": CORPUS, "created_at": "2026-01-02T03:04:05+00:00",
        }))
        svc.index(str(path), CORPUS)
    moved = tmp_path / "moved-home"
    home.rename(moved)
    home.symlink_to(moved, target_is_directory=True)
    # The registry still names the files by their pre-move spelling.
    assert sorted(svc.registry.get(CORPUS).sources) == sorted(
        str(notes / f"{doc_id}.md") for doc_id in DOCS)
    return notes


def _vector_texts(svc: VecgrepService) -> list[str]:
    return [payload.get("text", "")
            for _id, payload in svc.store.iter_payloads(_collection_for(CORPUS))]


def _lexical_texts(svc: VecgrepService) -> list[str]:
    return [payload.get("text", "") for payload in svc.bm25.iter_payloads(CORPUS)]


def _files(svc: VecgrepService) -> list[Path]:
    return sorted(Path(s).resolve() for s in svc.registry.get(CORPUS).sources)


def _confirm(svc, store, pr, corpus_dir) -> C.ConfirmResult:
    store.put(pr)
    return C.confirm(pr.proposal_id, store, svc, CORPUS, corpus_dir,
                     confirmed_by="owner")


@pytest.fixture(params=["pickle", "sqlite"])
def any_bm25_svc(request, vg_home, stub_backend, monkeypatch):
    monkeypatch.setenv("VECGREP_BM25_BACKEND", request.param)
    monkeypatch.setattr(cfg_mod, "_settings", None)
    svc = VecgrepService(ephemeral=False)
    svc._backend_cache["auto"] = CachedBackend(stub_backend, svc._embed_cache)
    yield svc
    try:
        svc.store.client.close()
    except Exception:  # noqa: BLE001, S110 -- Test cleanup must not mask the alias assertions.
        pass


@pytest.fixture
def store(tmp_path):
    return C.ProposalStore(tmp_path / "pending")


def test_edit_confirm_replaces_the_doc_registered_under_its_old_spelling(
        any_bm25_svc, store, tmp_path):
    svc = any_bm25_svc
    notes = _legacy_corpus(svc, tmp_path)
    edited = DOCS["notes-001"].replace("July 25 at the zanzibar", "July 17 at the harbour")
    pr = P.propose(CORPUS, edited, notes, meta={"origin": "bot-suggested"},
                   edit_id="notes-001")

    res = _confirm(svc, store, pr, notes)

    assert res.ok, res.message
    assert _files(svc) == sorted((notes / f"{d}.md").resolve() for d in DOCS)
    for texts in (_vector_texts(svc), _lexical_texts(svc)):
        assert not any(RETRACTED in t for t in texts)
        assert any("harbour" in t for t in texts)


def test_delete_confirm_removes_the_chunks_of_a_doc_under_its_old_spelling(
        svc, store, tmp_path):
    notes = _legacy_corpus(svc, tmp_path)
    pr = P.propose_delete(CORPUS, "notes-001", notes, meta={"origin": "bot-suggested"})

    res = _confirm(svc, store, pr, notes)

    assert res.ok, res.message
    assert not (notes / "notes-001.md").exists()
    assert _files(svc) == [(notes / "notes-002.md").resolve()]
    for texts in (_vector_texts(svc), _lexical_texts(svc)):
        assert not any(RETRACTED in t for t in texts)


def test_merge_confirm_leaves_one_entry_for_the_canonical_doc_and_none_for_the_absorbed(
        svc, store, tmp_path):
    notes = _legacy_corpus(svc, tmp_path)
    pr = P.propose_merge(CORPUS, ["notes-001", "notes-002"],
                         "# Meeting and budget\n\nThe meeting was on July 17.",
                         notes, meta={"origin": "bot-suggested"})

    # The confirming CLI names the write dir by the resolved path, unlike the
    # proposing MCP server, so neither spelling can be assumed.
    res = _confirm(svc, store, pr, notes.resolve())

    assert res.ok, res.message
    assert _files(svc) == [(notes / "notes-001.md").resolve()]
    for texts in (_vector_texts(svc), _lexical_texts(svc)):
        assert not any(RETRACTED in t or "hardware" in t for t in texts)


def test_indexing_a_file_keeps_a_different_file_with_the_same_name(svc, tmp_path):
    first = tmp_path / "a" / "notes-001.md"
    second = tmp_path / "b" / "notes-001.md"
    for path, text in ((first, "alpha text"), (second, "beta text")):
        path.parent.mkdir()
        path.write_text(text)
        svc.index(str(path), CORPUS)

    first.write_text("alpha text, revised")
    svc.index(str(first), CORPUS)

    assert _files(svc) == sorted([first.resolve(), second.resolve()])
