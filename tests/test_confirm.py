from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import vecgrep.backend.write.confirm as C_write_confirm
import vecgrep.backend.write.confirm as C_write_protected
import vecgrep.backend.write.confirm as C_write_source_alias
import vecgrep.backend.write.confirm as C_write_wall
import vecgrep.backend.write.confirm as confirm_module
from vecgrep.backend import config as cfg_mod
from vecgrep.backend.embed.cache import CachedBackend
from vecgrep.backend.ingestion.adapters.markdown import parse_frontmatter
from vecgrep.backend.service import VecgrepService, _collection_for
from vecgrep.backend.write import confirm as C
from vecgrep.backend.write import proposal as P
from vecgrep.backend.write import proposal as proposal_module
from vecgrep.cli import main as cli_main

# ── test_confirm_merge ──

class FakeSvc:
    def __init__(self, fail_index: bool = False):
        self.calls = []
        self.fail_index = fail_index

    def index(self, path, corpus):
        self.calls.append(("index", path))
        if self.fail_index:
            raise RuntimeError("embed down")

    def delete_file(self, corpus, path):
        self.calls.append(("delete_file", path))

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
    # Ordering: the canonical index() precedes the absorbed delete_file().
    ops = [op for op, _ in svc.calls]
    assert ops.index("index") < ops.index("delete_file")
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

# ── test_write_protected ──

@pytest.fixture
def corpus_dir(tmp_path):
    d = tmp_path / "notes"; d.mkdir(); return d


@pytest.fixture
def store(tmp_path):
    return C_write_protected.ProposalStore(tmp_path / "pending")


class _FakeResult:
    source_id: str | None = None


class _FakeService:
    _live: str | None = None

    def __init__(self): self.indexed = []
    def index(self, s, c, **k): self.indexed.append(s); return (1, 1, 0)
    def search(self, q, corpus_name=None, top_k=5, **k):
        r = _FakeResult(); r.source_id = self._live; return [r]


OWNER = "owner-123"


def _put_protected_proposal(corpus_dir, store):
    pr = P.propose("notes", "safety-critical fact", corpus_dir,
                   meta={"origin": "human", "tier": "protected"})
    store.put(pr)
    return pr


def test_protected_write_rejected_without_id_ack(corpus_dir, store):
    pr = _put_protected_proposal(corpus_dir, store)
    svc = _FakeService(); svc._live = pr.target_path
    with pytest.raises(C_write_protected.ConfirmError):
        C_write_protected.confirm(pr.proposal_id, store, svc, "notes", corpus_dir,
                  confirmed_by=OWNER)  # no protected_ack


def test_protected_write_rejected_with_wrong_id_ack(corpus_dir, store):
    pr = _put_protected_proposal(corpus_dir, store)
    svc = _FakeService(); svc._live = pr.target_path
    with pytest.raises(C_write_protected.ConfirmError):
        C_write_protected.confirm(pr.proposal_id, store, svc, "notes", corpus_dir,
                  confirmed_by=OWNER, protected_ack="wrong-id")


def test_protected_write_succeeds_with_correct_id_ack(corpus_dir, store):
    pr = _put_protected_proposal(corpus_dir, store)
    svc = _FakeService(); svc._live = pr.target_path
    res = C_write_protected.confirm(pr.proposal_id, store, svc, "notes", corpus_dir,
                    confirmed_by=OWNER, protected_ack=pr.doc_id)
    assert res.ok is True


def test_normal_tier_needs_no_ack(corpus_dir, store):
    pr = P.propose("notes", "ordinary note", corpus_dir, meta={"origin": "human"})
    store.put(pr)
    svc = _FakeService(); svc._live = pr.target_path
    res = C_write_protected.confirm(pr.proposal_id, store, svc, "notes", corpus_dir,
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
    with pytest.raises(C_write_protected.ConfirmError):
        C_write_protected.confirm(pr.proposal_id, store, svc, "notes", corpus_dir,
                  confirmed_by=OWNER)  # no ack → blocked because target is protected


def test_protected_edit_succeeds_with_human_and_ack(corpus_dir, store):
    (corpus_dir / "notes-001.md").write_text(
        "---\nid: notes-001\nstatus: active\ntier: protected\n---\nidentity fact\n")
    pr = P.propose("notes", "legit correction", corpus_dir, edit_id="notes-001",
                   meta={"origin": "human", "tier": "protected"})
    store.put(pr)
    svc = _FakeService(); svc._live = pr.target_path
    res = C_write_protected.confirm(pr.proposal_id, store, svc, "notes", corpus_dir,
                    confirmed_by=OWNER, protected_ack="notes-001")
    assert res.ok is True

# ── test_write_wall ──

@pytest.fixture
def corpus_dir_write_wall(tmp_path):
    d = tmp_path / "notes"; d.mkdir(); return d


@pytest.fixture
def store_write_wall(tmp_path):
    return C_write_wall.ProposalStore(tmp_path / "pending")


class _FakeResult_write_wall:
    source_id: str | None = None


class _FakeService_write_wall:
    _live: str | None = None

    def __init__(self): self.indexed = []
    def index(self, s, c, **k): self.indexed.append(s); return (1, 1, 0)
    def search(self, q, corpus_name=None, top_k=5, **k):
        r = _FakeResult_write_wall(); r.source_id = self._live; return [r]


OWNER_write_wall = "owner-123"


# ─────────── ATTACK 1: ingested content trying to become a write ───────────

def test_ingested_origin_is_rejected_at_propose(corpus_dir_write_wall):
    # A chunk a bot READ (origin traces to retrieval) must not be proposable.
    with pytest.raises(P.ProposalError):
        P.propose("notes", "poisoned payload", corpus_dir_write_wall,
                  meta={"origin": "ingested"})


def test_retrieved_origin_is_rejected_at_propose(corpus_dir_write_wall):
    with pytest.raises(P.ProposalError):
        P.propose("notes", "x", corpus_dir_write_wall, meta={"origin": "retrieved"})


def test_only_human_or_bot_suggested_origins_allowed(corpus_dir_write_wall):
    # The ONLY two legal provenances. Anything else (the attack surface) fails.
    for bad in ["tool", "chunk", "web", "auto", "system", ""]:
        with pytest.raises(P.ProposalError):
            P.propose("notes", "x", corpus_dir_write_wall, meta={"origin": bad})
    # the two legal ones construct fine
    assert P.propose("notes", "x", corpus_dir_write_wall, meta={"origin": "human"})
    assert P.propose("notes", "x", corpus_dir_write_wall, meta={"origin": "bot-suggested"})


# ─────────── ATTACK 2: a bot authorizing its own write ───────────

def test_bot_suggested_cannot_confirm_without_human(corpus_dir_write_wall, store_write_wall):
    # A bot proposes (fine) then tries to confirm with no human → BLOCKED.
    pr = P.propose("notes", "bot wants this saved", corpus_dir_write_wall,
                   meta={"origin": "bot-suggested"})
    store_write_wall.put(pr)
    svc = _FakeService_write_wall()
    with pytest.raises(C_write_wall.ConfirmError):
        C_write_wall.confirm(pr.proposal_id, store_write_wall, svc, "notes", corpus_dir_write_wall,
                  confirmed_by=None)


def test_confirm_requires_a_human_identity(corpus_dir_write_wall, store_write_wall):
    pr = P.propose("notes", "x", corpus_dir_write_wall, meta={"origin": "human"})
    store_write_wall.put(pr)
    svc = _FakeService_write_wall()
    for bad in [None, "", "   "]:
        store_write_wall.put(pr)  # re-put (confirm consumes it)
        with pytest.raises(C_write_wall.ConfirmError):
            C_write_wall.confirm(pr.proposal_id, store_write_wall, svc, "notes", corpus_dir_write_wall,
                      confirmed_by=bad)


# ─────────── happy path: human-authorized write still works ───────────

def test_human_confirmed_write_succeeds(corpus_dir_write_wall, store_write_wall):
    pr = P.propose("notes", "a fact the human stated", corpus_dir_write_wall,
                   meta={"origin": "human"})
    store_write_wall.put(pr)
    svc = _FakeService_write_wall(); svc._live = pr.target_path
    res = C_write_wall.confirm(pr.proposal_id, store_write_wall, svc, "notes", corpus_dir_write_wall,
                    confirmed_by=OWNER_write_wall)
    assert res.ok is True


def test_bot_suggested_write_succeeds_WITH_human_confirm(corpus_dir_write_wall, store_write_wall):
    # The legit path: a bot proposes, a HUMAN confirms → allowed.
    pr = P.propose("notes", "bot noticed this", corpus_dir_write_wall,
                   meta={"origin": "bot-suggested"})
    store_write_wall.put(pr)
    svc = _FakeService_write_wall(); svc._live = pr.target_path
    res = C_write_wall.confirm(pr.proposal_id, store_write_wall, svc, "notes", corpus_dir_write_wall,
                    confirmed_by=OWNER_write_wall)
    assert res.ok is True


def test_confirmer_identity_recorded_in_written_doc(corpus_dir_write_wall, store_write_wall):
    pr = P.propose("notes", "audit me", corpus_dir_write_wall, meta={"origin": "human"})
    store_write_wall.put(pr)
    svc = _FakeService_write_wall(); svc._live = pr.target_path
    C_write_wall.confirm(pr.proposal_id, store_write_wall, svc, "notes", corpus_dir_write_wall, confirmed_by=OWNER_write_wall)
    written = (corpus_dir_write_wall / f"{pr.doc_id}.md").read_text()
    assert f"confirmed_by: {OWNER_write_wall}" in written  # provenance recorded on disk

# ── test_write_source_alias ──

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


def _confirm(svc, store_write_source_alias, pr, corpus_dir) -> C_write_source_alias.ConfirmResult:
    store_write_source_alias.put(pr)
    return C_write_source_alias.confirm(pr.proposal_id, store_write_source_alias, svc, CORPUS, corpus_dir,
                     confirmed_by="owner")


@pytest.fixture(params=["pickle", "sqlite"])
def any_bm25_svc(request, vg_home, stub_backend, monkeypatch):
    monkeypatch.setenv("VECGREP_BM25_BACKEND", request.param)
    monkeypatch.setattr(cfg_mod, "_settings", None)
    svc = VecgrepService(ephemeral=False)
    assert svc._embed_cache is not None
    svc._backend_cache["auto"] = CachedBackend(stub_backend, svc._embed_cache)
    yield svc
    try:
        svc.store.client.close()
    except Exception:  # noqa: BLE001, S110 -- Test cleanup must not mask the alias assertions.
        pass


@pytest.fixture
def store_write_source_alias(tmp_path):
    return C_write_source_alias.ProposalStore(tmp_path / "pending")


def test_edit_confirm_replaces_the_doc_registered_under_its_old_spelling(
        any_bm25_svc, store_write_source_alias, tmp_path):
    svc = any_bm25_svc
    notes = _legacy_corpus(svc, tmp_path)
    edited = DOCS["notes-001"].replace("July 25 at the zanzibar", "July 17 at the harbour")
    pr = P.propose(CORPUS, edited, notes, meta={"origin": "bot-suggested"},
                   edit_id="notes-001")

    res = _confirm(svc, store_write_source_alias, pr, notes)

    assert res.ok, res.message
    assert _files(svc) == sorted((notes / f"{d}.md").resolve() for d in DOCS)
    for texts in (_vector_texts(svc), _lexical_texts(svc)):
        assert not any(RETRACTED in t for t in texts)
        assert any("harbour" in t for t in texts)


def test_delete_confirm_removes_the_chunks_of_a_doc_under_its_old_spelling(
        svc, store_write_source_alias, tmp_path):
    notes = _legacy_corpus(svc, tmp_path)
    pr = P.propose_delete(CORPUS, "notes-001", notes, meta={"origin": "bot-suggested"})

    res = _confirm(svc, store_write_source_alias, pr, notes)

    assert res.ok, res.message
    assert not (notes / "notes-001.md").exists()
    assert _files(svc) == [(notes / "notes-002.md").resolve()]
    for texts in (_vector_texts(svc), _lexical_texts(svc)):
        assert not any(RETRACTED in t for t in texts)


def test_merge_confirm_leaves_one_entry_for_the_canonical_doc_and_none_for_the_absorbed(
        svc, store_write_source_alias, tmp_path):
    notes = _legacy_corpus(svc, tmp_path)
    pr = P.propose_merge(CORPUS, ["notes-001", "notes-002"],
                         "# Meeting and budget\n\nThe meeting was on July 17.",
                         notes, meta={"origin": "bot-suggested"})

    # The confirming CLI names the write dir by the resolved path, unlike the
    # proposing MCP server, so neither spelling can be assumed.
    res = _confirm(svc, store_write_source_alias, pr, notes.resolve())

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

# ── test_write_confirm ──

# Import the submodule explicitly — the package __init__ re-exports a `confirm`
# FUNCTION, which would shadow the module name under `from ... import confirm`.


@pytest.fixture
def corpus_dir_write_confirm(tmp_path):
    d = tmp_path / "notes"
    d.mkdir()
    return d


@pytest.fixture
def store_write_confirm(tmp_path):
    return C_write_confirm.ProposalStore(tmp_path / "pending")


class _FakeResult_write_confirm:
    source_id: str | None = None


class _FakeService_write_confirm:
    """Stand-in for VecgrepService: records index()/delete_file() calls, fakes
    search verify."""

    _last_written: str | None = None

    def __init__(self, found=True):
        self.indexed = []
        self.deleted = []
        self._found = found

    def index(self, source, corpus_name, **kw):
        self.indexed.append((source, corpus_name))
        return (1, 1, 0)

    def delete_file(self, corpus_name, path):
        self.deleted.append((corpus_name, path))

    def search(self, query, corpus_name=None, top_k=5, **kw):
        if not self._found:
            return []
        r = _FakeResult_write_confirm()
        r.source_id = self._last_written  # set by the test via attribute
        return [r]


def test_confirm_requires_known_proposal(corpus_dir_write_confirm, store_write_confirm):
    svc = _FakeService_write_confirm()
    with pytest.raises(C_write_confirm.ConfirmError):
        C_write_confirm.confirm("prop-does-not-exist", store_write_confirm, svc, "notes", corpus_dir_write_confirm, confirmed_by="owner")


def test_confirm_writes_the_file(corpus_dir_write_confirm, store_write_confirm):
    pr = P.propose("notes", "a real note", corpus_dir_write_confirm, meta={"origin": "human"})
    store_write_confirm.put(pr)
    svc = _FakeService_write_confirm()
    svc._last_written = pr.target_path
    res = C_write_confirm.confirm(pr.proposal_id, store_write_confirm, svc, "notes", corpus_dir_write_confirm, confirmed_by="owner")
    written = Path(pr.target_path)
    assert written.exists()
    assert "a real note" in written.read_text()
    assert res.ok is True


def test_new_write_refuses_to_clobber_existing_id(corpus_dir_write_confirm, store_write_confirm):
    # A NEW entry (not an edit) must never silently overwrite an existing doc id.
    pr = P.propose("notes", "first", corpus_dir_write_confirm, meta={"origin": "human"})
    store_write_confirm.put(pr)
    Path(pr.target_path).write_text("PRE-EXISTING — must not be lost")
    svc = _FakeService_write_confirm()
    with pytest.raises(C_write_confirm.ConfirmError):
        C_write_confirm.confirm(pr.proposal_id, store_write_confirm, svc, "notes", corpus_dir_write_confirm, confirmed_by="owner")
    assert "PRE-EXISTING" in Path(pr.target_path).read_text()


def test_edit_overwrites_in_place(corpus_dir_write_confirm, store_write_confirm):
    # An EDIT targets an existing id on purpose and overwrites it (simple, no
    # versioning). Seed a doc, then edit it.
    (corpus_dir_write_confirm / "notes-001.md").write_text("---\nid: notes-001\nstatus: active\n---\nold\n")
    pr = P.propose("notes", "new content", corpus_dir_write_confirm, edit_id="notes-001",
                   meta={"origin": "human"})
    store_write_confirm.put(pr)
    svc = _FakeService_write_confirm(); svc._last_written = pr.target_path
    res = C_write_confirm.confirm(pr.proposal_id, store_write_confirm, svc, "notes", corpus_dir_write_confirm, confirmed_by="owner")
    assert res.ok is True
    body = (corpus_dir_write_confirm / "notes-001.md").read_text()
    assert "new content" in body and "old" not in body


def test_edit_confirm_stamps_edited_at_and_the_confirmer(corpus_dir_write_confirm, store_write_confirm):
    (corpus_dir_write_confirm / "notes-001.md").write_text(
        "---\nid: notes-001\ncreated_at: 2026-01-02\nconfirmed_by: someone\n---\nold\n")
    pr = P.propose("notes", "new content", corpus_dir_write_confirm, edit_id="notes-001",
                   meta={"origin": "human"})
    store_write_confirm.put(pr)
    svc = _FakeService_write_confirm(); svc._last_written = pr.target_path
    C_write_confirm.confirm(pr.proposal_id, store_write_confirm, svc, "notes", corpus_dir_write_confirm, confirmed_by="owner")
    fm = parse_frontmatter((corpus_dir_write_confirm / "notes-001.md").read_text())
    assert fm["created_at"] == "2026-01-02"
    assert fm["confirmed_by"] == "owner"
    assert fm["edited_at"]


def test_edit_of_missing_target_rejected(corpus_dir_write_confirm, store_write_confirm):
    pr = P.propose("notes", "x", corpus_dir_write_confirm, edit_id="notes-999", meta={"origin": "human"})
    store_write_confirm.put(pr)
    svc = _FakeService_write_confirm()
    with pytest.raises(C_write_confirm.ConfirmError):
        C_write_confirm.confirm(pr.proposal_id, store_write_confirm, svc, "notes", corpus_dir_write_confirm, confirmed_by="owner")


def test_confirm_reembeds_the_written_file(corpus_dir_write_confirm, store_write_confirm):
    pr = P.propose("notes", "embed me", corpus_dir_write_confirm, meta={"origin": "human"})
    store_write_confirm.put(pr)
    svc = _FakeService_write_confirm()
    svc._last_written = pr.target_path
    C_write_confirm.confirm(pr.proposal_id, store_write_confirm, svc, "notes", corpus_dir_write_confirm, confirmed_by="owner")
    assert svc.indexed == [(pr.target_path, "notes")]  # re-embedded exactly the new file


def test_confirm_flags_when_verify_fails(corpus_dir_write_confirm, store_write_confirm):
    # Written but not retrievable → result flags it, does not silently succeed.
    pr = P.propose("notes", "ghost note", corpus_dir_write_confirm, meta={"origin": "human"})
    store_write_confirm.put(pr)
    svc = _FakeService_write_confirm(found=False)
    res = C_write_confirm.confirm(pr.proposal_id, store_write_confirm, svc, "notes", corpus_dir_write_confirm, confirmed_by="owner")
    assert res.ok is False
    assert "verif" in res.message.lower()
    assert Path(pr.target_path).exists()  # the file was still written


def test_confirm_consumes_the_proposal(corpus_dir_write_confirm, store_write_confirm):
    # After a successful confirm the proposal is gone — a replayed confirm fails.
    pr = P.propose("notes", "once", corpus_dir_write_confirm, meta={"origin": "human"})
    store_write_confirm.put(pr)
    svc = _FakeService_write_confirm()
    svc._last_written = pr.target_path
    C_write_confirm.confirm(pr.proposal_id, store_write_confirm, svc, "notes", corpus_dir_write_confirm, confirmed_by="owner")
    with pytest.raises(C_write_confirm.ConfirmError):
        C_write_confirm.confirm(pr.proposal_id, store_write_confirm, svc, "notes", corpus_dir_write_confirm, confirmed_by="owner")


def test_proposal_store_roundtrip(store_write_confirm, corpus_dir_write_confirm):
    pr = P.propose("notes", "persist me", corpus_dir_write_confirm)
    store_write_confirm.put(pr)
    got = store_write_confirm.get(pr.proposal_id)
    assert got is not None
    assert got.doc_id == pr.doc_id
    assert got.rendered == pr.rendered


# --- delete proposals: confirm removes the file + de-indexes ---

def test_confirm_delete_removes_file_and_deindexes(corpus_dir_write_confirm, store_write_confirm):
    # Seed a doc, propose its delete, confirm → file gone + delete_file called.
    pr_w = P.propose("notes", "to be deleted", corpus_dir_write_confirm, meta={"origin": "human"})
    store_write_confirm.put(pr_w)
    svc = _FakeService_write_confirm()
    svc._last_written = pr_w.target_path
    C_write_confirm.confirm(pr_w.proposal_id, store_write_confirm, svc, "notes", corpus_dir_write_confirm, confirmed_by="owner")
    assert Path(pr_w.target_path).exists()

    pr_d = P.propose_delete("notes", pr_w.doc_id, corpus_dir_write_confirm, meta={"origin": "human"})
    store_write_confirm.put(pr_d)
    res = C_write_confirm.confirm(pr_d.proposal_id, store_write_confirm, svc, "notes", corpus_dir_write_confirm,
                    confirmed_by="owner")
    assert res.ok and "deleted" in res.message
    assert not Path(pr_w.target_path).exists()          # file removed
    assert (("notes", pr_w.target_path) in svc.deleted)  # embeddings removed
    assert store_write_confirm.get(pr_d.proposal_id) is None           # proposal consumed


def test_confirm_delete_requires_human_confirmer(corpus_dir_write_confirm, store_write_confirm):
    # The wall holds for deletes too: no confirmer identity → no delete.
    (corpus_dir_write_confirm / "notes-001.md").write_text("---\nid: notes-001\n---\n\nx")
    pr = P.propose_delete("notes", "notes-001", corpus_dir_write_confirm, meta={"origin": "human"})
    store_write_confirm.put(pr)
    with pytest.raises(C_write_confirm.ConfirmError):
        C_write_confirm.confirm(pr.proposal_id, store_write_confirm, _FakeService_write_confirm(), "notes", corpus_dir_write_confirm,
                  confirmed_by="")  # no human → rejected, file untouched
    assert (corpus_dir_write_confirm / "notes-001.md").exists()


def test_confirm_delete_protected_needs_ack(corpus_dir_write_confirm, store_write_confirm):
    # A protected doc can't be deleted without re-stating its exact id.
    (corpus_dir_write_confirm / "notes-001.md").write_text(
        "---\nid: notes-001\ntier: protected\n---\n\nimportant")
    pr = P.propose_delete("notes", "notes-001", corpus_dir_write_confirm, meta={"origin": "human"})
    store_write_confirm.put(pr)
    svc = _FakeService_write_confirm()
    with pytest.raises(C_write_confirm.ConfirmError):
        C_write_confirm.confirm(pr.proposal_id, store_write_confirm, svc, "notes", corpus_dir_write_confirm,
                  confirmed_by="owner", protected_ack="wrong")
    assert (corpus_dir_write_confirm / "notes-001.md").exists()  # still there
    # correct ack → deletes
    res = C_write_confirm.confirm(pr.proposal_id, store_write_confirm, svc, "notes", corpus_dir_write_confirm,
                    confirmed_by="owner", protected_ack="notes-001")
    assert res.ok and not (corpus_dir_write_confirm / "notes-001.md").exists()


def test_confirm_delete_deindex_failure_still_removes_file(corpus_dir_write_confirm, store_write_confirm):
    # If de-indexing raises, the file is STILL removed and the warning surfaced
    # (orphaned vectors are recoverable by a reindex; an undeleted file isn't the
    # user's intent).
    (corpus_dir_write_confirm / "notes-001.md").write_text("---\nid: notes-001\n---\n\nx")
    pr = P.propose_delete("notes", "notes-001", corpus_dir_write_confirm, meta={"origin": "human"})
    store_write_confirm.put(pr)

    class _Boom(_FakeService_write_confirm):
        def delete_file(self, corpus_name, path):
            raise RuntimeError("qdrant down")

    res = C_write_confirm.confirm(pr.proposal_id, store_write_confirm, _Boom(), "notes", corpus_dir_write_confirm,
                    confirmed_by="owner")
    assert res.ok and "de-index warning" in res.message
    assert not (corpus_dir_write_confirm / "notes-001.md").exists()


def test_proposal_store_roundtrips_is_delete(store_write_confirm, corpus_dir_write_confirm):
    # The new is_delete flag must survive the JSON round-trip in the store.
    (corpus_dir_write_confirm / "notes-001.md").write_text("---\nid: notes-001\n---\n\nx")
    pr = P.propose_delete("notes", "notes-001", corpus_dir_write_confirm)
    store_write_confirm.put(pr)
    got = store_write_confirm.get(pr.proposal_id)
    assert got is not None and got.is_delete is True


# --- write-through: confirm routes a mirror-corpus operation upstream ---

def test_writethrough_edit_routes_upstream_skips_mirror(corpus_dir_write_confirm, store_write_confirm, monkeypatch, tmp_path):
    # A mirror doc exists locally; with a write-through cmd set, confirm must NOT
    # touch the mirror — it runs the cmd with the op JSON and consumes the proposal.
    (corpus_dir_write_confirm / "memory-103.md").write_text("---\nid: memory-103\n---\n\nold body")
    pr = P.propose("shared-memory", "new body", corpus_dir_write_confirm, edit_id="memory-103",
                   meta={"origin": "human"})
    store_write_confirm.put(pr)
    seen = tmp_path / "seen.json"
    # The env key is derived from the corpus name.
    monkeypatch.setenv("VECGREP_WRITETHROUGH_SHARED_MEMORY", f"cat > {seen}")
    svc = _FakeService_write_confirm()
    res = C_write_confirm.confirm(pr.proposal_id, store_write_confirm, svc, "shared-memory", corpus_dir_write_confirm,
                    confirmed_by="owner")
    assert res.ok and "routed upstream" in res.message
    # the op JSON reached the command
    got = json.loads(seen.read_text())
    assert got["op"] == "edit" and got["doc_id"] == "memory-103"
    assert got["body"] == "new body" and got["confirmed_by"] == "owner"
    # mirror untouched (still old), no embed, proposal consumed
    assert "old body" in (corpus_dir_write_confirm / "memory-103.md").read_text()
    assert svc.indexed == []
    assert store_write_confirm.get(pr.proposal_id) is None


def test_writethrough_delete_routes_upstream(corpus_dir_write_confirm, store_write_confirm, monkeypatch, tmp_path):
    (corpus_dir_write_confirm / "memory-50.md").write_text("---\nid: memory-50\n---\n\nx")
    pr = P.propose_delete("shared-memory", "memory-50", corpus_dir_write_confirm, meta={"origin": "human"})
    store_write_confirm.put(pr)
    seen = tmp_path / "seen.json"
    monkeypatch.setenv("VECGREP_WRITETHROUGH_SHARED_MEMORY", f"cat > {seen}")
    svc = _FakeService_write_confirm()
    res = C_write_confirm.confirm(pr.proposal_id, store_write_confirm, svc, "shared-memory", corpus_dir_write_confirm,
                    confirmed_by="owner")
    assert res.ok
    got = json.loads(seen.read_text())
    assert got["op"] == "delete" and got["doc_id"] == "memory-50"
    # the mirror file is NOT removed locally (the upstream's dump owns it)
    assert (corpus_dir_write_confirm / "memory-50.md").exists()
    assert svc.deleted == []  # de-index skipped — upstream + dump reconcile


def test_writethrough_failure_does_not_consume_proposal(corpus_dir_write_confirm, store_write_confirm, monkeypatch):
    # If the upstream command fails, the proposal must SURVIVE so the human can
    # retry — a failed write-through is not a silent drop.
    (corpus_dir_write_confirm / "memory-7.md").write_text("---\nid: memory-7\n---\n\nx")
    pr = P.propose("shared-memory", "new", corpus_dir_write_confirm, edit_id="memory-7",
                   meta={"origin": "human"})
    store_write_confirm.put(pr)
    monkeypatch.setenv("VECGREP_WRITETHROUGH_SHARED_MEMORY", "exit 3")
    res = C_write_confirm.confirm(pr.proposal_id, store_write_confirm, _FakeService_write_confirm(), "shared-memory",
                    corpus_dir_write_confirm, confirmed_by="owner")
    assert res.ok is False and "write-through failed" in res.message
    assert store_write_confirm.get(pr.proposal_id) is not None  # NOT consumed — retryable


def test_writethrough_protected_needs_ack(corpus_dir_write_confirm, store_write_confirm, monkeypatch, tmp_path):
    # A protected mirror doc still needs the exact-id ack before routing upstream.
    (corpus_dir_write_confirm / "memory-9.md").write_text(
        "---\nid: memory-9\ntier: protected\n---\n\nimportant")
    pr = P.propose("shared-memory", "x", corpus_dir_write_confirm, edit_id="memory-9",
                   meta={"origin": "human"})
    store_write_confirm.put(pr)
    seen = tmp_path / "seen.json"
    monkeypatch.setenv("VECGREP_WRITETHROUGH_SHARED_MEMORY", f"cat > {seen}")
    with pytest.raises(C_write_confirm.ConfirmError):
        C_write_confirm.confirm(pr.proposal_id, store_write_confirm, _FakeService_write_confirm(), "shared-memory", corpus_dir_write_confirm,
                  confirmed_by="owner", protected_ack="wrong")
    assert not seen.exists()  # never reached the command


def test_no_writethrough_env_uses_normal_path(corpus_dir_write_confirm, store_write_confirm, monkeypatch):
    # Without a write-through cmd for the corpus, confirm writes the mirror as usual.
    monkeypatch.delenv("VECGREP_WRITETHROUGH_SHARED_MEMORY", raising=False)
    pr = P.propose("shared-memory", "local body", corpus_dir_write_confirm, meta={"origin": "human"})
    store_write_confirm.put(pr)
    svc = _FakeService_write_confirm(); svc._last_written = pr.target_path
    res = C_write_confirm.confirm(pr.proposal_id, store_write_confirm, svc, "shared-memory", corpus_dir_write_confirm,
                    confirmed_by="owner")
    assert res.ok and Path(pr.target_path).exists()  # normal local write
    assert svc.indexed == [(pr.target_path, "shared-memory")]


def test_writethrough_payload_carries_meta(corpus_dir_write_confirm, store_write_confirm, monkeypatch, tmp_path):
    # meta (source_kind, tags, ...) must reach the write-through command so it can
    # route a NEW entry to the right upstream record type, not just edit by id.
    pr = P.propose("shared-memory", "remember this", corpus_dir_write_confirm,
                   meta={"origin": "human", "source_kind": "memory",
                         "tags": ["infra", "vecgrep"]})
    store_write_confirm.put(pr)
    seen = tmp_path / "seen.json"
    monkeypatch.setenv("VECGREP_WRITETHROUGH_SHARED_MEMORY", f"cat > {seen}")
    svc = _FakeService_write_confirm()
    res = C_write_confirm.confirm(pr.proposal_id, store_write_confirm, svc, "shared-memory", corpus_dir_write_confirm,
                    confirmed_by="owner")
    assert res.ok
    got = json.loads(seen.read_text())
    assert got["op"] == "write"
    assert got["meta"]["source_kind"] == "memory"
    assert got["meta"]["tags"] == ["infra", "vecgrep"]



def test_confirm_verifies_retrievability_with_the_body_not_the_frontmatter(svc, corpus_dir_write_confirm, store_write_confirm):
    # The probe used to be the first line of the rendered doc, the `---` fence,
    # which finds nothing, so every real confirm reported a failed verify.
    pr = P.propose("notes", "# Harbour office move\n\nThe team moves in May.", corpus_dir_write_confirm,
                   meta={"origin": "human"})
    store_write_confirm.put(pr)
    res = C_write_confirm.confirm(pr.proposal_id, store_write_confirm, svc, "notes", corpus_dir_write_confirm, confirmed_by="owner")
    assert res.ok, res.message

# ── test_cli_confirm_protected_ack ──

class FakeService:
    def __init__(self, *args, **kwargs):
        self.indexed = ""

    def index(self, source, _corpus, **_kwargs):
        self.indexed = str(source)
        return (1, 1, 0)

    def search(self, *_args, **_kwargs):
        return [SimpleNamespace(source_id=self.indexed)] if self.indexed else []

    def delete_source(self, *_args, **_kwargs):
        return None


@pytest.fixture
def cli_home(tmp_path, monkeypatch):
    home = tmp_path / "vg"
    monkeypatch.setenv("VECGREP_HOME", str(home))
    from vecgrep.backend import config

    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(cli_main, "VecgrepService", FakeService)
    return home, CliRunner()


def _seed_cli_confirm_protected_ack(corpus_dir: Path, doc_id: str, *, tier: str = "normal") -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / f"{doc_id}.md").write_text(
        f"---\nid: {doc_id}\ntier: {tier}\n---\n\nbody {doc_id}\n"
    )


def _pending_protected(home: Path, operation: str):
    corpus_dir = home / "write" / "notes"
    if operation == "write":
        proposal = proposal_module.propose(
            "notes",
            "protected body",
            corpus_dir,
            meta={"origin": "bot-suggested", "tier": "protected"},
        )
    elif operation == "edit":
        _seed_cli_confirm_protected_ack(corpus_dir, "notes-1", tier="protected")
        proposal = proposal_module.propose(
            "notes", "changed", corpus_dir, edit_id="notes-1"
        )
    elif operation == "delete":
        _seed_cli_confirm_protected_ack(corpus_dir, "notes-1", tier="protected")
        proposal = proposal_module.propose_delete("notes", "notes-1", corpus_dir)
    elif operation == "merge":
        _seed_cli_confirm_protected_ack(corpus_dir, "notes-1")
        _seed_cli_confirm_protected_ack(corpus_dir, "notes-2", tier="protected")
        proposal = proposal_module.propose_merge(
            "notes", ["notes-1", "notes-2"], "merged", corpus_dir
        )
    else:  # pragma: no cover - test helper contract
        raise AssertionError(operation)

    store = confirm_module.ProposalStore(home / "write" / "_pending")
    store.put(proposal)
    return proposal, store, corpus_dir


@pytest.mark.parametrize("operation", ["write", "edit", "delete", "merge"])
def test_cli_requires_explicit_ack_for_every_protected_operation(
    cli_home, operation
):
    home, runner = cli_home
    proposal, store, corpus_dir = _pending_protected(home, operation)
    before = {path.name: path.read_text() for path in corpus_dir.glob("*.md")}

    result = runner.invoke(cli_main.cli, ["confirm", proposal.proposal_id])

    assert result.exit_code != 0
    assert "protected" in result.output.lower()
    assert store.get(proposal.proposal_id) is not None
    after = {path.name: path.read_text() for path in corpus_dir.glob("*.md")}
    assert after == before


def test_cli_exact_ack_allows_protected_confirmation(cli_home):
    home, runner = cli_home
    proposal, store, _corpus_dir = _pending_protected(home, "write")

    result = runner.invoke(
        cli_main.cli,
        ["confirm", proposal.proposal_id, "--ack", proposal.doc_id],
    )

    assert result.exit_code == 0, result.output
    assert Path(proposal.target_path).exists()
    assert store.get(proposal.proposal_id) is None


def test_cli_wrong_ack_keeps_protected_proposal_pending(cli_home):
    home, runner = cli_home
    proposal, store, _corpus_dir = _pending_protected(home, "write")

    result = runner.invoke(
        cli_main.cli,
        ["confirm", proposal.proposal_id, "--ack", "wrong-id"],
    )

    assert result.exit_code != 0
    assert store.get(proposal.proposal_id) is not None
    assert not Path(proposal.target_path).exists()


@pytest.mark.parametrize("tier_line", ["tier:\tprotected", "tier:  protected"])
def test_cli_requires_ack_for_whitespace_variant_protected_tier(
    cli_home, tier_line
):
    home, runner = cli_home
    corpus_dir = home / "write" / "notes"
    _seed_cli_confirm_protected_ack(corpus_dir, "notes-1")
    target = corpus_dir / "notes-1.md"
    target.write_text(target.read_text().replace("tier: normal", tier_line))
    proposal = proposal_module.propose(
        "notes", "changed", corpus_dir, edit_id="notes-1"
    )
    store = confirm_module.ProposalStore(home / "write" / "_pending")
    store.put(proposal)

    result = runner.invoke(cli_main.cli, ["confirm", proposal.proposal_id])

    assert result.exit_code != 0
    assert "protected" in result.output.lower()
    assert store.get(proposal.proposal_id) is not None
    assert "changed" not in target.read_text()


def test_cli_ordinary_proposal_still_needs_no_ack(cli_home):
    home, runner = cli_home
    corpus_dir = home / "write" / "notes"
    proposal = proposal_module.propose("notes", "ordinary", corpus_dir)
    store = confirm_module.ProposalStore(home / "write" / "_pending")
    store.put(proposal)

    result = runner.invoke(cli_main.cli, ["confirm", proposal.proposal_id])

    assert result.exit_code == 0, result.output
    assert Path(proposal.target_path).exists()


@pytest.mark.parametrize("operation", ["write", "merge"])
def test_core_requires_ack_before_protected_writethrough(
    cli_home, monkeypatch, operation
):
    home, _runner = cli_home
    proposal, store, corpus_dir = _pending_protected(home, operation)
    monkeypatch.setenv("VECGREP_WRITETHROUGH_NOTES", "/bin/true")
    called = False

    def fake_writethrough(*_args, **_kwargs):
        nonlocal called
        called = True
        return confirm_module.ConfirmResult(
            ok=True, doc_id=proposal.doc_id, path="", message="routed"
        )

    monkeypatch.setattr(confirm_module, "_run_writethrough", fake_writethrough)

    with pytest.raises(confirm_module.ConfirmError, match="protected"):
        confirm_module.confirm(
            proposal.proposal_id,
            store,
            FakeService(),
            "notes",
            corpus_dir,
            confirmed_by="owner",
        )

    assert called is False
    assert store.get(proposal.proposal_id) is not None


def test_writethrough_merge_checks_absorbed_docs_in_real_mirror_directory(
    cli_home, monkeypatch
):
    home, _runner = cli_home
    mirror_dir = home / "mirror" / "notes"
    _seed_cli_confirm_protected_ack(mirror_dir, "notes-1")
    _seed_cli_confirm_protected_ack(mirror_dir, "notes-2", tier="protected")
    proposal = proposal_module.propose_merge(
        "notes", ["notes-1", "notes-2"], "merged", mirror_dir
    )
    store = confirm_module.ProposalStore(home / "write" / "_pending")
    store.put(proposal)
    monkeypatch.setenv("VECGREP_WRITETHROUGH_NOTES", "/bin/true")
    called = False

    def fake_writethrough(*_args, **_kwargs):
        nonlocal called
        called = True
        return confirm_module.ConfirmResult(
            ok=True, doc_id=proposal.doc_id, path="", message="routed"
        )

    monkeypatch.setattr(confirm_module, "_run_writethrough", fake_writethrough)

    with pytest.raises(confirm_module.ConfirmError, match="protected"):
        confirm_module.confirm(
            proposal.proposal_id,
            store,
            FakeService(),
            "notes",
            home / "write" / "notes",
            confirmed_by="owner",
        )

    assert called is False
    assert store.get(proposal.proposal_id) is not None
