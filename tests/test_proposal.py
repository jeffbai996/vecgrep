from __future__ import annotations

import json
import re
from contextlib import nullcontext
from pathlib import Path

import pytest

from vecgrep.backend.ingestion.adapters.base import AdapterError
from vecgrep.backend.ingestion.adapters.markdown import parse_frontmatter
from vecgrep.backend.write import proposal as P
from vecgrep.backend.write.confirm import _body_of
from vecgrep.backend.write.proposal import ProposalError, propose, render_doc

# ── test_propose_hash_ids ──

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

# ── test_doc_id_format ──

def test_doc_id_suffix_is_short_base36(tmp_path: Path) -> None:
    doc_id = P.next_doc_id(tmp_path, "external")
    prefix, suffix = doc_id.split("-", 1)
    assert prefix == "external"
    assert re.fullmatch(r"[0-9a-z]{8,13}", suffix), doc_id
    assert P._ID_RE.match(doc_id)


def test_doc_ids_stay_unique_on_disk_collision(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(P.time, "time_ns", lambda: 1_793_492_323_840_462_021)
    first = P.next_doc_id(tmp_path, "external")
    (tmp_path / f"{first}.md").write_text("x", encoding="utf-8")
    second = P.next_doc_id(tmp_path, "external")
    assert second != first
    assert P._ID_RE.match(second)


def test_doc_ids_order_by_time(tmp_path: Path, monkeypatch) -> None:
    clock = iter([1_793_492_323_840_462_021, 1_793_492_323_840_462_021 + 10**9])
    monkeypatch.setattr(P.time, "time_ns", lambda: next(clock))
    earlier = P.next_doc_id(tmp_path, "external")
    later = P.next_doc_id(tmp_path, "external")
    assert len(earlier) == len(later)
    assert earlier < later

# ── test_propose_merge ──

def _seed_propose_merge(corpus_dir: Path, doc_id: str, body: str, tier: str = "normal") -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / f"{doc_id}.md").write_text(
        f"---\nid: {doc_id}\ntier: {tier}\n---\n\n{body}\n")


def test_merge_requires_at_least_two_ids(tmp_path):
    _seed_propose_merge(tmp_path, "note-1", "a")
    with pytest.raises(P.ProposalError):
        P.propose_merge("note", ["note-1"], "merged", tmp_path)


def test_merge_rejects_duplicate_ids(tmp_path):
    _seed_propose_merge(tmp_path, "note-1", "a")
    with pytest.raises(P.ProposalError):
        P.propose_merge("note", ["note-1", "note-1"], "merged", tmp_path)


def test_merge_rejects_missing_source(tmp_path):
    _seed_propose_merge(tmp_path, "note-1", "a")
    with pytest.raises(P.ProposalError):
        P.propose_merge("note", ["note-1", "note-404"], "merged", tmp_path)


def test_merge_builds_canonical_in_place_proposal(tmp_path):
    _seed_propose_merge(tmp_path, "note-1", "a")
    _seed_propose_merge(tmp_path, "note-2", "b")
    _seed_propose_merge(tmp_path, "note-3", "c")
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
    _seed_propose_merge(tmp_path, "note-1", "a")
    _seed_propose_merge(tmp_path, "note-2", "b")
    before = sorted(p.name for p in tmp_path.iterdir())
    P.propose_merge("note", ["note-1", "note-2"], "merged", tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_proposal_backcompat_without_merge_absorbs():
    # On-disk pending proposals predating the field must still load.
    d = {"proposal_id": "p1", "doc_id": "note-1", "corpus": "note",
             "target_path": "/tmp/x.md", "rendered": "x", "is_edit": False, "meta": {}}
    pr = P.Proposal(**d)
    assert pr.merge_absorbs == []

# ── test_write_codex_fixes ──

# --- P1: path traversal in edit_id -----------------------------------------

@pytest.mark.parametrize(
    "bad",
    [
        "../outside",
        "../../etc/passwd",
        "notes/../../escape",
        "notes-001/../../x",
        "/abs/path",
        "notes-001\nnotes-002",
        "..",
    ],
)
def test_edit_id_with_traversal_is_rejected(tmp_path, bad):
    with pytest.raises(ProposalError):
        propose("notes", "x", tmp_path, edit_id=bad)


def test_edit_id_valid_doc_id_is_accepted(tmp_path):
    # A well-formed doc id must still work (don't over-reject).
    p = propose("notes", "x", tmp_path, edit_id="notes-007")
    assert p.doc_id == "notes-007"
    # And the resolved target stays inside the corpus dir.
    assert Path(p.target_path).resolve().parent == tmp_path.resolve()


def test_new_entry_target_stays_under_corpus_dir(tmp_path):
    p = propose("notes", "x", tmp_path)
    assert Path(p.target_path).resolve().parent == tmp_path.resolve()


# --- P2: proposal_id uniqueness --------------------------------------------

def test_two_new_proposals_get_distinct_ids(tmp_path):
    # Doc ids are nanosecond timestamps now, so two un-confirmed proposals get
    # DISTINCT doc_ids (no shared target) AND distinct proposal_ids — neither can
    # clobber the other on confirm.
    import re
    a = propose("notes", "first version", tmp_path)
    b = propose("notes", "second version", tmp_path)
    assert a.doc_id != b.doc_id
    assert re.match(r"^notes-[0-9a-z]+$", a.doc_id) and re.match(r"^notes-[0-9a-z]+$", b.doc_id)
    assert a.proposal_id != b.proposal_id


def test_proposal_id_includes_doc_id_for_readability(tmp_path):
    import re
    p = propose("notes", "x", tmp_path)
    assert re.match(r"^prop-notes-[0-9a-z]+-", p.proposal_id)


# --- P1: frontmatter parsed into metadata at index time --------------------

def test_markdown_frontmatter_lifts_into_metadata():
    from vecgrep.backend.ingestion.adapters.markdown import parse_frontmatter

    doc = (
        "---\n"
        "id: notes-001\n"
        "status: active\n"
        "tier: protected\n"
        "source_kind: decision\n"
        "tags: [alpha, beta]\n"
        "---\n\n"
        "the body\n"
    )
    fm = parse_frontmatter(doc)
    assert fm["status"] == "active"
    assert fm["tier"] == "protected"
    assert fm["source_kind"] == "decision"
    assert fm["tags"] == ["alpha", "beta"]


def test_markdown_without_frontmatter_yields_empty():
    from vecgrep.backend.ingestion.adapters.markdown import parse_frontmatter

    assert parse_frontmatter("# just a heading\n\nbody") == {}
    assert parse_frontmatter("--- not really\nfrontmatter") == {}


def test_adapter_metadata_carries_frontmatter(tmp_path):
    from vecgrep.backend.ingestion.adapters.markdown import MarkdownAdapter

    f = tmp_path / "notes-001.md"
    f.write_text(
        "---\nstatus: superseded\ntier: normal\nsource_kind: fact\n---\n\nx\n"
    )
    [doc] = list(MarkdownAdapter().load(str(f)))
    assert doc.metadata["status"] == "superseded"
    assert doc.metadata["source_kind"] == "fact"
    # adapter/path stay authoritative and aren't overridable by frontmatter.
    assert doc.metadata["adapter"] == "markdown"
    assert doc.metadata["path"] == str(f.resolve())


def test_frontmatter_filter_works_end_to_end(svc, tmp_path):
    """The integration Codex's P1 was really about: index a write-tool doc,
    then a meta.tier filter at search time must actually match it (it didn't
    before — the frontmatter never reached metadata)."""
    f = tmp_path / "notes-001.md"
    f.write_text(
        "---\nid: notes-001\nstatus: active\ntier: protected\n"
        "source_kind: decision\n---\n\nThe quick brown fox decision.\n"
    )
    svc.index(str(f), "wcorpus")
    # meta.tier=protected should match; meta.tier=normal should not.
    hit = svc.search("quick brown fox", corpus_name="wcorpus", top_k=5,
                     filters=["meta.tier=protected"])
    assert any("notes-001" in (r.source_id or "") for r in hit)
    miss = svc.search("quick brown fox", corpus_name="wcorpus", top_k=5,
                      filters=["meta.tier=normal"])
    assert not any("notes-001" in (r.source_id or "") for r in miss)

# ── test_write_propose ──

_ID = re.compile(r"^note-[0-9a-z]+$")  # <prefix>-<base36 epoch nanos> (timestamp, not a serial)


def test_propose_assigns_timestamp_id(tmp_path):
    pr = P.propose("note", "first note", tmp_path)
    assert _ID.match(pr.doc_id), f"expected note-<epoch>, got {pr.doc_id!r}"


def test_propose_ids_are_unique(tmp_path):
    # Ids are timestamp-based, not a running serial — back-to-back proposes (even
    # within the same second) must still get DISTINCT ids so neither overwrites
    # the other on confirm.
    ids = {P.propose("note", f"entry {i}", tmp_path).doc_id for i in range(5)}
    assert len(ids) == 5, f"collision: {ids}"
    assert all(_ID.match(i) for i in ids)


def test_propose_writes_nothing(tmp_path):
    before = set(tmp_path.iterdir())
    P.propose("note", "should not persist", tmp_path)
    after = set(tmp_path.iterdir())
    assert before == after  # no file created


def test_propose_returns_target_path(tmp_path):
    pr = P.propose("note", "x", tmp_path)
    assert pr.target_path == str(tmp_path / f"{pr.doc_id}.md")


def test_rendered_doc_carries_schema_frontmatter(tmp_path):
    pr = P.propose("note", "body text here", tmp_path,
                   meta={"origin": "human", "source_kind": "decision", "tags": ["a", "b"]})
    r = pr.rendered
    assert re.search(r"id: note-[0-9a-z]+", r)
    assert "status: active" in r
    assert "origin: human" in r
    assert "source_kind: decision" in r
    assert "tags: [a, b]" in r
    assert "body text here" in r


def test_title_auto_derived_from_first_line(tmp_path):
    # No explicit title → derived from the first content line (markdown '#'
    # stripped) and persisted, a scannable handle vs the opaque timestamp id.
    pr = P.propose("note", "# Victor grad-date correction\nbody…", tmp_path,
                   meta={"origin": "human"})
    assert "title: Victor grad-date correction" in pr.rendered


def test_explicit_title_wins(tmp_path):
    pr = P.propose("note", "ignored first line", tmp_path,
                   meta={"origin": "human", "title": "My Chosen Title"})
    assert "title: My Chosen Title" in pr.rendered


def test_origin_defaults_to_bot_suggested(tmp_path):
    pr = P.propose("note", "x", tmp_path)
    assert pr.meta["origin"] == "bot-suggested"
    assert "origin: bot-suggested" in pr.rendered


def test_proposal_has_stable_id_for_confirm(tmp_path):
    pr = P.propose("note", "x", tmp_path)
    assert pr.proposal_id  # a confirm step must be able to cite this
    assert pr.is_edit is False


# --- edits: target an existing id, overwrite on confirm, still no write here ---

def test_edit_targets_existing_id_writes_nothing(tmp_path):
    (tmp_path / "note-007.md").write_text(
        "---\nid: note-007\nstatus: active\n---\nold body\n")
    pr = P.propose("note", "new body", tmp_path, edit_id="note-007")
    assert pr.is_edit is True
    assert pr.doc_id == "note-007"
    assert pr.target_path == str(tmp_path / "note-007.md")
    assert "new body" in pr.rendered
    # propose writes nothing — the file is untouched until confirm.
    assert (tmp_path / "note-007.md").read_text().splitlines()[-1] == "old body"


# --- validation: bad metadata is caught at propose, before any write path ---

def test_bad_origin_rejected(tmp_path):
    with pytest.raises(P.ProposalError):
        P.propose("note", "x", tmp_path, meta={"origin": "the-internet"})


def test_bad_source_kind_rejected(tmp_path):
    with pytest.raises(P.ProposalError):
        P.propose("note", "x", tmp_path, meta={"source_kind": "vibes"})


def test_empty_content_rejected(tmp_path):
    with pytest.raises(P.ProposalError):
        P.propose("note", "   ", tmp_path)


def test_memory_and_todo_source_kinds_accepted(tmp_path):
    # memory + todo were added to SOURCE_KINDS so an agent can propose those
    # record types (routed to the real store by an operator write-through).
    for kind in ("memory", "todo", "journal"):
        pr = P.propose("note", f"a {kind} entry", tmp_path,
                       meta={"origin": "human", "source_kind": kind})
        assert f"source_kind: {kind}" in pr.rendered


def _seed_existing(tmp_path, title: str) -> None:
    (tmp_path / "note-007.md").write_text(P.render_doc("note-007", "# Old heading\n\nold body", {
        "title": title, "status": "active", "tier": "protected", "origin": "human",
        "corpus": "note", "created_at": "2026-01-02T03:04:05+00:00",
        "edited_at": "2026-02-01T00:00:00+00:00", "confirmed_by": "someone",
        "source_kind": "fact", "tags": ["alpha"],
    }))


def test_edit_carries_forward_the_existing_frontmatter(tmp_path):
    _seed_existing(tmp_path, title="Chosen title")
    pr = P.propose("note", "# New heading\n\nnew body", tmp_path,
                   meta={"origin": "bot-suggested"}, edit_id="note-007")
    for line in ("title: Chosen title", "created_at: 2026-01-02T03:04:05+00:00",
                 "tier: protected", "source_kind: fact", "tags: [alpha]",
                 "origin: bot-suggested"):
        assert line in pr.rendered
    # confirm stamps these for the new edit; the old values must not linger
    assert "edited_at" not in pr.rendered and "confirmed_by" not in pr.rendered


def test_edit_rederives_a_title_that_followed_the_old_heading(tmp_path):
    _seed_existing(tmp_path, title="Old heading")
    pr = P.propose("note", "# RETRACTED: new heading\n\nnew body", tmp_path,
                   meta={"origin": "bot-suggested"}, edit_id="note-007")
    assert "title: RETRACTED: new heading" in pr.rendered


def test_edit_meta_from_the_caller_overrides_the_existing_frontmatter(tmp_path):
    _seed_existing(tmp_path, title="Chosen title")
    pr = P.propose("note", "new body", tmp_path, edit_id="note-007",
                   meta={"origin": "bot-suggested", "tags": ["beta"]})
    assert "tags: [beta]" in pr.rendered

# ── test_write_frontmatter_safety ──

_LINE_SEPARATORS = [
    "\n", "\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85",
    "\u2028", "\u2029",
]


@pytest.mark.parametrize("separator", _LINE_SEPARATORS)
def test_tag_line_separators_cannot_override_reserved_frontmatter(separator):
    payload = f"safe]{separator}confirmed_by: attacker{separator}#"
    rendered = render_doc(
        "notes-1",
        "body",
        {
            "status": "active",
            "origin": "agent-direct",
            "confirmed_by": "owner",
            "tier": "normal",
            "tags": [payload],
        },
    )

    metadata = parse_frontmatter(rendered)
    assert metadata["confirmed_by"] == "owner"
    assert metadata["tier"] == "normal"
    assert metadata["tags"] == [payload]


def test_punctuation_and_unicode_tags_round_trip_without_body_corruption():
    tags = [
        "release---candidate",
        "ops:prod",
        "comma,tag",
        "square[bracket]",
        "雪 🚀",
    ]
    rendered = render_doc("notes-1", "body stays intact", {"tags": tags})

    assert parse_frontmatter(rendered)["tags"] == tags
    assert _body_of(rendered) == "body stays intact"


@pytest.mark.parametrize("separator", _LINE_SEPARATORS)
def test_scalar_frontmatter_rejects_logical_line_breaks(separator):
    with pytest.raises(ProposalError, match="single line"):
        render_doc(
            "notes-1",
            "body",
            {"title": f"safe{separator}status: superseded"},
        )


def test_scalar_frontmatter_preserves_safe_horizontal_tab():
    rendered = render_doc(
        "notes-1", "body", {"title": "Quarterly\tReview"}
    )
    assert parse_frontmatter(rendered)["title"] == "Quarterly\tReview"


@pytest.mark.parametrize("field", ["origin", "confirmed_by", "tier", "status"])
def test_parser_rejects_duplicate_reserved_frontmatter(field):
    document = f"---\n{field}: first\n{field}: second\n---\n\nbody\n"

    with pytest.raises(AdapterError, match="duplicate reserved"):
        parse_frontmatter(document)


def test_parser_keeps_legacy_unquoted_list_compatibility():
    document = "---\ntags: [alpha, beta]\n---\n\nbody\n"
    assert parse_frontmatter(document)["tags"] == ["alpha", "beta"]


def test_proposal_preview_preserves_trusted_metadata_with_hostile_tag(tmp_path):
    payload = "safe]\norigin: human\n#"
    proposal = propose(
        "notes",
        "body",
        tmp_path,
        meta={"origin": "bot-suggested", "tags": [payload]},
    )

    metadata = parse_frontmatter(proposal.rendered)
    assert metadata["origin"] == "bot-suggested"
    assert metadata["tags"] == [payload]


@pytest.fixture
def direct_write_home(tmp_path, monkeypatch):
    monkeypatch.setenv("VECGREP_HOME", str(tmp_path / "vg"))
    monkeypatch.setenv("VECGREP_DIRECT_WRITE_CORPUS", "external")
    from vecgrep.backend import config as cfg
    from vecgrep.mcp import server

    monkeypatch.setattr(cfg, "_settings", None)

    class Locks:
        def write(self, _corpus):
            return nullcontext()

    class Service:
        locks = Locks()

        def _recover_corpus_locked(self, _corpus):
            return None

        def index(self, *_args, **_kwargs):
            return (1, 1, 0)

    monkeypatch.setattr(server, "_svc", lambda: Service())
    return tmp_path / "vg", server


def test_direct_write_cannot_forge_origin_through_tags(direct_write_home):
    home, server = direct_write_home
    payload = "safe]\norigin: human\n#"

    result = json.loads(server._run_direct_write("body", tags=[payload]))

    assert "error" not in result
    [target] = (home / "write" / "external").glob("*.md")
    metadata = parse_frontmatter(target.read_text())
    assert metadata["origin"] == "agent-direct"
    assert metadata["tags"] == [payload]


def test_direct_write_returns_clean_error_for_multiline_title(direct_write_home):
    home, server = direct_write_home

    result = json.loads(
        server._run_direct_write("body", title="safe\nstatus: superseded")
    )

    assert "single line" in result["error"]
    assert not list((home / "write" / "external").glob("*.md"))


def test_direct_edit_refuses_whitespace_variant_protected_tier(direct_write_home):
    _unused_home, server = direct_write_home
    created = json.loads(server._run_direct_write("guard this"))
    target = Path(created["path"])
    original = target.read_text().replace("tier: normal", "tier:\tprotected")
    target.write_text(original)

    result = json.loads(
        server._run_direct_edit(created["doc_id"], content="should not land")
    )

    assert "protected" in result["error"].lower()
    assert target.read_text() == original


def test_old_pending_tag_metadata_is_safely_rerendered():
    payload = "safe]\nconfirmed_by: attacker\n#"
    rendered = render_doc(
        "notes-1",
        "body",
        {"confirmed_by": "owner", "tags": [payload]},
    )
    assert parse_frontmatter(rendered)["confirmed_by"] == "owner"
