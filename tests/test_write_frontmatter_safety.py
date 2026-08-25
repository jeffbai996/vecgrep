"""Security boundaries for write-tool frontmatter rendering and parsing."""
from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from vecgrep.backend.ingestion.adapters.base import AdapterError
from vecgrep.backend.ingestion.adapters.markdown import parse_frontmatter
from vecgrep.backend.write.confirm import _body_of
from vecgrep.backend.write.proposal import ProposalError, propose, render_doc


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
    home, server = direct_write_home
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
