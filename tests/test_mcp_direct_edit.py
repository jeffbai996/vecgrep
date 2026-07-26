"""MCP direct-edit path — unconfirmed edit, same single corpus as direct write.

The counterpart to `propose_edit`, for the corpus the operator opened up. Edits
are genuinely more dangerous than appends (an append is noise; an overwrite
destroys), so this path keeps two extra guards the write path doesn't need:

  - a .bak of the previous body is kept before every edit, so an unreviewed
    overwrite is always recoverable
  - the doc must already live in the direct-write corpus; a doc_id from any other
    corpus is rejected before anything is read or written

Deletes are deliberately NOT here — those stay propose-only everywhere.
"""
from __future__ import annotations

import glob
import json
import os

import pytest

pytest.importorskip("mcp")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("VECGREP_HOME", str(tmp_path / "vg"))
    monkeypatch.setenv("VECGREP_DIRECT_WRITE_CORPUS", "external")
    import vecgrep.backend.config as cfg
    monkeypatch.setattr(cfg, "_settings", None)
    return tmp_path


def _seed(content="original body text here"):
    """Land a doc via the direct-write path and return its id."""
    from vecgrep.mcp import server as S
    r = json.loads(S._run_direct_write(content, source_kind="fact"))
    assert "error" not in r, r
    return r["doc_id"]


# ── the kill switch ──────────────────────────────────────────────────────────

def test_disabled_when_unset(home, monkeypatch):
    doc_id = _seed()
    monkeypatch.delenv("VECGREP_DIRECT_WRITE_CORPUS", raising=False)
    from vecgrep.mcp import server as S
    r = json.loads(S._run_direct_edit(doc_id, content="replacement"))
    assert "error" in r and "not enabled" in r["error"].lower()


# ── full overwrite ───────────────────────────────────────────────────────────

def test_full_overwrite(home):
    from vecgrep.mcp import server as S
    doc_id = _seed("original body text here")
    r = json.loads(S._run_direct_edit(doc_id, content="totally new body"))
    assert "error" not in r, r
    assert r["committed"] is True
    body = open(r["path"]).read()
    assert "totally new body" in body
    assert "original body text here" not in body
    # frontmatter survives an edit
    assert f"id: {doc_id}" in body
    assert "corpus: external" in body


def test_backup_kept_before_overwrite(home):
    """An unreviewed overwrite must always be recoverable."""
    from vecgrep.mcp import server as S
    doc_id = _seed("precious original")
    r = json.loads(S._run_direct_edit(doc_id, content="clobbered"))
    assert "error" not in r, r
    baks = glob.glob(f"{os.environ['VECGREP_HOME']}/write/external/{doc_id}.md.bak*")
    assert len(baks) == 1, baks
    assert "precious original" in open(baks[0]).read()


# ── surgical patch ───────────────────────────────────────────────────────────

def test_edit_stamps_edited_at(home):
    """An unreviewed edit must leave an audit trace. render_doc's key order is a
    WHITELIST, so a new frontmatter key is silently dropped unless it's listed —
    this pins that edited_at actually survives to disk."""
    from vecgrep.mcp import server as S
    doc_id = _seed("before")
    r = json.loads(S._run_direct_edit(doc_id, content="after"))
    assert "error" not in r, r
    body = open(r["path"]).read()
    assert "edited_at:" in body


def test_patch_mode(home):
    from vecgrep.mcp import server as S
    doc_id = _seed("alpha beta gamma")
    r = json.loads(S._run_direct_edit(doc_id, old_str="beta", new_str="DELTA"))
    assert "error" not in r, r
    body = open(r["path"]).read()
    assert "alpha DELTA gamma" in body


def test_patch_requires_unique_match(home):
    """A non-unique old_str is a hard error, never a silent wrong-edit."""
    from vecgrep.mcp import server as S
    doc_id = _seed("repeat repeat repeat")
    r = json.loads(S._run_direct_edit(doc_id, old_str="repeat", new_str="x"))
    assert "error" in r and "unique" in r["error"].lower()
    # unchanged
    assert "repeat repeat repeat" in _body_on_disk(doc_id)


def test_patch_missing_old_str(home):
    from vecgrep.mcp import server as S
    doc_id = _seed("some body")
    r = json.loads(S._run_direct_edit(doc_id, old_str="absent", new_str="x"))
    assert "error" in r and "not found" in r["error"].lower()


def test_content_and_patch_are_mutually_exclusive(home):
    from vecgrep.mcp import server as S
    doc_id = _seed()
    r = json.loads(S._run_direct_edit(doc_id, content="whole",
                                      old_str="a", new_str="b"))
    assert "error" in r


def test_needs_content_or_patch(home):
    from vecgrep.mcp import server as S
    doc_id = _seed()
    r = json.loads(S._run_direct_edit(doc_id))
    assert "error" in r


# ── the corpus wall ──────────────────────────────────────────────────────────

def test_no_corpus_parameter(home):
    """Same reasoning as direct write: an absent parameter can't be argued around."""
    import inspect
    from vecgrep.mcp import server as S
    assert "corpus" not in inspect.signature(S._run_direct_edit).parameters


def test_unknown_doc_rejected(home):
    from vecgrep.mcp import server as S
    r = json.loads(S._run_direct_edit("external-999999", content="x"))
    assert "error" in r and "not found" in r["error"].lower()


def test_foreign_corpus_doc_id_rejected(home):
    """A doc_id whose prefix belongs to another corpus must not resolve, even if
    such a doc exists there — this path only ever touches its own corpus."""
    from vecgrep.mcp import server as S
    r = json.loads(S._run_direct_edit("squadstore-123", content="x"))
    assert "error" in r


def test_refuses_corpus_with_writethrough(home, monkeypatch):
    from vecgrep.mcp import server as S
    doc_id = _seed()
    monkeypatch.setenv("VECGREP_WRITETHROUGH_EXTERNAL", "/bin/true")
    r = json.loads(S._run_direct_edit(doc_id, content="upstream mutation"))
    assert "error" in r and "write-through" in r["error"].lower()


# ── no deletes here ──────────────────────────────────────────────────────────

def test_no_delete_capability(home):
    import inspect
    from vecgrep.mcp import server as S
    params = inspect.signature(S._run_direct_edit).parameters
    for forbidden in ("delete", "delete_id", "remove"):
        assert forbidden not in params


# ── protected tier still protected ───────────────────────────────────────────

def test_protected_doc_refused(home):
    """`tier: protected` means human-only, even in the open corpus."""
    from vecgrep.mcp import server as S
    doc_id = _seed("guard this")
    p = f"{os.environ['VECGREP_HOME']}/write/external/{doc_id}.md"
    s = open(p).read().replace("tier: normal", "tier: protected")
    open(p, "w").write(s)
    r = json.loads(S._run_direct_edit(doc_id, content="should not land"))
    assert "error" in r and "protected" in r["error"].lower()
    assert "guard this" in open(p).read()


def _body_on_disk(doc_id: str) -> str:
    return open(f"{os.environ['VECGREP_HOME']}/write/external/{doc_id}.md").read()
