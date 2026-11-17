from __future__ import annotations

import pytest

pytest.importorskip("mcp")

import glob
import json
import os
from pathlib import Path

import pytest

# ── test_mcp_direct_write ──

@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("VECGREP_HOME", str(tmp_path / "vg"))
    monkeypatch.setenv("VECGREP_DIRECT_WRITE_CORPUS", "external")
    import vecgrep.backend.config as cfg
    monkeypatch.setattr(cfg, "_settings", None)
    return tmp_path


# ── the kill switch ──────────────────────────────────────────────────────────

def test_disabled_when_unset(home, monkeypatch):
    """No VECGREP_DIRECT_WRITE_CORPUS -> the tool refuses. A default-on
    unconfirmed write in a public repo would be a footgun for every other user."""
    monkeypatch.delenv("VECGREP_DIRECT_WRITE_CORPUS", raising=False)
    from vecgrep.mcp import server as S
    r = json.loads(S._run_direct_write("a fact"))
    assert "error" in r
    assert "not enabled" in r["error"].lower()
    assert not glob.glob(f"{os.environ['VECGREP_HOME']}/write/*/*.md")


# ── the happy path ───────────────────────────────────────────────────────────

def test_direct_write_commits_immediately(home):
    """Unlike propose, this lands a real doc with NO pending proposal."""
    from vecgrep.mcp import server as S
    r = json.loads(S._run_direct_write("a directly written fact",
                                       source_kind="fact"))
    assert "error" not in r, r
    assert r["corpus"] == "external"
    assert r["committed"] is True

    h = os.environ["VECGREP_HOME"]
    docs = glob.glob(f"{h}/write/external/external-*.md")
    assert len(docs) == 1
    body = Path(docs[0]).read_text()
    assert "a directly written fact" in body
    # Stamped honestly: neither 'human' (nobody approved it) nor 'bot-suggested'
    # (nothing was suggested — it was committed).
    assert "origin: agent-direct" in body
    # and crucially: nothing is waiting for a human
    assert not glob.glob(f"{h}/write/_pending/*.json")


def test_metadata_is_recorded(home):
    from vecgrep.mcp import server as S
    r = json.loads(S._run_direct_write("body text", source_kind="memory",
                                       tags=["alpha", "beta"], title="A Title"))
    assert "error" not in r, r
    body = Path(glob.glob(f"{os.environ['VECGREP_HOME']}/write/external/*.md")[0]).read_text()
    assert "source_kind: memory" in body
    assert "alpha" in body and "beta" in body
    assert "A Title" in body


# ── the corpus wall ──────────────────────────────────────────────────────────

def test_corpus_is_not_a_caller_parameter(home):
    """The signature must not accept a corpus. A blocked parameter can be argued
    around by a clever caller; an ABSENT one cannot be expressed at all."""
    import inspect

    from vecgrep.mcp import server as S
    params = inspect.signature(S._run_direct_write).parameters
    assert "corpus" not in params


def test_refuses_corpus_with_writethrough(home, monkeypatch):
    """A write-through routes a confirmed op into an upstream store. Combining it
    with an UNCONFIRMED write would let an agent mutate that upstream with no
    human in the loop — refuse to even start."""
    monkeypatch.setenv("VECGREP_DIRECT_WRITE_CORPUS", "mirrored")
    monkeypatch.setenv("VECGREP_WRITETHROUGH_MIRRORED", "/bin/true")
    from vecgrep.mcp import server as S
    r = json.loads(S._run_direct_write("should never land"))
    assert "error" in r
    assert "write-through" in r["error"].lower()
    assert not glob.glob(f"{os.environ['VECGREP_HOME']}/write/mirrored/*.md")


# ── append-only ──────────────────────────────────────────────────────────────

def test_no_edit_or_delete_on_this_path(home):
    """Append-only by construction: a bad direct write is NOISE, never
    destruction. Edits/deletes stay propose-only for every corpus."""
    import inspect

    from vecgrep.mcp import server as S
    params = inspect.signature(S._run_direct_write).parameters
    for forbidden in ("edit_id", "doc_id", "delete_id", "old_str", "new_str"):
        assert forbidden not in params, forbidden


# ── caps ─────────────────────────────────────────────────────────────────────

def test_size_cap(home, monkeypatch):
    monkeypatch.setenv("VECGREP_DIRECT_WRITE_MAX_BYTES", "100")
    from vecgrep.mcp import server as S
    r = json.loads(S._run_direct_write("x" * 101))
    assert "error" in r and "cap" in r["error"].lower()
    assert not glob.glob(f"{os.environ['VECGREP_HOME']}/write/external/*.md")


def test_rate_cap(home, monkeypatch):
    """Runaway-loop protection. A human note-taker never approaches the cap; an
    injected 'write this 10,000 times' hits it after N."""
    monkeypatch.setenv("VECGREP_DIRECT_WRITE_MAX_PER_HOUR", "3")
    from vecgrep.mcp import server as S
    for i in range(3):
        r = json.loads(S._run_direct_write(f"entry {i}"))
        assert "error" not in r, r
    r = json.loads(S._run_direct_write("one too many"))
    assert "error" in r and "rate" in r["error"].lower()
    assert len(glob.glob(f"{os.environ['VECGREP_HOME']}/write/external/*.md")) == 3


def test_empty_content_rejected(home):
    from vecgrep.mcp import server as S
    for bad in ("", "   "):
        r = json.loads(S._run_direct_write(bad))
        assert "error" in r


# ── propose is untouched ─────────────────────────────────────────────────────

def test_propose_still_default_deny_for_other_corpora(home, monkeypatch):
    """Enabling direct write must NOT widen the propose allowlist. Every other
    corpus stays propose-only, human-confirmed."""
    monkeypatch.delenv("VECGREP_PROPOSE_ALLOWED_CORPORA", raising=False)
    from vecgrep.mcp import server as S
    r = json.loads(S._run_propose("shared-corpus", "sneak this in"))
    assert "error" in r
    assert "not agent-writable" in r["error"]

# ── test_mcp_direct_edit ──

@pytest.fixture
def home_mcp_direct_edit(tmp_path, monkeypatch):
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

def test_mcp_direct_edit_disabled_when_unset(home_mcp_direct_edit, monkeypatch):
    doc_id = _seed()
    monkeypatch.delenv("VECGREP_DIRECT_WRITE_CORPUS", raising=False)
    from vecgrep.mcp import server as S
    r = json.loads(S._run_direct_edit(doc_id, content="replacement"))
    assert "error" in r and "not enabled" in r["error"].lower()


# ── full overwrite ───────────────────────────────────────────────────────────

def test_full_overwrite(home_mcp_direct_edit):
    from vecgrep.mcp import server as S
    doc_id = _seed("original body text here")
    r = json.loads(S._run_direct_edit(doc_id, content="totally new body"))
    assert "error" not in r, r
    assert r["committed"] is True
    body = Path(r["path"]).read_text()
    assert "totally new body" in body
    assert "original body text here" not in body
    # frontmatter survives an edit
    assert f"id: {doc_id}" in body
    assert "corpus: external" in body


def test_backup_kept_before_overwrite(home_mcp_direct_edit):
    """An unreviewed overwrite must always be recoverable."""
    from vecgrep.mcp import server as S
    doc_id = _seed("precious original")
    r = json.loads(S._run_direct_edit(doc_id, content="clobbered"))
    assert "error" not in r, r
    baks = glob.glob(f"{os.environ['VECGREP_HOME']}/write/external/{doc_id}.md.bak*")
    assert len(baks) == 1, baks
    assert "precious original" in Path(baks[0]).read_text()


# ── surgical patch ───────────────────────────────────────────────────────────

def test_edit_stamps_edited_at(home_mcp_direct_edit):
    """An unreviewed edit must leave an audit trace. render_doc's key order is a
    WHITELIST, so a new frontmatter key is silently dropped unless it's listed —
    this pins that edited_at actually survives to disk."""
    from vecgrep.mcp import server as S
    doc_id = _seed("before")
    r = json.loads(S._run_direct_edit(doc_id, content="after"))
    assert "error" not in r, r
    body = Path(r["path"]).read_text()
    assert "edited_at:" in body


def test_patch_mode(home_mcp_direct_edit):
    from vecgrep.mcp import server as S
    doc_id = _seed("alpha beta gamma")
    r = json.loads(S._run_direct_edit(doc_id, old_str="beta", new_str="DELTA"))
    assert "error" not in r, r
    body = Path(r["path"]).read_text()
    assert "alpha DELTA gamma" in body


def test_patch_requires_unique_match(home_mcp_direct_edit):
    """A non-unique old_str is a hard error, never a silent wrong-edit."""
    from vecgrep.mcp import server as S
    doc_id = _seed("repeat repeat repeat")
    r = json.loads(S._run_direct_edit(doc_id, old_str="repeat", new_str="x"))
    assert "error" in r and "unique" in r["error"].lower()
    # unchanged
    assert "repeat repeat repeat" in _body_on_disk(doc_id)


def test_patch_missing_old_str(home_mcp_direct_edit):
    from vecgrep.mcp import server as S
    doc_id = _seed("some body")
    r = json.loads(S._run_direct_edit(doc_id, old_str="absent", new_str="x"))
    assert "error" in r and "not found" in r["error"].lower()


def test_content_and_patch_are_mutually_exclusive(home_mcp_direct_edit):
    from vecgrep.mcp import server as S
    doc_id = _seed()
    r = json.loads(S._run_direct_edit(doc_id, content="whole",
                                      old_str="a", new_str="b"))
    assert "error" in r


def test_needs_content_or_patch(home_mcp_direct_edit):
    from vecgrep.mcp import server as S
    doc_id = _seed()
    r = json.loads(S._run_direct_edit(doc_id))
    assert "error" in r


# ── the corpus wall ──────────────────────────────────────────────────────────

def test_no_corpus_parameter(home_mcp_direct_edit):
    """Same reasoning as direct write: an absent parameter can't be argued around."""
    import inspect

    from vecgrep.mcp import server as S
    assert "corpus" not in inspect.signature(S._run_direct_edit).parameters


def test_unknown_doc_rejected(home_mcp_direct_edit):
    from vecgrep.mcp import server as S
    r = json.loads(S._run_direct_edit("external-999999", content="x"))
    assert "error" in r and "not found" in r["error"].lower()


def test_foreign_corpus_doc_id_rejected(home_mcp_direct_edit):
    """A doc_id whose prefix belongs to another corpus must not resolve, even if
    such a doc exists there — this path only ever touches its own corpus."""
    from vecgrep.mcp import server as S
    r = json.loads(S._run_direct_edit("squadstore-123", content="x"))
    assert "error" in r


def test_traversal_doc_id_cannot_overwrite_outside_corpus(home_mcp_direct_edit):
    from vecgrep.mcp import server as S

    outside = Path(os.environ["VECGREP_HOME"]) / "outside.md"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("must survive")
    r = json.loads(S._run_direct_edit("../../outside", content="clobbered"))
    assert "error" in r
    assert outside.read_text() == "must survive"


def test_symlink_doc_cannot_escape_direct_edit_corpus(home_mcp_direct_edit):
    from vecgrep.mcp import server as S

    corpus_dir = Path(os.environ["VECGREP_HOME"]) / "write" / "external"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    outside = Path(os.environ["VECGREP_HOME"]) / "outside.md"
    outside.write_text("must survive")
    (corpus_dir / "external-123.md").symlink_to(outside)
    r = json.loads(S._run_direct_edit("external-123", content="clobbered"))
    assert "error" in r and "symbolic" in r["error"].lower()
    assert outside.read_text() == "must survive"


def test_mcp_direct_edit_refuses_corpus_with_writethrough(home_mcp_direct_edit, monkeypatch):
    from vecgrep.mcp import server as S
    doc_id = _seed()
    monkeypatch.setenv("VECGREP_WRITETHROUGH_EXTERNAL", "/bin/true")
    r = json.loads(S._run_direct_edit(doc_id, content="upstream mutation"))
    assert "error" in r and "write-through" in r["error"].lower()


# ── no deletes here ──────────────────────────────────────────────────────────

def test_no_delete_capability(home_mcp_direct_edit):
    import inspect

    from vecgrep.mcp import server as S
    params = inspect.signature(S._run_direct_edit).parameters
    for forbidden in ("delete", "delete_id", "remove"):
        assert forbidden not in params


# ── protected tier still protected ───────────────────────────────────────────

def test_protected_doc_refused(home_mcp_direct_edit):
    """`tier: protected` means human-only, even in the open corpus."""
    from vecgrep.mcp import server as S
    doc_id = _seed("guard this")
    p = f"{os.environ['VECGREP_HOME']}/write/external/{doc_id}.md"
    s = Path(p).read_text().replace("tier: normal", "tier: protected")
    Path(p).write_text(s)
    r = json.loads(S._run_direct_edit(doc_id, content="should not land"))
    assert "error" in r and "protected" in r["error"].lower()
    assert "guard this" in Path(p).read_text()


def _body_on_disk(doc_id: str) -> str:
    return Path(f"{os.environ['VECGREP_HOME']}/write/external/{doc_id}.md").read_text()


def test_edit_receipt_keeps_unchanged_context_at_the_mutation_version(home_mcp_direct_edit):
    from vecgrep.mcp import server as S
    original='\n'.join(f'Line {i}' for i in range(50))
    doc_id=_seed(original)
    result=json.loads(S._run_direct_edit(doc_id,old_str='Line 25',new_str='Revised line'))
    diff=result['unified_diff']
    assert ' Line 24\n' in diff and ' Line 26\n' in diff
    assert '-Line 25\n' in diff and '+Revised line\n' in diff
    assert ' Line 0\n' not in diff
    assert result['diff_truncated'] is False
    S._run_direct_edit(doc_id,old_str='Line 24',new_str='Later edit')
    assert 'Later edit' not in diff


def test_edit_diff_positions_match_both_real_files_including_frontmatter(home_mcp_direct_edit):
    import re

    from vecgrep.mcp import server as S
    doc_id=_seed('Above\nOld detail\nBelow')
    before=_body_on_disk(doc_id)
    result=json.loads(S._run_direct_edit(doc_id,old_str='Old detail',new_str='New detail'))
    after=_body_on_disk(doc_id)
    old_line=next(i for i,line in enumerate(before.splitlines(),1) if line=='Old detail')
    new_line=next(i for i,line in enumerate(after.splitlines(),1) if line=='New detail')
    old=next_=0
    for line in result['unified_diff'].splitlines():
        match=re.match(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@',line)
        if match:old,next_=map(int,match.groups());continue
        if line=='-Old detail':assert old==old_line
        if line=='+New detail':assert next_==new_line
        if line[:1] in (' ','-'):old+=1
        if line[:1] in (' ','+'):next_+=1
    assert result['line_number_scope']=='file'


def test_edit_can_include_its_plain_language_description_in_the_same_call(home_mcp_direct_edit):
    from vecgrep.mcp import server as S
    doc_id=_seed('Old detail')
    result=json.loads(S._run_direct_edit(doc_id,old_str='Old detail',new_str='New detail',change_summary='Corrected the project start date.'))
    assert result['committed'] is True
    assert result['change_summary']=='Corrected the project start date.'
    rejected=json.loads(S._run_direct_edit(doc_id,content='Must not be written',change_summary='x'*301))
    assert 'error' in rejected and 'Must not be written' not in _body_on_disk(doc_id)
