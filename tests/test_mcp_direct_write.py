"""MCP direct-write path — the ONE unconfirmed write surface, single-corpus.

`propose_*` is the injection-safe default for every corpus: inert until a human
confirms. This module pins the deliberate exception — a `write` tool that commits
immediately, into exactly one operator-designated corpus, for the case where the
operator IS the one asking (personal note-taking, not dragnet ingestion).

The invariants under test are the whole security story:
  - OFF unless VECGREP_DIRECT_WRITE_CORPUS is set (a fresh clone has no
    unconfirmed write path at all)
  - the corpus is NOT a caller parameter, so no other corpus is even expressible
  - append-only: no edit / delete / patch on this path
  - refuses to operate on a corpus that has a write-through (that would make an
    unconfirmed write mutate an upstream store)
  - rate-capped, size-capped, and stamped origin=agent-direct for audit
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
    body = open(docs[0]).read()
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
    body = open(glob.glob(f"{os.environ['VECGREP_HOME']}/write/external/*.md")[0]).read()
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
