"""MCP propose_merge — the merge proposal surface.

Same wall as the other propose tools: default-deny corpus, size cap, and
nothing written until a human confirms. These pin the tool's contract, not
the merge mechanics (those live in test_propose_merge/test_confirm_merge).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("mcp")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("VECGREP_HOME", str(tmp_path / "vg"))
    monkeypatch.setenv("VECGREP_PROPOSE_ALLOWED_CORPORA", "notes")
    import vecgrep.backend.config as cfg
    monkeypatch.setattr(cfg, "_settings", None)
    return tmp_path


def _seed(d: Path, doc_id: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{doc_id}.md").write_text(f"---\nid: {doc_id}\n---\n\nbody {doc_id}\n")


def test_mcp_merge_happy_path(home):
    from vecgrep.mcp import server as S
    d = S._corpus_doc_dir("notes")
    _seed(d, "notes-1")
    _seed(d, "notes-2")
    out = json.loads(S._run_propose_merge("notes", ["notes-1", "notes-2"],
                                          "merged synthesis"))
    assert out.get("doc_id") == "notes-1"
    assert out.get("absorbs") == ["notes-2"]
    assert "pending" in out.get("status", "")
    # Proposal only — the absorbed doc is still on disk.
    assert (d / "notes-2.md").exists()


def test_mcp_merge_respects_corpus_allowlist(home, monkeypatch):
    from vecgrep.mcp import server as S
    monkeypatch.setenv("VECGREP_PROPOSE_ALLOWED_CORPORA", "notes")
    out = json.loads(S._run_propose_merge("forbidden", ["x-1", "x-2"], "m"))
    assert "error" in out
    assert "not agent-writable" in out["error"]


def test_mcp_merge_rejects_empty_content(home):
    from vecgrep.mcp import server as S
    d = S._corpus_doc_dir("notes")
    _seed(d, "notes-1")
    _seed(d, "notes-2")
    out = json.loads(S._run_propose_merge("notes", ["notes-1", "notes-2"], "   "))
    assert "error" in out
    assert "no auto-concat" in out["error"]


def test_mcp_merge_surfaces_proposal_errors(home):
    from vecgrep.mcp import server as S
    d = S._corpus_doc_dir("notes")
    _seed(d, "notes-1")
    # One id — propose_merge raises ProposalError, which must surface as JSON.
    out = json.loads(S._run_propose_merge("notes", ["notes-1"], "m"))
    assert "error" in out
    assert "at least two" in out["error"]
