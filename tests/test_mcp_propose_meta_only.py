"""Meta-only (retag) mode on the propose path: tags/source_kind with no
content = a frontmatter-only edit proposal; the body is reused untouched."""
from __future__ import annotations

import glob
import json
import os

import pytest

pytest.importorskip("mcp")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("VECGREP_HOME", str(tmp_path / "vg"))
    monkeypatch.setenv("VECGREP_PROPOSE_ALLOWED_CORPORA", "notes")
    import vecgrep.backend.config as cfg
    monkeypatch.setattr(cfg, "_settings", None)
    return tmp_path


def _seed_doc(doc_id: str, body: str) -> None:
    from vecgrep.mcp import server as S
    d = S._write_dir("notes")
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{doc_id}.md").write_text(f"---\nid: {doc_id}\n---\n\n{body}\n")


def test_meta_only_edit_keeps_body_and_sets_tags(home):
    from vecgrep.mcp import server as S
    _seed_doc("notes-1", "unchanged body")
    out = json.loads(S._run_propose("notes", None, edit_id="notes-1",
                                    tags=["ops", "runbook"]))
    assert "error" not in out, out
    assert "unchanged body" in out["preview"]
    assert "tags: [ops, runbook]" in out["preview"]
    h = os.environ["VECGREP_HOME"]
    pend = json.loads(open(glob.glob(f"{h}/write/_pending/*.json")[0]).read())
    assert pend["meta"].get("meta_only") is True
    # The marker is routing metadata, never rendered frontmatter.
    assert "meta_only" not in pend["rendered"]


def test_meta_only_without_any_meta_is_an_error(home):
    from vecgrep.mcp import server as S
    _seed_doc("notes-1", "x")
    out = json.loads(S._run_propose("notes", None, edit_id="notes-1"))
    assert "error" in out


def test_meta_only_requires_edit_target(home):
    from vecgrep.mcp import server as S
    out = json.loads(S._run_propose("notes", None, tags=["x"]))
    assert "error" in out
