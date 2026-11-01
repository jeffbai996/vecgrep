"""Retire: stale docs leave search without a human confirm, and can come back.

Invariants under test:
  - OFF unless VECGREP_DIRECT_WRITE_CORPUS is set, and only that corpus
  - a retired doc leaves search at once; its body stays on disk
  - unretire returns it to search
  - a reason is required; protected docs are refused
  - retiring counts against the hourly direct-write cap; unretiring never does
  - the sweep purges only docs retired longer than the retention period, and
    keeps a .bak of each
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vecgrep.backend.write import retire as R
from vecgrep.backend.write.proposal import ProposalError

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def direct(svc, vg_home, monkeypatch):
    pytest.importorskip("mcp")
    from vecgrep.mcp import server as S
    monkeypatch.setenv("VECGREP_DIRECT_WRITE_CORPUS", "external")
    monkeypatch.setattr(S, "_svc", lambda: svc)
    return S


def _seed(S, content: str) -> str:
    r = json.loads(S._run_direct_write(content, source_kind="fact"))
    assert r.get("committed") is True, r
    return r["doc_id"]


def _path(doc_id: str) -> Path:
    return Path(os.environ["VECGREP_HOME"]) / "write" / "external" / f"{doc_id}.md"


def _found(svc, query: str, doc_id: str) -> bool:
    hits = svc.search(query, corpus_name="external", top_k=10)
    return any(Path(h.source_id).stem == doc_id for h in hits)


def _retired_doc(path: Path, body: str, retired_at: datetime | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = ("status: retired\nretired_reason: stale\n"
          f"retired_at: {retired_at.isoformat()}\n") if retired_at else "status: active\n"
    path.write_text(f"---\nid: {path.stem}\n{fm}---\n\n{body}\n")
    return path


# ── metadata ─────────────────────────────────────────────────────────────────

def test_retire_meta_needs_a_reason():
    with pytest.raises(ProposalError, match="reason"):
        R.retire_meta({"status": "active"}, "   ", now=NOW)


def test_retire_meta_flattens_the_reason_to_one_line():
    meta = R.retire_meta({"status": "active"}, "moved to\nthe runbook", now=NOW)
    assert meta["retired_reason"] == "moved to the runbook"
    assert meta["status"] == "retired"
    assert meta["retired_at"] == NOW.isoformat(timespec="seconds")


def test_unretire_meta_clears_every_retirement_field():
    meta = R.retire_meta({"status": "active", "tier": "normal"}, "stale",
                         superseded_by="external-2", now=NOW)
    back = R.unretire_meta(meta)
    assert back == {"status": "active", "tier": "normal"}


# ── the MCP tools ────────────────────────────────────────────────────────────

def test_retire_is_off_without_a_direct_write_corpus(direct, monkeypatch):
    doc_id = _seed(direct, "heron nesting site west pier")
    monkeypatch.delenv("VECGREP_DIRECT_WRITE_CORPUS")
    r = json.loads(direct._run_retire(doc_id, "stale"))
    assert "not enabled" in r["error"]
    assert "status: active" in _path(doc_id).read_text()


def test_retired_doc_leaves_search_and_keeps_its_body(direct, svc):
    doc_id = _seed(direct, "heron nesting site west pier")
    assert _found(svc, "heron nesting site", doc_id)

    r = json.loads(direct._run_retire(doc_id, "the herons moved",
                                      superseded_by="external-9"))

    assert r["retired"] is True, r
    assert not _found(svc, "heron nesting site", doc_id)
    text = _path(doc_id).read_text()
    assert "heron nesting site west pier" in text
    assert "status: retired" in text
    assert "retired_reason: the herons moved" in text
    assert "superseded_by: external-9" in text


def test_unretire_returns_the_doc_to_search(direct, svc):
    doc_id = _seed(direct, "heron nesting site west pier")
    direct._run_retire(doc_id, "wrong call")

    r = json.loads(direct._run_unretire(doc_id))

    assert r["unretired"] is True, r
    assert _found(svc, "heron nesting site", doc_id)
    assert "retired_" not in _path(doc_id).read_text()


def test_unretire_refuses_an_active_doc(direct):
    doc_id = _seed(direct, "heron nesting site west pier")
    r = json.loads(direct._run_unretire(doc_id))
    assert "not retired" in r["error"]


def test_retiring_twice_keeps_the_first_retired_at(direct):
    doc_id = _seed(direct, "heron nesting site west pier")
    first = json.loads(direct._run_retire(doc_id, "stale"))
    second = json.loads(direct._run_retire(doc_id, "stale again"))
    assert second["retired_at"] == first["retired_at"]
    assert "retired_reason: stale\n" in _path(doc_id).read_text()


def test_retire_refuses_protected_docs(direct):
    doc_id = _seed(direct, "heron nesting site west pier")
    p = _path(doc_id)
    p.write_text(p.read_text().replace("tier: normal", "tier: protected"))
    r = json.loads(direct._run_retire(doc_id, "stale"))
    assert "protected" in r["error"]
    assert "status: active" in p.read_text()


@pytest.mark.parametrize("doc_id", ["notes-abc", "../external-1", "external-"])
def test_retire_refuses_docs_outside_the_direct_write_corpus(direct, doc_id):
    r = json.loads(direct._run_retire(doc_id, "stale"))
    assert "error" in r


def test_retire_counts_against_the_hourly_cap(direct, monkeypatch):
    doc_id = _seed(direct, "heron nesting site west pier")
    monkeypatch.setenv("VECGREP_DIRECT_WRITE_MAX_PER_HOUR", "1")
    r = json.loads(direct._run_retire(doc_id, "stale"))
    assert "rate cap" in r["error"]
    assert "status: active" in _path(doc_id).read_text()


def test_unretire_is_never_rate_limited(direct, monkeypatch):
    doc_id = _seed(direct, "heron nesting site west pier")
    direct._run_retire(doc_id, "stale")
    monkeypatch.setenv("VECGREP_DIRECT_WRITE_MAX_PER_HOUR", "1")
    r = json.loads(direct._run_unretire(doc_id))
    assert r["unretired"] is True, r


def test_retire_sweeps_old_retirements_first(direct, vg_home):
    old = _retired_doc(vg_home / "write" / "external" / "external-1.md",
                       "egret roost", NOW - timedelta(days=400))
    doc_id = _seed(direct, "heron nesting site west pier")
    r = json.loads(direct._run_retire(doc_id, "stale"))
    assert r["swept"] == 1
    assert not old.exists()


# ── the sweep ────────────────────────────────────────────────────────────────

def test_sweep_purges_only_docs_retired_past_retention(svc, vg_home, monkeypatch):
    monkeypatch.setenv("VECGREP_DIRECT_WRITE_CORPUS", "external")
    wdir = vg_home / "write" / "external"
    old = _retired_doc(wdir / "external-1.md", "egret roost old", NOW - timedelta(days=31))
    recent = _retired_doc(wdir / "external-2.md", "egret roost recent", NOW - timedelta(days=5))
    live = _retired_doc(wdir / "external-3.md", "egret roost live", None)
    for p in (old, recent, live):
        svc.index(str(p), "external")

    removed = R.sweep_retired(svc, wdir, now=NOW)

    assert removed == ["external-1"]
    assert not old.exists() and recent.exists() and live.exists()
    assert list(wdir.glob("external-1.md.bak-*"))
    sources = svc.registry.get("external").sources
    assert str(old.resolve()) not in sources
    assert str(recent.resolve()) in sources


def test_sweep_is_a_no_op_without_a_direct_write_corpus(svc, vg_home):
    wdir = vg_home / "write" / "external"
    old = _retired_doc(wdir / "external-1.md", "egret roost", NOW - timedelta(days=400))
    assert R.sweep_retired(svc, wdir, now=NOW) == []
    assert old.exists()


def test_cli_sweep_purges_old_retirements(vg_home, monkeypatch):
    from click.testing import CliRunner

    from vecgrep.cli.main import cli
    monkeypatch.setenv("VECGREP_DIRECT_WRITE_CORPUS", "external")
    old = _retired_doc(vg_home / "write" / "external" / "external-1.md",
                       "egret roost", NOW - timedelta(days=400))
    r = CliRunner().invoke(cli, ["retired", "sweep"])
    assert r.exit_code == 0, r.output
    assert "swept 1" in r.output
    assert not old.exists()
