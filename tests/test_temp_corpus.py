"""Temp corpus: docs that carry an expiry and leave search when it passes.

Invariants under test:
  - OFF unless VECGREP_TEMP_CORPUS is set
  - every temp doc has an expiry, capped at VECGREP_TEMP_MAX_TTL_DAYS
  - search drops a doc once its own expiry has passed, before any sweep
  - the sweep deletes only expired docs, and only in the temp corpus
  - the plain direct-write tool cannot target the temp corpus
"""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vecgrep.backend.write import temp as T
from vecgrep.backend.write.proposal import ProposalError

NOW = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)


# ── expiry parsing ───────────────────────────────────────────────────────────

def test_relative_expiry_counts_from_now():
    assert T.parse_expiry("3d", now=NOW) == NOW + timedelta(days=3)
    assert T.parse_expiry("12h", now=NOW) == NOW + timedelta(hours=12)
    assert T.parse_expiry("2w", now=NOW) == NOW + timedelta(weeks=2)


def test_bare_date_expiry_keeps_the_doc_through_that_day():
    assert T.parse_expiry("2026-11-02", now=NOW) == datetime(
        2026, 11, 2, 23, 59, 59, tzinfo=timezone.utc)


@pytest.mark.parametrize("value", [None, "", "  ", "soon", "3 days"])
def test_missing_or_unparseable_expiry_is_refused(value):
    with pytest.raises(ProposalError):
        T.parse_expiry(value, now=NOW)


def test_past_expiry_is_refused():
    with pytest.raises(ProposalError, match="not in the future"):
        T.parse_expiry("2026-10-01", now=NOW)


def test_expiry_past_the_cap_is_refused(monkeypatch):
    monkeypatch.setenv("VECGREP_TEMP_MAX_TTL_DAYS", "30")
    assert T.parse_expiry("30d", now=NOW)
    with pytest.raises(ProposalError, match="30-day limit"):
        T.parse_expiry("31d", now=NOW)


def test_time_left_is_coarse_and_marks_expired():
    meta = {"expires_at": (NOW + timedelta(days=3, hours=5)).isoformat()}
    assert T.time_left(meta, now=NOW) == "3d"
    assert T.time_left({"expires_at": (NOW + timedelta(hours=5)).isoformat()}, now=NOW) == "5h"
    assert T.time_left({"expires_at": (NOW - timedelta(minutes=1)).isoformat()}, now=NOW) == "expired"
    assert T.time_left({}, now=NOW) is None


def test_doc_without_expiry_never_expires():
    assert T.is_expired({}, now=NOW) is False
    assert T.is_expired({"expires_at": "garbage"}, now=NOW) is False


# ── the MCP write path ───────────────────────────────────────────────────────

@pytest.fixture
def temp_env(vg_home, monkeypatch):
    pytest.importorskip("mcp")
    monkeypatch.setenv("VECGREP_TEMP_CORPUS", "temp")
    return vg_home


def _temp_docs(home) -> list[str]:
    return glob.glob(f"{home}/write/temp/*.md")


def test_temp_write_is_off_when_unset(vg_home):
    pytest.importorskip("mcp")
    from vecgrep.mcp import server as S
    r = json.loads(S._run_temp_write("a fact", expires="3d"))
    assert "not enabled" in r["error"]
    assert not glob.glob(f"{vg_home}/write/*/*.md")


def test_temp_write_without_expiry_writes_nothing(temp_env):
    from vecgrep.mcp import server as S
    r = json.loads(S._run_temp_write("a fact", expires=""))
    assert "expiry" in r["error"]
    assert not _temp_docs(temp_env)


def test_temp_write_records_expiry_and_condition(temp_env):
    from vecgrep.mcp import server as S
    r = json.loads(S._run_temp_write(
        "ollama is on CPU while the GPU is out", expires="3d",
        until="the GPU is back", title="GPU out"))
    assert r["committed"] is True, r
    assert r["corpus"] == "temp"
    assert r["expires_in"] == "2d" or r["expires_in"] == "3d"
    body = Path(_temp_docs(temp_env)[0]).read_text()
    assert "expires_at: " in body
    assert "until: the GPU is back" in body
    assert "origin: agent-direct" in body


def test_temp_write_sweeps_expired_docs_first(temp_env):
    from vecgrep.mcp import server as S
    stale = temp_env / "write" / "temp" / "temp-1.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("---\nid: temp-1\nexpires_at: 2020-01-01T00:00:00+00:00\n---\n\nold\n")
    r = json.loads(S._run_temp_write("fresh", expires="1d"))
    assert r["swept"] == 1
    assert not stale.exists()


def test_direct_write_refuses_the_temp_corpus(temp_env, monkeypatch):
    from vecgrep.mcp import server as S
    monkeypatch.setenv("VECGREP_DIRECT_WRITE_CORPUS", "temp")
    r = json.loads(S._run_direct_write("no expiry here"))
    assert "temp" in r["error"]
    assert not _temp_docs(temp_env)


def test_renew_extends_the_expiry(temp_env):
    from vecgrep.mcp import server as S
    doc_id = json.loads(S._run_temp_write("short", expires="1h"))["doc_id"]
    r = json.loads(S._run_temp_renew(doc_id, expires="5d"))
    assert r["expires_in"] in ("4d", "5d"), r
    body = Path(_temp_docs(temp_env)[0]).read_text()
    assert "short" in body and body.count("expires_at:") == 1


def test_renew_refuses_unknown_doc_and_bad_expiry(temp_env):
    from vecgrep.mcp import server as S
    assert "not found" in json.loads(S._run_temp_renew("temp-123", expires="1d"))["error"]
    assert "error" in json.loads(S._run_temp_renew("../../etc/passwd", expires="1d"))
    doc_id = json.loads(S._run_temp_write("x", expires="1h"))["doc_id"]
    assert "limit" in json.loads(S._run_temp_renew(doc_id, expires="400d"))["error"]


# ── search and sweep against the service layer ──────────────────────────────

def _doc(path, body, expires_at=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = f"expires_at: {expires_at}\n" if expires_at else ""
    path.write_text(f"---\nstatus: active\n{fm}---\n\n{body}\n")
    return path


def test_search_drops_expired_docs_before_any_sweep(svc, tmp_path):
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    d = tmp_path / "docs"
    svc.index(str(_doc(d / "a.md", "zebra migration notes stale", past)), "temp")
    svc.index(str(_doc(d / "b.md", "zebra migration notes current", future)), "temp")
    svc.index(str(_doc(d / "c.md", "zebra migration notes permanent")), "temp")
    hits = svc.search("zebra migration notes", corpus_name="temp", top_k=10)
    sources = {os.path.basename(h.source_id) for h in hits}
    assert sources == {"b.md", "c.md"}


def test_result_payload_shows_time_left_only_for_temp_docs(svc, tmp_path):
    pytest.importorskip("mcp")
    from vecgrep.mcp.server import _result_payload
    future = (datetime.now(timezone.utc) + timedelta(days=3, hours=1)).isoformat()
    d = tmp_path / "docs"
    svc.index(str(_doc(d / "a.md", "quokka release pointer", future)), "temp")
    svc.index(str(_doc(d / "b.md", "quokka permanent record")), "keep")
    by_corpus = {h.corpus: _result_payload(h)
                 for h in svc.search("quokka", top_k=10)}
    assert by_corpus["temp"]["expires_in"] == "3d"
    assert "expires_in" not in by_corpus["keep"]


def test_sweep_removes_only_expired_temp_docs(svc, vg_home, monkeypatch):
    monkeypatch.setenv("VECGREP_TEMP_CORPUS", "temp")
    past = "2020-01-01T00:00:00+00:00"
    future = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
    tdir = vg_home / "write" / "temp"
    old = _doc(tdir / "temp-1.md", "walrus stale fact", past)
    new = _doc(tdir / "temp-2.md", "walrus live fact", future)
    other = _doc(vg_home / "write" / "keep" / "keep-1.md", "walrus other corpus", past)
    for p, c in ((old, "temp"), (new, "temp"), (other, "keep")):
        svc.index(str(p), c)

    removed = T.sweep_expired(svc, tdir)

    assert removed == ["temp-1"]
    assert not old.exists() and new.exists() and other.exists()
    temp_sources = svc.registry.get("temp").sources
    assert str(old.resolve()) not in temp_sources
    assert str(new.resolve()) in temp_sources
    assert str(other.resolve()) in svc.registry.get("keep").sources


def test_sweep_is_a_no_op_when_temp_is_off(svc, vg_home):
    tdir = vg_home / "write" / "temp"
    old = _doc(tdir / "temp-1.md", "stale", "2020-01-01T00:00:00+00:00")
    assert T.sweep_expired(svc, tdir) == []
    assert old.exists()


def test_cli_sweep_removes_expired_temp_docs(vg_home, monkeypatch):
    from click.testing import CliRunner

    from vecgrep.cli.main import cli
    monkeypatch.setenv("VECGREP_TEMP_CORPUS", "temp")
    old = _doc(vg_home / "write" / "temp" / "temp-1.md", "stale",
               "2020-01-01T00:00:00+00:00")
    r = CliRunner().invoke(cli, ["temp", "sweep"])
    assert r.exit_code == 0, r.output
    assert "swept 1" in r.output
    assert not old.exists()
