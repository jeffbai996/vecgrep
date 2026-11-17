from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import click
import httpx
import pytest
from click.testing import CliRunner
from watchfiles import Change

from vecgrep import __version__
from vecgrep.cli import main as cli_main
from vecgrep.cli import main as m
from vecgrep.cli.main import cli

# ── test_cli_watch_multiple_roots ──

def test_watch_indexes_and_follows_both_roots(tmp_path: Path):
    roots = [tmp_path / 'local', tmp_path / 'desktop']
    for root in roots:
        root.mkdir()
        (root / 'session.md').write_text('new session content')
    indexed = []
    watched = []

    def events(*paths, **kwargs):
        watched.extend(paths)
        yield {(Change.modified, str(root / 'session.md')) for root in roots}
        yield {(Change.modified, str(roots[1] / 'state.json'))}

    m._WATCH_SEEN_HASHES.clear()
    m._WATCH_PENDING.clear()
    with patch('watchfiles.watch', events), patch.object(
        m, '_do_index', side_effect=lambda source, *a, **kw: indexed.append(source)
    ):
        result = CliRunner().invoke(m.cli, ['watch', str(roots[0]), '--also-watch',
            str(roots[1]), '--corpus', 'test', '--include', '*.md', '--quiet-period', '0'])
    assert result.exit_code == 0, result.output
    assert watched == list(map(str, roots))
    assert indexed[:2] == list(map(str, roots))
    assert set(indexed[2:]) == {str(root / 'session.md') for root in roots}


def test_watch_validates_all_roots_before_indexing(tmp_path: Path):
    with patch.object(m, '_do_index') as index:
        result = CliRunner().invoke(m.cli, ['watch', str(tmp_path), '--also-watch',
            str(tmp_path / 'missing'), '--corpus', 'test'])
    assert result.exit_code != 0
    assert 'must be a directory' in result.output
    index.assert_not_called()


def test_initial_failure_does_not_skip_other_root(tmp_path: Path):
    other = tmp_path / 'desktop'
    other.mkdir()
    with patch('watchfiles.watch', return_value=iter(())), patch.object(
        m, '_do_index', side_effect=[TimeoutError('temporary'), None]
    ) as index:
        result = CliRunner().invoke(m.cli, ['watch', str(tmp_path), '--also-watch',
            str(other), '--corpus', 'test'])
    assert result.exit_code == 0, result.output
    assert [call.args[0] for call in index.call_args_list] == [str(tmp_path), str(other)]
    assert 'watching anyway' in result.output

# ── test_cli_watch_unchanged_skip ──

def test_unchanged_file_is_not_dispatched(tmp_path: Path) -> None:
    """Second event on identical content must not call the index path."""
    f = tmp_path / "a.md"
    f.write_text("hello world", encoding="utf-8")

    calls: list[str] = []
    with patch.object(m, "_do_index", side_effect=lambda s, *a, **k: calls.append(s)):
        assert m._watch_should_index(str(f)) is True
        m._do_index(str(f), "c", "auto", force=False)
        # Same bytes, new event (mtime touched, content identical).
        assert m._watch_should_index(str(f)) is False, (
            "an unchanged file must be skipped BEFORE the index dispatch — "
            "that skip is what stops the runaway re-index loop")
    assert calls == [str(f)], calls


def test_changed_content_still_indexes(tmp_path: Path) -> None:
    """A real edit must still get through — the guard is a skip, not a block."""
    f = tmp_path / "a.md"
    f.write_text("v1", encoding="utf-8")
    assert m._watch_should_index(str(f)) is True
    f.write_text("v2 — genuinely different", encoding="utf-8")
    assert m._watch_should_index(str(f)) is True, "content changed → must index"


def test_unreadable_file_defaults_to_indexing(tmp_path: Path) -> None:
    """If we can't hash it, fall through to the old behaviour rather than
    silently dropping a real change."""
    assert m._watch_should_index(str(tmp_path / "does-not-exist.md")) is True

# ── test_cli_narrowed_handlers ──

def _detail_of(response: httpx.Response) -> str:
    with pytest.raises(click.ClickException) as raised:
        cli_main._raise_for_status(response)
    return raised.value.message


def test_error_detail_uses_the_json_detail_field() -> None:
    assert _detail_of(httpx.Response(400, json={"detail": "unknown corpus"})) == "unknown corpus"


def test_error_detail_falls_back_to_body_text_when_body_is_not_json() -> None:
    assert _detail_of(httpx.Response(502, text="bad gateway")) == "bad gateway"


@pytest.mark.parametrize("body", [b'["a list"]', b'"a string"', b"7", b"null"])
def test_error_detail_falls_back_to_body_text_when_json_is_not_an_object(body: bytes) -> None:
    assert _detail_of(httpx.Response(500, content=body)) == body.decode()


def test_error_detail_does_not_mask_an_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_json(self, **kwargs):
        raise RuntimeError("bug in the response handling")

    monkeypatch.setattr(httpx.Response, "json", broken_json)
    with pytest.raises(RuntimeError, match="bug in the response handling"):
        cli_main._raise_for_status(httpx.Response(500, text="x"))


def test_pending_lists_readable_proposals_and_skips_unreadable_ones(tmp_path, monkeypatch) -> None:
    home = tmp_path / "vg"
    monkeypatch.setenv("VECGREP_HOME", str(home))
    from vecgrep.backend import config

    monkeypatch.setattr(config, "_settings", None)
    pending = home / "write" / "_pending"
    pending.mkdir(parents=True)
    (pending / "a-good.json").write_text(json.dumps({
        "proposal_id": "p-good", "doc_id": "notes-001", "corpus": "notes",
        "meta": {"origin": "test"},
    }))
    (pending / "b-truncated.json").write_text('{"proposal_id": ')
    (pending / "c-not-utf8.json").write_bytes(b"\xff\xfe\x00")
    (pending / "d-directory.json").mkdir()

    result = CliRunner().invoke(cli_main.cli, ["pending"])

    assert result.exit_code == 0, result.output
    assert "p-good  [new] notes-001 (notes)  origin=test" in result.output
    assert result.output.count("\n") == 1

# ── test_cli_doctor ──

class _DoctorService:
    def __init__(self, post_repair_issues=None) -> None:
        self.diagnose_scope = None
        self.reconcile_scope = None
        self.reindex = None
        self.reconciled = False
        self.post_repair_issues = post_repair_issues

    def list_corpora(self):
        return [SimpleNamespace(name="cli")]

    def diagnose(self, *, corpora=None):
        self.diagnose_scope = corpora
        if self.reconciled and self.post_repair_issues is not None:
            return self.post_repair_issues
        return [{"corpus": "cli", "kind": "count_drift", "detail": "short", "fixable": True}]

    def reconcile(self, *, reindex=False, corpora=None):
        self.reindex = reindex
        self.reconcile_scope = corpora
        self.reconciled = True
        return [{"corpus": "cli", "kind": "count_drift", "action": "reindexed"}]


def test_doctor_fix_scopes_recovery_and_uses_read_only_cache(monkeypatch):
    service = _DoctorService()
    made = {}

    def build_service(*, embed_cache_read_only=False):
        made["read_only"] = embed_cache_read_only
        return service

    monkeypatch.setattr("vecgrep.cli.main.VecgrepService", build_service)
    result = CliRunner().invoke(cli, ["doctor", "--fix", "--corpus", "cli"])

    assert result.exit_code == 0, result.output
    assert made["read_only"] is True
    assert service.diagnose_scope == {"cli"}
    assert service.reconcile_scope == {"cli"}
    assert service.reindex is True


def test_doctor_require_healthy_succeeds_only_after_repair(monkeypatch):
    service = _DoctorService(post_repair_issues=[])
    monkeypatch.setattr("vecgrep.cli.main.VecgrepService", lambda **_: service)

    result = CliRunner().invoke(
        cli, ["doctor", "--fix", "--corpus", "cli", "--require-healthy"]
    )

    assert result.exit_code == 0, result.output
    assert service.reconciled is True


def test_doctor_require_healthy_fails_when_repair_is_incomplete(monkeypatch):
    remaining = [{"corpus": "cli", "kind": "count_drift", "detail": "short", "fixable": True}]
    service = _DoctorService(post_repair_issues=remaining)
    monkeypatch.setattr("vecgrep.cli.main.VecgrepService", lambda **_: service)

    result = CliRunner().invoke(
        cli, ["doctor", "--fix", "--corpus", "cli", "--require-healthy"]
    )

    assert result.exit_code != 0
    assert "still unhealthy" in result.output

# ── test_cli_watch_resilience ──

def _file(tmp_path: Path) -> str:
    f = tmp_path / "a.md"
    f.write_text("content", encoding="utf-8")
    return str(f)


def test_transient_backend_error_is_swallowed_and_reported(tmp_path: Path) -> None:
    p = _file(tmp_path)
    m._WATCH_SEEN_HASHES.clear()
    with patch.object(m, "_do_index", side_effect=TimeoutError("backend slow")):
        assert m._watch_index_resilient(p, "c", "auto", None) is False


def test_click_exception_is_swallowed_too(tmp_path: Path) -> None:
    p = _file(tmp_path)
    m._WATCH_SEEN_HASHES.clear()
    with patch.object(m, "_do_index",
                      side_effect=click.ClickException("bad file")):
        assert m._watch_index_resilient(p, "c", "auto", None) is False


def test_keyboard_interrupt_still_propagates(tmp_path: Path) -> None:
    """Ctrl+C must stop the watcher — the guard is for failures, not intent."""
    p = _file(tmp_path)
    m._WATCH_SEEN_HASHES.clear()
    with patch.object(m, "_do_index", side_effect=KeyboardInterrupt), pytest.raises(KeyboardInterrupt):
        m._watch_index_resilient(p, "c", "auto", None)


def test_successful_index_reports_clean(tmp_path: Path) -> None:
    p = _file(tmp_path)
    m._WATCH_SEEN_HASHES.clear()
    with patch.object(m, "_do_index"):
        assert m._watch_index_resilient(p, "c", "auto", None) is True


def test_a_failed_file_is_retried_on_its_next_event(tmp_path: Path) -> None:
    """The failure path must not poison the seen-hash cache: the file's next
    event has identical content, and a hash recorded before a FAILED dispatch
    would make the skip-guard drop it forever."""
    p = _file(tmp_path)
    m._WATCH_SEEN_HASHES.clear()
    with patch.object(m, "_do_index", side_effect=TimeoutError("slow")):
        m._watch_index_resilient(p, "c", "auto", None)
    calls: list[str] = []
    with patch.object(m, "_do_index",
                      side_effect=lambda s, *a, **k: calls.append(s)):
        m._watch_index_resilient(p, "c", "auto", None)
    assert calls == [p], (
        "same content after a failed dispatch must still be indexed — "
        "the hash guard may only remember SUCCESSFUL dispatches")

# ── test_cli_watch_quiet_period ──

def _aged(path: Path, seconds_ago: float) -> str:
    past = time.time() - seconds_ago
    os.utime(path, (past, past))
    return str(path)


def test_freshly_written_file_is_not_quiet(tmp_path: Path) -> None:
    f = tmp_path / "live.md"
    f.write_text("a live session is appending here", encoding="utf-8")
    assert m._watch_is_quiet(str(f), quiet_period=300) is False, (
        "a file modified seconds ago belongs to a live session — indexing it "
        "now is the re-index treadmill")


def test_old_file_is_quiet(tmp_path: Path) -> None:
    f = tmp_path / "settled.md"
    f.write_text("session over", encoding="utf-8")
    assert m._watch_is_quiet(_aged(f, 20 * 60), quiet_period=300) is True


def test_zero_quiet_period_disables_the_guard(tmp_path: Path) -> None:
    f = tmp_path / "live.md"
    f.write_text("fresh", encoding="utf-8")
    assert m._watch_is_quiet(str(f), quiet_period=0) is True


def test_missing_file_counts_as_quiet(tmp_path: Path) -> None:
    """Can't stat it → let the downstream path handle (and report) it rather
    than parking it in pending forever."""
    assert m._watch_is_quiet(str(tmp_path / "gone.md"), quiet_period=300) is True


def test_deferred_file_becomes_due_after_going_quiet(tmp_path: Path) -> None:
    f = tmp_path / "transcript.md"
    f.write_text("turn 1", encoding="utf-8")

    m._WATCH_PENDING.clear()
    m._WATCH_PENDING.add(str(f))
    assert m._watch_due_pending(quiet_period=300) == [], (
        "still hot — must stay parked")
    assert str(f) in m._WATCH_PENDING

    _aged(f, 20 * 60)
    assert m._watch_due_pending(quiet_period=300) == [str(f)], (
        "quiet now — the deferred write must surface for indexing")
    assert str(f) not in m._WATCH_PENDING, "due files leave the pending set"


def test_deleted_file_leaves_the_pending_set(tmp_path: Path) -> None:
    """A deferred file that gets deleted must not resurface for indexing."""
    f = tmp_path / "doomed.md"
    f.write_text("x", encoding="utf-8")
    m._WATCH_PENDING.clear()
    m._WATCH_PENDING.add(str(f))
    f.unlink()
    m._WATCH_PENDING.discard(str(f))
    assert m._watch_due_pending(quiet_period=300) == []

# ── test_cli_write ──

@pytest.fixture
def runner_home(svc, monkeypatch):
    import importlib

    cli_module = importlib.import_module("vecgrep.cli.main")
    # Reuse the hermetic service and stub embedder across commands in one test.
    monkeypatch.setattr(cli_module, "VecgrepService", lambda **kwargs: svc)
    return CliRunner()


def test_write_creates_and_indexes_a_doc(runner_home):
    import re
    r = runner_home.invoke(cli, ["write", "notes", "Cats sit on mats."])
    assert r.exit_code == 0, r.output
    # ids are nanosecond timestamps now (notes-<digits>), not a serial.
    assert re.search(r"notes-[0-9a-z]+", r.output)  # assigned id reported
    # the markdown doc was written to the corpus dir (files-are-truth)
    import glob
    import os as _os
    files = glob.glob(f"{_os.environ['VECGREP_HOME']}/write/notes/notes-*.md")
    assert files, "doc file should exist on disk"
    assert "Cats sit on mats" in Path(files[0]).read_text()
    # (full semantic-search retrieval needs a live embedder — covered by the
    #  service-layer tests, not this hermetic CLI test.)


def test_write_records_metadata_and_confirmer(runner_home):
    r = runner_home.invoke(cli, ["write", "notes", "a decision",
                                 "--source-kind", "decision", "--tag", "x", "--tag", "y"])
    assert r.exit_code == 0
    # the on-disk file carries the schema + a human confirmer
    home = os.environ["VECGREP_HOME"]
    import glob
    files = glob.glob(f"{home}/write/notes/notes-*.md")
    assert files
    body = Path(files[0]).read_text()
    assert "source_kind: decision" in body
    assert "origin: human" in body            # CLI writes are human-origin
    assert "confirmed_by:" in body            # the local user recorded


def test_edit_overwrites_existing(runner_home):
    import glob
    import re

    w = runner_home.invoke(cli, ["write", "notes", "old content"])
    assert w.exit_code == 0, w.output
    # Use the generated id so the edit exercises the actual newly written doc.
    match = re.search(r"notes-[0-9a-z]+", w.output)
    assert match is not None
    doc_id = match.group(0)
    r = runner_home.invoke(cli, ["edit", doc_id, "new content"])
    assert r.exit_code == 0, r.output
    f = glob.glob(f"{os.environ['VECGREP_HOME']}/write/notes/{doc_id}.md")[0]
    body = Path(f).read_text()
    assert "new content" in body and "old content" not in body


def test_edit_missing_target_errors(runner_home):
    r = runner_home.invoke(cli, ["edit", "notes-999", "x"])
    assert r.exit_code != 0
    assert "exist" in r.output.lower() or "not found" in r.output.lower()


def test_write_empty_content_errors(runner_home):
    r = runner_home.invoke(cli, ["write", "notes", "   "])
    assert r.exit_code != 0

# ── test_cli_search_watch ──

def _hit(sid: str, pct: float = 90.0) -> dict[str, Any]:
    return {
        "similarity_pct": pct,
        "chunk": f"chunk for {sid}",
        "context_before": "",
        "context_after": "",
        "source_id": sid,
        "corpus": "test",
        "metadata": {},
        "matched_by": ["V"],
        "explain": {},
    }


def test_watch_diff_shows_new_and_removed_sources(vg_home: Path) -> None:
    """Two iterations: first returns {a, b}, second returns {b, c}.
    Output should mark 'c' as new and 'a' as gone."""
    runner = CliRunner()
    iterations = [
        {"hits": [_hit("a"), _hit("b")]},
        {"hits": [_hit("b"), _hit("c")]},
    ]
    call_count = {"n": 0}

    def fake_post(path: str, payload: dict) -> Any:
        i = call_count["n"]
        call_count["n"] += 1
        if i >= len(iterations):
            raise KeyboardInterrupt
        return iterations[i]

    with (
        patch("vecgrep.cli.main._api_alive", return_value=True),
        patch("vecgrep.cli.main._post", side_effect=fake_post),
        patch("vecgrep.cli.main.time.sleep", return_value=None),
    ):
        result = runner.invoke(cli, ["search", "x", "--corpus", "test", "--watch", "--interval", "0"])
    assert result.exit_code == 0, result.output
    # 'c' arrived in iteration 2; 'a' disappeared. Diff lines start with '+ ' and '- '.
    iter2 = result.output.split("iteration 2")[-1]
    assert "+" in iter2 and "test" in iter2 and " c" in iter2, iter2
    assert "-" in iter2 and "test" in iter2 and " a" in iter2, iter2


def test_watch_first_iteration_treats_all_as_new(vg_home: Path) -> None:
    """The first pass has no baseline, so every hit is 'new'. The diff
    block should NOT call existing hits 'gone' on iteration 1."""
    runner = CliRunner()
    iterations = [{"hits": [_hit("a"), _hit("b")]}]
    call_count = {"n": 0}

    def fake_post(path: str, payload: dict) -> Any:
        i = call_count["n"]
        call_count["n"] += 1
        if i >= len(iterations):
            raise KeyboardInterrupt
        return iterations[i]

    with (
        patch("vecgrep.cli.main._api_alive", return_value=True),
        patch("vecgrep.cli.main._post", side_effect=fake_post),
        patch("vecgrep.cli.main.time.sleep", return_value=None),
    ):
        result = runner.invoke(cli, ["search", "x", "--corpus", "test", "--watch", "--interval", "0"])
    assert result.exit_code == 0, result.output
    # We can't pin format precisely but iteration 1 should not contain '-' diff lines
    # for any of {a, b}. (Diff lines only appear from iteration 2 onward.)
    iter1_section = result.output.split("iteration 2")[0] if "iteration 2" in result.output else result.output
    assert "- a" not in iter1_section
    assert "- b" not in iter1_section


def test_watch_keyboard_interrupt_exits_cleanly(vg_home: Path) -> None:
    """Ctrl-C during watch should exit 0, not raise. Also ensures we don't
    crash if the very first iteration is interrupted."""
    runner = CliRunner()

    def fake_post(*a, **kw) -> Any:
        raise KeyboardInterrupt

    with (
        patch("vecgrep.cli.main._api_alive", return_value=True),
        patch("vecgrep.cli.main._post", side_effect=fake_post),
        patch("vecgrep.cli.main.time.sleep", return_value=None),
    ):
        result = runner.invoke(cli, ["search", "x", "--corpus", "test", "--watch", "--interval", "0"])
    assert result.exit_code == 0, result.output

# ── test_cli_status ──

def _fake_corpus_dict(name: str, docs: int, chunks: int, updated: float = 0.0) -> dict[str, Any]:
    return {
        "name": name,
        "embed_backend": "ollama",
        "embed_model": "nomic-embed-text",
        "dim": 768,
        "chunker": "sentence_window",
        "doc_count": docs,
        "chunk_count": chunks,
        "sources": [],
        "created_at": updated,
        "updated_at": updated,
    }


def test_status_daemon_up_human_output(vg_home: Path) -> None:
    """When the daemon is reachable, status hits /api/corpora and renders
    a human block including 'daemon: up' and the version."""
    runner = CliRunner()
    fake_corpora = [
        _fake_corpus_dict("chats", 3, 8200),
        _fake_corpus_dict("repos", 42, 4787),
    ]
    with (
        patch("vecgrep.cli.main._api_alive", return_value=True),
        patch("vecgrep.cli.main._get", return_value=fake_corpora),
    ):
        result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output
    assert "daemon" in result.output and "up" in result.output
    assert "chats" in result.output
    assert "repos" in result.output
    assert "8200" in result.output
    assert "4787" in result.output


def test_status_daemon_down_falls_back_to_in_process(vg_home: Path, svc, make_doc) -> None:
    """When the daemon is dead, status synthesizes its view from a local
    VecgrepService. No HTTP call should happen."""
    p = make_doc("a.txt", "hello world. small corpus for the status test.")
    svc.index(str(p), "fixture", "sentence_window")

    runner = CliRunner()
    with (
        patch("vecgrep.cli.main._api_alive", return_value=False),
        patch("vecgrep.cli.main._get") as mock_get,
        patch("vecgrep.cli.main.VecgrepService", return_value=svc),
    ):
        result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0, result.output
    mock_get.assert_not_called()
    assert "daemon" in result.output
    assert "down" in result.output
    assert "fixture" in result.output


def test_status_json_output_shape(vg_home: Path) -> None:
    """--json emits a structured object: top-level keys version, daemon,
    auth, home, corpora (list)."""
    runner = CliRunner()
    fake_corpora = [_fake_corpus_dict("solo", 1, 5)]
    with (
        patch("vecgrep.cli.main._api_alive", return_value=True),
        patch("vecgrep.cli.main._get", return_value=fake_corpora),
    ):
        result = runner.invoke(cli, ["status", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["version"] == __version__
    assert payload["daemon"]["alive"] is True
    assert "auth" in payload
    assert "home" in payload
    assert isinstance(payload["corpora"], list)
    assert payload["corpora"][0]["name"] == "solo"
    assert payload["corpora"][0]["chunk_count"] == 5
    assert payload["totals"]["chunks"] == 5
    assert payload["totals"]["docs"] == 1


def test_status_auth_field_reflects_token_setting(vg_home: Path, monkeypatch) -> None:
    """When VECGREP_API_TOKEN is set, status reports auth: enabled."""
    monkeypatch.setenv("VECGREP_API_TOKEN", "deadbeef")
    # Force settings reload so the env var wins.
    from vecgrep.backend import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "_settings", None)

    runner = CliRunner()
    with (
        patch("vecgrep.cli.main._api_alive", return_value=True),
        patch("vecgrep.cli.main._get", return_value=[]),
    ):
        result = runner.invoke(cli, ["status", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["auth"]["enabled"] is True


def test_status_empty_corpora_renders_cleanly(vg_home: Path) -> None:
    """No corpora → still prints the header block and a 'no corpora' line,
    not a crash or empty table."""
    runner = CliRunner()
    with (
        patch("vecgrep.cli.main._api_alive", return_value=True),
        patch("vecgrep.cli.main._get", return_value=[]),
    ):
        result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0, result.output
    assert "no corpora" in result.output.lower()


def test_status_renders_updated_age_column(vg_home: Path) -> None:
    """The text table should include a relative-time UPDATED column so
    you can spot a stale corpus across machines."""
    import time

    runner = CliRunner()
    now = time.time()
    fake_corpora = [
        _fake_corpus_dict("fresh", 1, 10, updated=now - 60),         # ~1m ago
        _fake_corpus_dict("medium", 1, 10, updated=now - 3600 * 4),  # ~4h ago
        _fake_corpus_dict("stale", 1, 10, updated=now - 86400 * 3),  # ~3d ago
    ]
    with (
        patch("vecgrep.cli.main._api_alive", return_value=True),
        patch("vecgrep.cli.main._get", return_value=fake_corpora),
    ):
        result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0, result.output
    assert "UPDATED" in result.output
    # All three rows must render *some* age string. We don't pin the exact
    # format ("1m ago" vs "1 min ago") — just that it isn't empty / "0".
    for line in result.output.splitlines():
        if line.strip().startswith(("fresh", "medium", "stale")):
            # Last token on the row should be the relative time.
            tokens = line.split()
            assert tokens[-1] not in ("0", ""), f"row missing age: {line!r}"


def test_relative_time_helper_handles_ranges() -> None:
    """The helper covers the bands we care about: now, minutes, hours,
    days, and 'unknown' for a zero/missing timestamp."""
    import time

    from vecgrep.cli.main import _relative_age

    now = time.time()
    assert "now" in _relative_age(now).lower() or "s" in _relative_age(now)
    assert "m" in _relative_age(now - 120)         # ~2m
    assert "h" in _relative_age(now - 3600 * 5)    # ~5h
    assert "d" in _relative_age(now - 86400 * 2)   # ~2d
    assert _relative_age(0).lower() in ("never", "—", "-", "unknown")


def test_status_totals_in_human_output(vg_home: Path) -> None:
    """The text table footer should include summed docs and chunks across
    all corpora — that's the whole point of running status."""
    runner = CliRunner()
    fake_corpora = [
        _fake_corpus_dict("a", 10, 100),
        _fake_corpus_dict("b", 20, 250),
    ]
    with (
        patch("vecgrep.cli.main._api_alive", return_value=True),
        patch("vecgrep.cli.main._get", return_value=fake_corpora),
    ):
        result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0, result.output
    # Totals row: 30 docs, 350 chunks.
    assert "30" in result.output
    assert "350" in result.output
