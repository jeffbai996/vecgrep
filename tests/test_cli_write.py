"""CLI write/edit — the human-driven surface for the write tool.

`vecgrep write <corpus> "<content>"` proposes + confirms in one step (the CLI
caller IS the human, so confirmed_by = local user — the wall's human-confirm is
satisfied by the person typing the command). `vecgrep edit <id> "<content>"`
overwrites an existing doc. Docs land under the corpus dir and get indexed so
they're immediately searchable.
"""
from __future__ import annotations

import os

import pytest
from click.testing import CliRunner

from vecgrep.cli.main import cli


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
    assert re.search(r"notes-\d+", r.output)  # assigned id reported
    # the markdown doc was written to the corpus dir (files-are-truth)
    import glob, os as _os
    files = glob.glob(f"{_os.environ['VECGREP_HOME']}/write/notes/notes-*.md")
    assert files, "doc file should exist on disk"
    assert "Cats sit on mats" in open(files[0]).read()
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
    body = open(files[0]).read()
    assert "source_kind: decision" in body
    assert "origin: human" in body            # CLI writes are human-origin
    assert "confirmed_by:" in body            # the local user recorded


def test_edit_overwrites_existing(runner_home):
    import glob
    import re

    w = runner_home.invoke(cli, ["write", "notes", "old content"])
    assert w.exit_code == 0, w.output
    # Use the generated id so the edit exercises the actual newly written doc.
    doc_id = re.search(r"notes-\d+", w.output).group(0)
    r = runner_home.invoke(cli, ["edit", doc_id, "new content"])
    assert r.exit_code == 0, r.output
    f = glob.glob(f"{os.environ['VECGREP_HOME']}/write/notes/{doc_id}.md")[0]
    body = open(f).read()
    assert "new content" in body and "old content" not in body


def test_edit_missing_target_errors(runner_home):
    r = runner_home.invoke(cli, ["edit", "notes-999", "x"])
    assert r.exit_code != 0
    assert "exist" in r.output.lower() or "not found" in r.output.lower()


def test_write_empty_content_errors(runner_home):
    r = runner_home.invoke(cli, ["write", "notes", "   "])
    assert r.exit_code != 0
