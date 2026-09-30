"""Path moves must not hide duplicated source identities from diagnostics."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _move_to_symlink(svc, tmp_path, *, index_canonical=True):
    old_root = tmp_path / "snapshots"
    old_root.mkdir()
    old = old_root / "guide.md"
    old.write_text("Repository indexing preserves stable source identities.")
    svc.index(str(old), "fixture")
    real_root = tmp_path / "storage"
    old_root.rename(real_root)
    old_root.symlink_to(real_root, target_is_directory=True)
    real = real_root / old.name
    if index_canonical:
        svc.index(str(old), "fixture")
    return old, real


def _alias_issues(svc, corpus="fixture"):
    return [i for i in svc.diagnose(corpora={corpus})
            if i["kind"] == "source_path_aliases"]


def test_doctor_detects_the_move_then_symlink_failure(svc, tmp_path):
    old, real = _move_to_symlink(svc, tmp_path)
    issues = _alias_issues(svc)
    assert len(issues) == 1
    issue = issues[0]
    assert issue["fixable"] is False
    assert issue["extra_sources"] == 1
    assert issue["groups"] == [{
        "canonical_source_id": str(real),
        "source_ids": sorted([str(old), str(real)]),
        "canonical_registered": True,
    }]


def test_doctor_reports_aliases_even_without_a_registered_canonical_path(svc, tmp_path):
    old, real = _move_to_symlink(svc, tmp_path, index_canonical=False)
    second_root = tmp_path / "second"
    second_root.mkdir()
    second = second_root / old.name
    second.write_text(real.read_text())
    svc.index(str(second), "fixture")
    second.unlink()
    second_root.rmdir()
    second_root.symlink_to(real.parent, target_is_directory=True)

    issue, = _alias_issues(svc)
    assert issue["groups"] == [{
        "canonical_source_id": str(real),
        "source_ids": sorted([str(old), str(second)]),
        "canonical_registered": False,
    }]


def test_fix_does_not_choose_or_delete_an_alias(svc, tmp_path, stub_backend):
    _move_to_symlink(svc, tmp_path)
    before = svc.registry.get("fixture")
    vectors = svc.store.source_counts("vecgrep__fixture")
    lexical = list(svc.bm25.iter_sources("fixture"))
    calls = stub_backend.calls

    actions = svc.reconcile(reindex=True, corpora={"fixture"})

    assert actions == [{"corpus": "fixture", "kind": "source_path_aliases",
                        "action": "needs_source_deduplication"}]
    assert svc.registry.get("fixture") == before
    assert svc.store.source_counts("vecgrep__fixture") == vectors
    assert list(svc.bm25.iter_sources("fixture")) == lexical
    assert stub_backend.calls == calls


def test_distinct_files_with_equal_content_are_not_path_aliases(svc, make_doc):
    for name in ("one.md", "two.md"):
        svc.index(str(make_doc(name, "Identical content in distinct source files.")), "fixture")
    assert _alias_issues(svc) == []


def test_hard_links_retain_distinct_canonical_path_identities(svc, make_doc):
    first = make_doc("one.md", "Two intentional hard-link names.")
    second = first.with_name("two.md")
    second.hardlink_to(first)
    for path in (first, second):
        svc.index(str(path), "fixture")
    assert _alias_issues(svc) == []


def test_missing_aliases_are_orphans_and_urls_are_not_files(svc, tmp_path):
    old, real = _move_to_symlink(svc, tmp_path)
    real.unlink()
    corpus = svc.registry.get("fixture")
    corpus.sources.append("https://example.test/guide.md")
    svc.registry.upsert(corpus)
    issues = svc.diagnose(corpora={"fixture"})
    assert not [i for i in issues if i["kind"] == "source_path_aliases"]
    assert {i["source_id"] for i in issues if i["kind"] == "orphaned_source"} == {
        str(old), str(real),
    }


def test_alias_detection_respects_corpus_selection(svc, tmp_path, make_doc):
    _move_to_symlink(svc, tmp_path)
    svc.index(str(make_doc("other.md", "Another corpus.")), "other")
    assert _alias_issues(svc, "other") == []


@pytest.mark.parametrize("fix", [False, True])
def test_real_cli_reports_aliases_and_refuses_a_healthy_exit(svc, tmp_path, fix):
    _move_to_symlink(svc, tmp_path)
    svc.store.client.close()
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root))
    args = [sys.executable, "-m", "vecgrep.cli.main", "doctor",
            "--corpus", "fixture", "--require-healthy"]
    if fix:
        args.append("--fix")
    human = subprocess.run(args, cwd=root, env=env, text=True,
                           capture_output=True, timeout=30)
    assert human.returncode == 1, human.stderr
    assert "source_path_aliases" in human.stdout
    assert "confirm canonical coverage" in human.stdout
    assert "vecgrep index <source>" not in human.stdout
    machine = subprocess.run([*args, "--json"], cwd=root, env=env, text=True,
                             capture_output=True, timeout=30)
    assert machine.returncode == 1, machine.stderr
    report = json.loads(machine.stdout)
    assert report["remaining_issues"][0]["kind"] == "source_path_aliases"
