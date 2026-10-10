"""Corpora left out of a bare query's fan-out but still listed and searchable.

`cross_corpus_exclude` marks a corpus as a build artifact: it is dropped from
the unscoped fan-out AND from every corpus list, deliberately one setting so the
two cannot drift. That is right for eval-* copies and wrong for a large, real
corpus that only drowns the default query -- code or terminal transcripts whose
volume crowds curated corpora out of the reranked pool. `unscoped_exclude`
covers that case: the corpus leaves the default fan-out only. It stays in the
picker, and naming it reaches it.
"""
from __future__ import annotations

import json

from vecgrep.backend import config as config_mod


def _index(svc, make_doc, name, corpus, text="alpha beta gamma delta. epsilon zeta eta theta."):
    p = make_doc(name, text)
    svc.index(str(p), corpus)
    return str(p)


def test_unscoped_search_skips_a_corpus_in_unscoped_exclude(svc, make_doc, monkeypatch) -> None:
    _index(svc, make_doc, "live.md", "chats")
    _index(svc, make_doc, "code.md", "repos")
    monkeypatch.setattr(svc.settings, "unscoped_exclude", ["repos"])

    corpora = {h.corpus for h in svc.search("alpha beta gamma", top_k=20)}
    assert corpora == {"chats"}


def test_naming_an_unscoped_excluded_corpus_still_searches_it(svc, make_doc, monkeypatch) -> None:
    _index(svc, make_doc, "code.md", "repos")
    monkeypatch.setattr(svc.settings, "unscoped_exclude", ["repos"])

    hits = svc.search("alpha beta gamma", corpus_name="repos", top_k=20)
    assert {h.corpus for h in hits} == {"repos"}


def test_an_unscoped_excluded_corpus_is_not_hidden_from_lists(svc, monkeypatch) -> None:
    monkeypatch.setattr(svc.settings, "unscoped_exclude", ["repos", "cli"])
    assert svc.is_hidden_corpus("repos") is False
    assert svc.is_hidden_corpus("cli") is False


def test_build_artifacts_stay_excluded_alongside_it(svc, make_doc, monkeypatch) -> None:
    _index(svc, make_doc, "live.md", "chats")
    _index(svc, make_doc, "copy.md", "eval-chats-base")
    _index(svc, make_doc, "code.md", "repos")
    monkeypatch.setattr(svc.settings, "unscoped_exclude", ["repos"])

    corpora = {h.corpus for h in svc.search("alpha beta gamma", top_k=20)}
    assert corpora == {"chats"}


def test_the_default_excludes_nothing_extra() -> None:
    assert config_mod.Settings().unscoped_exclude == []


def test_unscoped_exclude_loads_from_the_config_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VECGREP_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"unscoped_exclude": ["repos", "cli"]}))
    assert config_mod.load_settings().unscoped_exclude == ["repos", "cli"]


def test_unscoped_exclude_is_an_editable_setting() -> None:
    assert "unscoped_exclude" in config_mod.EDITABLE_FIELDS


# ─── telling callers ────────────────────────────────────────────────────────
# A bot that omits `corpus` silently misses an unscoped-excluded corpus, so
# the tool surface has to say which corpora a bare query covers.

def test_in_default_search_reflects_both_exclusion_settings(svc, monkeypatch) -> None:
    monkeypatch.setattr(svc.settings, "unscoped_exclude", ["repos"])
    assert svc.in_default_search("chats") is True
    assert svc.in_default_search("repos") is False
    assert svc.in_default_search("eval-chats-base") is False


def test_list_corpora_marks_corpora_outside_the_default_search(svc, make_doc, monkeypatch) -> None:
    from vecgrep.mcp import server as mcp_server
    _index(svc, make_doc, "live.md", "chats")
    _index(svc, make_doc, "code.md", "repos")
    monkeypatch.setattr(svc.settings, "unscoped_exclude", ["repos"])
    monkeypatch.setattr(mcp_server, "_svc", lambda: svc)

    listed = {c["name"]: c["in_default_search"] for c in json.loads(mcp_server._run_list_corpora())}
    assert listed == {"chats": True, "repos": False}


def test_the_search_tool_no_longer_claims_omitting_corpus_searches_everything() -> None:
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "vecgrep" / "mcp" / "server.py"
    text = src.read_text()
    assert "Omit to search all corpora" not in text
    assert "in_default_search" in text
