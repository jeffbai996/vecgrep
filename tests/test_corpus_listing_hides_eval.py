"""An eval-* corpus is a build artifact, so it stays out of the corpus LISTS.

`cross_corpus_exclude` already keeps these out of an unscoped search, because
an eval corpus is a side-by-side copy and fanning out over one returns every
hit twice. The same corpora were still listed everywhere — the web UI's corpus
picker and the MCP `list_corpora` tool — where they roughly double the list and
shadow each real corpus with a near-identical name.

Same setting, so search-visibility and list-visibility cannot drift apart.

Filtering happens at the two PRESENTATION edges, never inside
`VecgrepService.list_corpora()`. Backup and the eval harness both call that,
and a backup that silently skips corpora is a far worse bug than an untidy
dropdown.
"""
from __future__ import annotations

import json

import pytest


class TestTheHiddenPredicate:
    def test_eval_prefixed_corpora_are_hidden(self, svc):
        assert svc.is_hidden_corpus("eval-docs-base") is True
        assert svc.is_hidden_corpus("eval-code-base") is True

    def test_real_corpora_are_not(self, svc):
        # "evaluation" is here on purpose: eval-* must not match a corpus that
        # merely starts with those letters.
        for name in ("docs", "code", "notes", "evaluation"):
            assert svc.is_hidden_corpus(name) is False, name

    def test_it_reads_the_same_setting_search_uses(self, svc):
        svc.settings.cross_corpus_exclude = ["scratch-*"]
        assert svc.is_hidden_corpus("scratch-1") is True
        assert svc.is_hidden_corpus("eval-docs-base") is False, (
            "one setting drives both, so it must not hard-code the eval- prefix"
        )

    def test_no_patterns_hides_nothing(self, svc):
        svc.settings.cross_corpus_exclude = []
        assert svc.is_hidden_corpus("eval-docs-base") is False


class TestTheServiceItselfStaysHonest:
    """The guard on the design: filtering belongs at the edges, not in here."""

    def test_list_corpora_still_returns_everything(self, svc, make_doc):
        svc.index(str(make_doc("a.md", "alpha beta gamma.")), "docs")
        svc.index(str(make_doc("b.md", "delta epsilon zeta.")), "eval-docs-base")
        names = {c.name for c in svc.list_corpora()}
        assert names == {"docs", "eval-docs-base"}, (
            "backup and the eval harness call this; hiding corpora here would "
            "silently drop them from a backup"
        )


def _corpus(name):
    """A real Corpus, not a stand-in — the HTTP edge runs asdict() over it."""
    from vecgrep.backend.store.corpora import Corpus
    return Corpus(
        name=name,
        embed_backend="ollama",
        embed_model="bge-m3",
        dim=8,
        description=f"context for {name}",
        use_for=["routing"],
        avoid_for=["guessing"],
    )


class _FakeSvc:
    def __init__(self, names, patterns=("eval-*",)):
        self._names = list(names)
        self._patterns = list(patterns)

    def list_corpora(self):
        return [_corpus(n) for n in self._names]

    def is_hidden_corpus(self, name):
        import fnmatch
        return any(fnmatch.fnmatch(name, p) for p in self._patterns)

    def filterable_fields(self, name):
        return {"filters": []}


NAMES = ["docs", "eval-docs-base", "code", "eval-code-base"]


class TestTheMcpTool:
    def test_eval_corpora_are_omitted_by_default(self, monkeypatch):
        from vecgrep.mcp import server as mcp_server
        monkeypatch.setattr(mcp_server, "_svc", lambda: _FakeSvc(NAMES))
        got = [c["name"] for c in json.loads(mcp_server._run_list_corpora())]
        assert got == ["docs", "code"]

    def test_they_can_still_be_asked_for(self, monkeypatch):
        from vecgrep.mcp import server as mcp_server
        monkeypatch.setattr(mcp_server, "_svc", lambda: _FakeSvc(NAMES))
        got = [c["name"] for c in
               json.loads(mcp_server._run_list_corpora(include_hidden=True))]
        assert got == NAMES, "the eval harness has to be able to see its own builds"

    def test_routing_context_is_exposed(self, monkeypatch):
        from vecgrep.mcp import server as mcp_server
        monkeypatch.setattr(mcp_server, "_svc", lambda: _FakeSvc(["docs"]))
        item = json.loads(mcp_server._run_list_corpora())[0]
        assert item["description"] == "context for docs"
        assert item["use_for"] == ["routing"]
        assert item["avoid_for"] == ["guessing"]
        detail = json.loads(mcp_server._run_get_corpus("docs"))
        assert detail["description"] == "context for docs"
        assert detail["use_for"] == ["routing"]
        assert detail["avoid_for"] == ["guessing"]


class TestTheHttpRoute:
    def test_eval_corpora_are_omitted_by_default(self, monkeypatch):
        from vecgrep.backend.api import routes
        monkeypatch.setattr(routes, "_service", lambda: _FakeSvc(NAMES))
        assert [c.name for c in routes.list_corpora()] == ["docs", "code"]

    def test_they_can_still_be_asked_for(self, monkeypatch):
        from vecgrep.backend.api import routes
        monkeypatch.setattr(routes, "_service", lambda: _FakeSvc(NAMES))
        assert [c.name for c in routes.list_corpora(include_hidden=True)] == NAMES
