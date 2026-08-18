"""Cross-corpus search scope: which corpora a bare query fans out over.

An `eval-*` corpus is a side-by-side copy of a live corpus built by
vecgrep.eval to price a config change. Including it in an unscoped search
double-counts the source corpus -- every hit appears twice, verbatim, eating
two slots -- and doubles the fan-out cost. The prefix is the existing
isolation convention; this makes search honour it.

Exclusion applies ONLY to unscoped search. Naming the corpus explicitly is an
intentional act and must still reach it, or the eval harness can't query its
own build.
"""
from __future__ import annotations

import pytest


def _index(svc, make_doc, name, corpus, text="alpha beta gamma delta. epsilon zeta eta theta."):
    p = make_doc(name, text)
    svc.index(str(p), corpus)
    return str(p)


def test_unscoped_search_skips_eval_corpora(svc, make_doc) -> None:
    _index(svc, make_doc, "live.md", "chats")
    _index(svc, make_doc, "copy.md", "eval-chats-base")

    hits = svc.search("alpha beta gamma", top_k=20)
    assert hits, "sanity: the live corpus must still match"
    assert {h.corpus for h in hits} == {"chats"}


def test_explicit_corpus_still_reaches_an_eval_corpus(svc, make_doc) -> None:
    _index(svc, make_doc, "copy.md", "eval-chats-base")
    hits = svc.search("alpha beta gamma", corpus_name="eval-chats-base", top_k=20)
    assert hits, "naming an eval corpus explicitly must still search it"
    assert {h.corpus for h in hits} == {"eval-chats-base"}


def test_exclusion_patterns_come_from_config(svc, make_doc, monkeypatch) -> None:
    """The list is config, not a hardcoded prefix."""
    _index(svc, make_doc, "live.md", "chats")
    _index(svc, make_doc, "scratch.md", "scratch-pad")

    # default config does not exclude scratch-*
    assert "scratch-pad" in {h.corpus for h in svc.search("alpha beta", top_k=20)}

    monkeypatch.setattr(svc.settings, "cross_corpus_exclude", ["scratch-*"])
    corpora = {h.corpus for h in svc.search("alpha beta", top_k=20)}
    assert "scratch-pad" not in corpora
    assert "chats" in corpora


def test_empty_exclusion_list_searches_everything(svc, make_doc, monkeypatch) -> None:
    _index(svc, make_doc, "live.md", "chats")
    _index(svc, make_doc, "copy.md", "eval-chats-base")
    monkeypatch.setattr(svc.settings, "cross_corpus_exclude", [])
    assert {h.corpus for h in svc.search("alpha beta", top_k=20)} == {"chats", "eval-chats-base"}


def test_exclusion_never_empties_an_explicit_single_corpus_search(svc, make_doc) -> None:
    """Guard against the exclusion accidentally applying to the scoped path."""
    _index(svc, make_doc, "copy.md", "eval-run-2")
    assert svc.search("alpha", corpus_name="eval-run-2", top_k=5)


def test_parallel_fanout_matches_serial_results(svc, make_doc, monkeypatch) -> None:
    """Concurrency must not change what comes back, only how fast."""
    for i, c in enumerate(("notes", "code", "docs", "wiki")):
        _index(svc, make_doc, f"d{i}.md", c,
               f"alpha beta gamma corpus {c}. delta epsilon zeta {c} tail.")

    parallel = svc.search("alpha beta gamma", top_k=20)

    monkeypatch.setattr(svc.settings, "search_fanout_workers", 1)
    serial = svc.search("alpha beta gamma", top_k=20)

    key = lambda rs: sorted((r.corpus, r.source_id, r.chunk_start) for r in rs)
    assert key(parallel) == key(serial)
    assert len(parallel) == len(serial)
