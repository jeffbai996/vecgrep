"""A corpus nobody can get into must not take the whole search down with it.

Corpus names here are deliberately generic -- this repo is public.

Admission used to be an unbounded wait, so one reader stalled on disk plus one
queued writer left that corpus unsearchable until the process was restarted:
no error, no log line, and every unscoped search hung with it because the
fan-out waits on each corpus future in turn.
"""
from __future__ import annotations

import threading
import time

import pytest

from vecgrep.backend.mutation import LockTimeout


@pytest.fixture
def two_corpora(svc, make_doc):
    svc.index(str(make_doc("a.md", "alpha bravo charlie delta echo")), "corpus-a")
    svc.index(str(make_doc("b.md", "alpha bravo foxtrot golf hotel")), "corpus-b")
    return svc


@pytest.fixture
def blocked_corpus_b(two_corpora):
    """Hold corpus-b's write lock for the duration of the test."""
    svc = two_corpora
    holding, release = threading.Event(), threading.Event()

    def writer():
        with svc.locks.write("corpus-b"):
            holding.set()
            release.wait(30)

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    assert holding.wait(5)
    try:
        yield svc
    finally:
        release.set()
        t.join(5)


def test_unscoped_search_answers_from_the_corpora_it_can_reach(
    blocked_corpus_b, monkeypatch
):
    svc = blocked_corpus_b
    monkeypatch.setattr(svc.settings, "search_lock_timeout_s", 0.3)

    started = time.monotonic()
    outcome = svc.search_with_diagnostics("alpha bravo", top_k=5)
    elapsed = time.monotonic() - started

    assert elapsed < 10, "unscoped search waited on the blocked corpus"
    assert outcome.results, "the reachable corpus returned nothing"
    assert {hit.corpus for hit in outcome.results} == {"corpus-a"}
    assert [w.corpus for w in outcome.warnings] == ["corpus-b"]


def test_named_corpus_search_reports_instead_of_hanging(
    blocked_corpus_b, monkeypatch
):
    svc = blocked_corpus_b
    monkeypatch.setattr(svc.settings, "search_lock_timeout_s", 0.3)

    started = time.monotonic()
    with pytest.raises(LockTimeout):
        svc.search("alpha bravo", corpus_name="corpus-b", top_k=5)
    assert time.monotonic() - started < 10


def test_an_unblocked_corpus_is_untouched_by_the_timeout(two_corpora, monkeypatch):
    svc = two_corpora
    monkeypatch.setattr(svc.settings, "search_lock_timeout_s", 0.3)
    outcome = svc.search_with_diagnostics("alpha bravo", top_k=5)
    assert outcome.warnings == []
    assert {hit.corpus for hit in outcome.results} == {"corpus-a", "corpus-b"}
