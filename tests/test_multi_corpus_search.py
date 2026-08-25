from __future__ import annotations

import logging

import pytest

from vecgrep.backend.store import CorpusError


def _seed(svc, make_doc) -> None:
    svc.index(str(make_doc("a.md", "alpha shared marker")), "a")
    svc.index(str(make_doc("b.md", "alpha shared marker")), "b")
    svc.index(str(make_doc("c.md", "alpha shared marker")), "c")
    svc.index(str(make_doc("eval.md", "alpha shared marker")), "eval-hidden")


def test_selected_subset_deduplicates_in_order_and_all_hides_eval(svc, make_doc) -> None:
    _seed(svc, make_doc)

    selected = svc.search(
        "alpha", corpus_names=["b", "a", "b"], top_k=20
    )
    assert {hit.corpus for hit in selected} == {"a", "b"}

    unscoped = svc.search("alpha", top_k=20)
    assert "eval-hidden" not in {hit.corpus for hit in unscoped}
    explicit_hidden = svc.search(
        "alpha", corpus_names=["eval-hidden"], top_k=20
    )
    assert {hit.corpus for hit in explicit_hidden} == {"eval-hidden"}


def test_scope_validation_happens_before_search(svc, make_doc, monkeypatch) -> None:
    _seed(svc, make_doc)
    calls = []
    monkeypatch.setattr(svc, "_search_one", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(CorpusError, match="mutually exclusive"):
        svc.search("alpha", "a", corpus_names=["b"])
    with pytest.raises(CorpusError, match="cannot be empty"):
        svc.search("alpha", corpus_names=[])
    with pytest.raises(CorpusError, match="No such corpus"):
        svc.search("alpha", corpus_names=["a", "missing"])
    assert calls == []


def test_partial_failures_warn_but_single_and_all_failed_raise(
    svc, make_doc, monkeypatch, caplog
) -> None:
    _seed(svc, make_doc)
    original = svc._search_one

    def fail_b(corpus, *args, **kwargs):
        if corpus.name == "b":
            raise RuntimeError("backend disconnected")
        return original(corpus, *args, **kwargs)

    monkeypatch.setattr(svc, "_search_one", fail_b)
    secret_query = "private query must not enter logs"
    with caplog.at_level(logging.INFO):
        outcome = svc.search_with_diagnostics(
            secret_query, corpus_names=["a", "b"], top_k=10
        )
    assert {hit.corpus for hit in outcome.results} == {"a"}
    assert [(w.corpus, w.code) for w in outcome.warnings] == [
        ("b", "search_failed")
    ]
    assert secret_query not in caplog.text

    with pytest.raises(RuntimeError):
        svc.search("alpha", "b")
    with pytest.raises(RuntimeError):
        svc.search("alpha", corpus_names=["b"])


def test_hybrid_hydrates_missing_payloads_in_one_batch(
    svc, make_doc, monkeypatch
) -> None:
    svc.index(str(make_doc("doc.md", "quartzite alpha beta gamma")), "notes")
    corpus = svc.registry.get("notes")
    raw = svc.bm25.search("notes", "quartzite", top_k=50)
    assert raw
    stripped = []
    for cid, score, payload in raw:
        payload = dict(payload)
        payload.pop("context_before", None)
        payload.pop("context_after", None)
        stripped.append((cid, score, payload))

    original_many = svc.store.get_many_by_id
    calls = []

    def get_many(collection, ids):
        calls.append(list(ids))
        return original_many(collection, ids)

    monkeypatch.setattr(svc.bm25, "search", lambda *args, **kwargs: stripped)
    monkeypatch.setattr(svc.store, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(svc.store, "get_many_by_id", get_many)
    monkeypatch.setattr(
        svc.store,
        "get_by_id",
        lambda *args, **kwargs: pytest.fail("per-hit hydration used"),
    )

    hits = svc._search_one_locked(corpus, "quartzite", 10, "hybrid")
    assert hits
    assert len(calls) == 1
    assert set(calls[0]) == {cid for cid, _, _ in stripped}
