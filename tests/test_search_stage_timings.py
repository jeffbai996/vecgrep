"""A search reports where its time went.

On 2026-10-09 the only way to learn that the cross-encoder was 86% of a search
was attaching py-spy to the server. The response now carries per-stage
milliseconds, and squad-store records them so latency drift has a history.

Retrieval runs one task per corpus in parallel, so embed/vector/bm25 are SUMS
over corpora and can exceed the wall-clock `retrieve`; `retrieve`, `rerank`
and `total` are wall-clock.
"""
from __future__ import annotations

import logging

STAGES = {"embed", "vector", "bm25", "retrieve", "total"}


def _index(svc, make_doc, name, corpus):
    svc.index(str(make_doc(name, "alpha beta gamma delta. epsilon zeta eta theta.")), corpus)


def test_a_hybrid_search_reports_every_retrieval_stage(svc, make_doc) -> None:
    _index(svc, make_doc, "a.md", "notes")
    outcome = svc.search_with_diagnostics("alpha beta", corpus_name="notes")
    t = outcome.timings_ms
    assert STAGES <= set(t)
    assert all(v >= 0 for v in t.values())
    assert t["total"] >= t["retrieve"]


def test_stage_times_sum_across_corpora_in_an_unscoped_search(svc, make_doc) -> None:
    _index(svc, make_doc, "a.md", "notes")
    _index(svc, make_doc, "b.md", "chats")
    t = svc.search_with_diagnostics("alpha beta").timings_ms
    assert STAGES <= set(t)
    assert t["corpora"] == 2


def test_rerank_time_is_reported_only_when_the_reranker_ran(svc, make_doc, monkeypatch) -> None:
    _index(svc, make_doc, "a.md", "notes")
    assert "rerank" not in svc.search_with_diagnostics("alpha", corpus_name="notes").timings_ms

    monkeypatch.setattr(svc, "_rerank_ready", lambda model=None: True)
    monkeypatch.setattr(
        "vecgrep.backend.rerank.rerank",
        lambda q, pairs, model: [(0.9, p[1]) for p in pairs],
    )
    t = svc.search_with_diagnostics("alpha", corpus_name="notes", rerank=True).timings_ms
    assert t["rerank"] >= 0


def test_the_rest_search_response_carries_the_timings(svc, make_doc, monkeypatch) -> None:
    from vecgrep.backend.api import routes
    from vecgrep.backend.api.schemas import SearchRequest

    _index(svc, make_doc, "a.md", "notes")
    monkeypatch.setattr(routes, "_service", lambda: svc)
    body = routes.search(SearchRequest(query="alpha", corpus="notes"))
    assert STAGES <= set(body.timings_ms)


def test_per_request_http_client_logging_is_quiet() -> None:
    """httpx logs every qdrant call at INFO: ~18k journal lines per 6 h."""
    from vecgrep.backend import main  # noqa: F401 -- importing applies the level
    assert logging.getLogger("httpx").level >= logging.WARNING
