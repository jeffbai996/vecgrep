from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import threading
import time

import pytest

from vecgrep.backend import service as service_module
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


@pytest.mark.parametrize("workers", [1, 8])
def test_multi_corpus_search_embeds_query_once_per_model(
    svc, make_doc, monkeypatch, workers
) -> None:
    _seed(svc, make_doc)
    monkeypatch.setattr(svc.settings, "search_fanout_workers", workers)
    inner = svc._backend_cache["auto"]._inner
    original = inner.embed
    calls: list[list[str]] = []

    def counted(texts):
        calls.append(list(texts))
        return original(texts)

    monkeypatch.setattr(inner, "embed", counted)
    svc.search(
        "one-off query embedding marker",
        corpus_names=["a", "b", "c"],
        mode="vector",
        top_k=20,
    )

    assert calls == [["one-off query embedding marker"]]


def test_multi_corpus_search_keeps_distinct_model_embeddings(
    svc, make_doc, monkeypatch
) -> None:
    _seed(svc, make_doc)
    corpus_b = svc.registry.get("b")
    corpus_b.embed_model = "stub-2"
    svc.registry.upsert(corpus_b)
    calls: list[str] = []

    def embed(corpus, _query):
        calls.append(corpus.embed_model)
        return [0.0] * corpus.dim

    monkeypatch.setattr(svc, "_embed_query_with_failover", embed)
    svc.search(
        "mixed model marker",
        corpus_names=["a", "b", "c"],
        mode="vector",
        top_k=20,
    )

    assert sorted(calls) == ["stub-1", "stub-2"]


def test_shared_embedding_failure_warns_for_each_affected_corpus(
    svc, make_doc, monkeypatch
) -> None:
    _seed(svc, make_doc)
    corpus_b = svc.registry.get("b")
    corpus_b.embed_model = "stub-2"
    svc.registry.upsert(corpus_b)
    calls: list[str] = []

    def embed(corpus, _query):
        calls.append(corpus.embed_model)
        if corpus.embed_model == "stub-1":
            raise RuntimeError("shared embed backend failed")
        return [0.0] * corpus.dim

    monkeypatch.setattr(svc, "_embed_query_with_failover", embed)
    outcome = svc.search_with_diagnostics(
        "shared failure marker",
        corpus_names=["a", "b", "c"],
        mode="vector",
        top_k=20,
    )

    assert calls.count("stub-1") == 1
    assert calls.count("stub-2") == 1
    assert [(w.corpus, w.code) for w in outcome.warnings] == [
        ("a", "search_failed"),
        ("c", "search_failed"),
    ]


def test_concurrent_requests_share_one_bounded_fanout_pool(
    svc, make_doc, monkeypatch
) -> None:
    _seed(svc, make_doc)
    monkeypatch.setattr(svc.settings, "search_fanout_workers", 2)
    active = 0
    peak = 0
    calls = 0
    lock = threading.Lock()
    two_active = threading.Event()
    release = threading.Event()
    request_barrier = threading.Barrier(3)

    def searched(*_args, **_kwargs):
        nonlocal active, peak, calls
        with lock:
            active += 1
            calls += 1
            peak = max(peak, active)
            if active == 2:
                two_active.set()
        try:
            if not release.wait(5):
                raise AssertionError("timed out waiting to release corpus searches")
            return []
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(svc, "_search_one", searched)

    def request() -> list:
        request_barrier.wait(timeout=2)
        return svc.search(
            "bounded fanout",
            corpus_names=["a", "b", "c"],
            mode="bm25",
        )

    with ThreadPoolExecutor(max_workers=3) as requests:
        futures = [requests.submit(request) for _ in range(3)]
        assert two_active.wait(2)
        # Give every outer request time to submit. Request-local pools would
        # reach six active corpus calls here; the shared pool must stay at two.
        time.sleep(0.05)
        assert peak == 2
        release.set()
        assert [future.result(timeout=5) for future in futures] == [[], [], []]

    assert calls == 9


def test_identical_concurrent_queries_share_one_embedding(
    svc, make_doc, monkeypatch
) -> None:
    _seed(svc, make_doc)
    corpus = svc.registry.get("a")
    secret_query = "private same-query singleflight marker"
    vector = [0.25] * corpus.dim
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    lock = threading.Lock()

    def embed(_corpus, query):
        nonlocal calls
        assert query == secret_query
        with lock:
            calls += 1
        entered.set()
        if not release.wait(5):
            raise AssertionError("timed out waiting to release query embedding")
        return vector

    monkeypatch.setattr(svc, "_embed_query_with_failover", embed)
    with ThreadPoolExecutor(max_workers=2) as requests:
        first = requests.submit(svc._embed_query_singleflight, corpus, secret_query)
        assert entered.wait(2)
        second = requests.submit(svc._embed_query_singleflight, corpus, secret_query)
        deadline = time.monotonic() + 2
        while not second.running() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert second.running()
        time.sleep(0.02)
        assert calls == 1

        runtime = svc._search_runtime()
        assert len(runtime._embedding_futures) == 1
        assert all(
            secret_query != part
            for key in runtime._embedding_futures
            for part in key
        )

        release.set()
        assert first.result(timeout=5) == vector
        assert second.result(timeout=5) == vector

    assert runtime._embedding_futures == {}


def test_singleflight_failure_is_shared_then_later_request_retries(
    svc, make_doc, monkeypatch
) -> None:
    _seed(svc, make_doc)
    corpus = svc.registry.get("a")
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def fail(_corpus, _query):
        nonlocal calls
        calls += 1
        entered.set()
        if not release.wait(5):
            raise AssertionError("timed out waiting to release failed embedding")
        raise RuntimeError("embedding backend failed")

    monkeypatch.setattr(svc, "_embed_query_with_failover", fail)
    with ThreadPoolExecutor(max_workers=2) as requests:
        first = requests.submit(svc._embed_query_singleflight, corpus, "retry marker")
        assert entered.wait(2)
        second = requests.submit(svc._embed_query_singleflight, corpus, "retry marker")
        deadline = time.monotonic() + 2
        while not second.running() and time.monotonic() < deadline:
            time.sleep(0.005)
        time.sleep(0.02)
        release.set()
        for future in (first, second):
            with pytest.raises(RuntimeError, match="embedding backend failed"):
                future.result(timeout=5)

    assert calls == 1
    assert svc._search_runtime()._embedding_futures == {}

    expected = [0.5] * corpus.dim
    monkeypatch.setattr(
        svc,
        "_embed_query_with_failover",
        lambda _corpus, _query: expected,
    )
    assert svc._embed_query_singleflight(corpus, "retry marker") == expected


def test_concurrent_cold_backend_resolution_builds_one_client(
    svc, make_doc, monkeypatch
) -> None:
    _seed(svc, make_doc)
    corpus = svc.registry.get("a")
    corpus.embed_backend = "cold-backend"
    corpus.embed_model = "cold-model"
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    class Backend:
        name = "cold-backend"
        model = "cold-model"
        dim = corpus.dim

    backend = Backend()

    def resolve(_settings, *, prefer, model):
        nonlocal calls
        assert (prefer, model) == ("cold-backend", "cold-model")
        calls += 1
        entered.set()
        if not release.wait(5):
            raise AssertionError("timed out waiting to release backend resolution")
        return backend

    monkeypatch.setattr(service_module, "get_embed_backend", resolve)
    with ThreadPoolExecutor(max_workers=2) as requests:
        first = requests.submit(svc._backend_for, corpus)
        assert entered.wait(2)
        second = requests.submit(svc._backend_for, corpus)
        deadline = time.monotonic() + 2
        while not second.running() and time.monotonic() < deadline:
            time.sleep(0.005)
        time.sleep(0.02)
        assert calls == 1
        release.set()
        first_backend = first.result(timeout=5)
        second_backend = second.result(timeout=5)
        assert first_backend is second_backend
        assert getattr(first_backend, "_inner", first_backend) is backend

    assert calls == 1
