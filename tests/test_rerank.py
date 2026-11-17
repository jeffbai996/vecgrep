from __future__ import annotations

import importlib
import os
import threading
import time
import types
from typing import Any

import pytest

from vecgrep.backend import rerank
from vecgrep.backend import rerank as R
from vecgrep.backend import rerank as rr
from vecgrep.backend import service as svc_mod
from vecgrep.backend.service import SearchResult

# ── test_rerank_failure_degrades ──

def test_a_rerank_that_raises_still_returns_results(svc, make_doc, monkeypatch):
    svc.index(str(make_doc("d.md", "Alpha topic. Beta topic. Gamma topic.")), "c")
    monkeypatch.setattr(svc, "_rerank_ready", lambda name: True)

    def _boom(*a, **kw):
        raise RuntimeError("CUDA error: unknown error")

    monkeypatch.setattr(svc, "_apply_rerank", _boom)
    hits = svc.search("alpha", "c", top_k=5, rerank=True)
    assert hits, "a broken reranker must not empty the result set"


def test_results_are_not_marked_reranked_when_it_failed(svc, make_doc, monkeypatch):
    svc.index(str(make_doc("d.md", "Alpha topic. Beta topic. Gamma topic.")), "c")
    monkeypatch.setattr(svc, "_rerank_ready", lambda name: True)
    monkeypatch.setattr(svc, "_apply_rerank",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    hits = svc.search("alpha", "c", top_k=5, rerank=True)
    assert all("rerank" not in h.matched_by for h in hits)


def test_a_working_reranker_is_untouched(svc, make_doc, monkeypatch):
    svc.index(str(make_doc("d.md", "Alpha topic. Beta topic. Gamma topic.")), "c")
    monkeypatch.setattr(svc, "_rerank_ready", lambda name: True)
    seen = {}

    def _ok(query, candidates, top_k, model_name, explain=False):
        seen["called"] = True
        return candidates[:top_k]

    monkeypatch.setattr(svc, "_apply_rerank", _ok)
    hits = svc.search("alpha", "c", top_k=5, rerank=True)
    assert seen.get("called") and hits

# ── test_rerank_max_length ──

def test_the_cap_is_off_by_default():
    assert rerank.RERANK_MAX_LENGTH == 0 or os.environ.get("VECGREP_RERANK_MAX_LENGTH")


def test_no_cap_means_the_model_keeps_its_own_ceiling(monkeypatch):
    seen = {}

    class FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            seen.update(name=name, kwargs=kwargs)

    monkeypatch.setattr(rerank, "RERANK_MAX_LENGTH", 0)
    _construct_with(monkeypatch, FakeCrossEncoder)
    assert "max_length" not in seen["kwargs"]


def test_a_cap_is_passed_through(monkeypatch):
    seen = {}

    class FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            seen.update(name=name, kwargs=kwargs)

    monkeypatch.setattr(rerank, "RERANK_MAX_LENGTH", 512)
    _construct_with(monkeypatch, FakeCrossEncoder)
    assert seen["kwargs"]["max_length"] == 512


def test_the_fp16_fast_path_survives_the_cap(monkeypatch):
    """The cap and the FP16 load used to share one kwargs dict; setting either
    must not drop the other."""
    seen = {}

    class FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            seen.update(kwargs=kwargs)

    monkeypatch.setattr(rerank, "RERANK_MAX_LENGTH", 512)
    _construct_with(monkeypatch, FakeCrossEncoder, cuda=True)
    assert seen["kwargs"]["max_length"] == 512
    assert "model_kwargs" in seen["kwargs"], seen["kwargs"]


def _construct_with(monkeypatch, cls, *, cuda: bool = False):
    """Run _construct_model against a stub CrossEncoder and a stub torch."""
    import sys
    import types

    st: Any = types.ModuleType("sentence_transformers")
    st.CrossEncoder = cls
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)

    torch: Any = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda,
                                       empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return rerank._construct_model("fake-model")

# ── test_rerank_prior ──

DAY = 86_400.0


def _hit(cid: str, corpus: str, ts: float | None = None) -> SearchResult:
    return SearchResult(
        score=0.5, similarity_pct=50.0, chunk=f"text {cid}", chunk_start=0,
        chunk_end=6, context_before="", context_after="", source_id=f"/{corpus}/{cid}.md",
        corpus=corpus, metadata={}, chunk_id=cid, matched_by=["vector"], doc_timestamp=ts,
    )


@pytest.fixture
def corpora(svc, make_doc):
    for name in ("notes", "chats"):
        svc.index(str(make_doc(f"{name}.md", "alpha beta gamma.")), name)
    svc.set_rank_weight("notes", 1.25)
    svc.set_decay("chats", 45.0)
    return svc


def _scores(by_id: dict[str, float]):
    """A fake cross-encoder that returns raw scores, sorted like the real one."""
    def fake(query, pairs, model):
        scored = [(by_id[p[1].chunk_id], p[1]) for p in pairs]
        return sorted(scored, key=lambda s: s[0], reverse=True)
    return fake


def _order(svc, hits, by_id, monkeypatch):
    monkeypatch.setattr("vecgrep.backend.rerank.rerank", _scores(by_id))
    return [r.chunk_id for r in svc._apply_rerank("q", hits, 10, None)]


def test_the_prior_is_off_by_default(corpora, monkeypatch):
    assert corpora.settings.rerank_prior is False
    hits = [_hit("chat", "chats", time.time()), _hit("note", "notes")]
    assert _order(corpora, hits, {"chat": 0.60, "note": 0.58}, monkeypatch) == ["chat", "note"]


def test_a_weighted_corpus_wins_a_close_call(corpora, monkeypatch):
    monkeypatch.setattr(corpora.settings, "rerank_prior", True)
    hits = [_hit("chat", "chats", time.time()), _hit("note", "notes")]
    assert _order(corpora, hits, {"chat": 0.60, "note": 0.58}, monkeypatch) == ["note", "chat"]


def test_the_weight_does_not_overturn_a_clear_reranker_verdict(corpora, monkeypatch):
    monkeypatch.setattr(corpora.settings, "rerank_prior", True)
    hits = [_hit("chat", "chats", time.time()), _hit("note", "notes")]
    assert _order(corpora, hits, {"chat": 0.80, "note": 0.40}, monkeypatch) == ["chat", "note"]


def test_recency_decay_is_not_applied_after_the_reranker(corpora, monkeypatch):
    """Decay picks the pool; after the reranker it buried every older day."""
    monkeypatch.setattr(corpora.settings, "rerank_prior", True)
    now = time.time()
    hits = [_hit("old", "chats", now - 120 * DAY), _hit("new", "chats", now - 1 * DAY)]
    assert _order(corpora, hits, {"old": 0.60, "new": 0.59}, monkeypatch) == ["old", "new"]


def test_the_displayed_percent_stays_the_reranker_value(corpora, monkeypatch):
    monkeypatch.setattr(corpora.settings, "rerank_prior", True)
    monkeypatch.setattr("vecgrep.backend.rerank.rerank", _scores({"note": 0.58}))
    (r,) = corpora._apply_rerank("q", [_hit("note", "notes")], 10, None)
    assert r.similarity_pct == pytest.approx(svc_mod._rerank_to_pct(0.58))
    assert r.explain["rerank_prior"] == pytest.approx(1.25)


def test_rerank_prior_is_an_editable_setting():
    from vecgrep.backend import config
    assert "rerank_prior" in config.EDITABLE_FIELDS

# ── test_rerank_length_batches ──

class RecordingModel:
    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, **kwargs):
        self.calls.append(list(pairs))
        return [float(len(text)) for _, text in pairs]


def test_long_candidate_is_scored_separately_without_truncation():
    model = RecordingModel()
    texts = ["short", "x" * 26000, "another short", "中文候选内容"]
    scores = rr._predict_length_grouped(model, "query", texts, batch_size=16)

    assert scores == [float(len(text)) for text in texts]
    assert sorted(text for batch in model.calls for _, text in batch) == sorted(texts)
    assert next(batch for batch in model.calls if any(len(t) == 26000 for _, t in batch)) == [
        ("query", texts[1])
    ]


def test_mixed_lengths_keep_batches_bounded_and_scores_in_input_order():
    model = RecordingModel()
    texts = ["x" * length for length in [10, 400, 20, 200, 30, 800, 15, 300]]
    assert rr._predict_length_grouped(model, "q", texts, batch_size=3) == [
        float(len(text)) for text in texts
    ]
    for batch in model.calls:
        lengths = [len(query) + len(text) for query, text in batch]
        assert len(batch) <= 3
        assert max(lengths) <= 2 * min(lengths)


def test_similar_lengths_keep_the_existing_single_predict_call():
    model = RecordingModel()
    texts = ["short", "second", "third"]
    rr._predict_length_grouped(model, "q", texts, batch_size=2)
    assert model.calls == [[("q", text) for text in texts]]


def test_query_length_is_included_in_the_padding_estimate():
    model = RecordingModel()
    texts = ["x", "x" * 100]
    rr._predict_length_grouped(model, "q" * 200, texts, batch_size=8)
    assert len(model.calls) == 1


def test_empty_input_does_not_call_the_model():
    model = RecordingModel()
    assert rr._predict_length_grouped(model, "q", [], batch_size=8) == []
    assert model.calls == []


def test_direct_rerank_uses_grouping_and_preserves_payload_ranking(monkeypatch):
    model = RecordingModel()
    monkeypatch.setattr(rr, "RERANK_WORKER_ENABLED", False)
    monkeypatch.setattr(rr, "_load", lambda _: model)
    monkeypatch.setattr(rr, "_release_cuda_cache", lambda: None)
    result = rr.rerank("q", [("short", {"id": 1}), ("x" * 26000, {"id": 2})])
    assert len(model.calls) == 2
    assert [payload["id"] for _, payload in result] == [2, 1]


def test_prediction_error_is_propagated(monkeypatch):
    class BrokenModel:
        def predict(self, pairs, **kwargs):
            raise RuntimeError("prediction failed")

    with pytest.raises(RuntimeError, match="prediction failed"):
        rr._predict_length_grouped(BrokenModel(), "q", ["short", "x" * 26000], batch_size=8)


def test_worker_uses_grouping_and_returns_scores_in_request_order(monkeypatch):
    class Connection:
        def __init__(self):
            self.requests = iter([("predict", "q", ["short", "x" * 26000]), ("stop",)])
            self.sent = []

        def recv(self):
            return next(self.requests)

        def send(self, value):
            self.sent.append(value)

        def close(self):
            pass

    model, connection = RecordingModel(), Connection()
    monkeypatch.setattr(rr, "_construct_model", lambda _: model)
    monkeypatch.setattr(rr, "_release_cuda_cache", lambda: None)
    monkeypatch.setattr(rr, "_process_committed_bytes", lambda: 100)
    rr._worker_main(connection, "model", batch_size=8, max_jobs=64, max_bytes=1000)
    assert len(model.calls) == 2
    assert connection.sent == [("ready",), ("result", [5.0, 26000.0], 100, False)]


def test_incomplete_batch_cannot_assign_a_default_score_to_a_candidate():
    class IncompleteModel:
        def predict(self, pairs, **kwargs):
            return []

    with pytest.raises(rr.RerankerError, match="incomplete score batch"):
        rr._predict_length_grouped(IncompleteModel(), "q", ["short", "x" * 26000], batch_size=8)

# ── test_rerank_memory_bounds ──

class _FakeModel:
    """Records how predict() was called, returns one logit per pair."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict(self, pairs, **kwargs):
        self.calls.append({"n": len(pairs), **kwargs})
        return [0.5] * len(pairs)


@pytest.fixture
def fake_model(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(rr, "_load", lambda name: model)
    return model


def _pairs(n: int):
    return [(f"chunk {i}", {"id": i}) for i in range(n)]


def test_predict_gets_an_explicit_bounded_batch_size(fake_model):
    """The library default is 32. A padded batch of 32 long transcript chunks
    is what produced the 12 GB spike, so the batch size must be ours, not the
    library's, and it must be small."""
    rr.rerank("q", _pairs(50))
    assert fake_model.calls, "predict was never called"
    batch = fake_model.calls[0].get("batch_size")
    assert batch is not None, "predict() called without an explicit batch_size"
    assert batch <= 16


def test_batch_size_is_env_tunable(monkeypatch, fake_model):
    """A box with more VRAM than the 3090 should be able to raise this without
    a code change."""
    monkeypatch.setenv("VECGREP_RERANK_BATCH", "4")
    importlib.reload(rr)
    try:
        monkeypatch.setattr(rr, "_load", lambda name: fake_model)
        rr.rerank("q", _pairs(10))
        assert fake_model.calls[-1]["batch_size"] == 4
    finally:
        monkeypatch.delenv("VECGREP_RERANK_BATCH", raising=False)
        importlib.reload(rr)


def test_cuda_reserve_is_released_after_scoring(monkeypatch, fake_model):
    """torch reserves the peak and never returns it on its own. Without this
    the high-water mark of one unlucky search is charged to the card until the
    process restarts."""
    released = []
    monkeypatch.setattr(rr, "_release_cuda_cache", lambda: released.append(True))
    rr.rerank("q", _pairs(5))
    assert released == [True]


def test_cuda_reserve_is_released_even_when_scoring_raises(monkeypatch):
    """A rerank that dies mid-batch is exactly when the card is most full."""
    class _Boom:
        def predict(self, pairs, **kwargs):
            raise RuntimeError("CUDA error: unknown error")

    monkeypatch.setattr(rr, "_load", lambda name: _Boom())
    released = []
    monkeypatch.setattr(rr, "_release_cuda_cache", lambda: released.append(True))
    with pytest.raises(RuntimeError):
        rr.rerank("q", _pairs(5))
    assert released == [True]


def test_release_is_a_no_op_without_torch(monkeypatch):
    """Reranking is an optional extra. On an install without torch, or on a
    CPU-only box, releasing must not raise."""
    import builtins

    real_import = builtins.__import__

    def _no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_torch)
    rr._release_cuda_cache()  # must not raise


def test_empty_candidates_skip_the_model_entirely(fake_model):
    rr.rerank("q", [])
    assert fake_model.calls == []

# ── test_rerank_pool_cap ──

def _result(i: int, pct: float) -> SearchResult:
    return SearchResult(
        score=pct / 100.0, similarity_pct=pct, chunk=f"chunk text {i}",
        chunk_start=0, chunk_end=len(f"chunk text {i}"),
        context_before="", context_after="", source_id=f"s{i}", corpus="c",
        metadata={}, chunk_id=f"c{i}", matched_by=["vector"],
    )


@pytest.fixture
def candidates():
    # 120 candidates, fusion score DESCENDING with index, deliberately shuffled
    # into the arrival order a fan-out produces (corpus by corpus, not global).
    pool = [_result(i, 100.0 - i) for i in range(120)]
    return pool[60:] + pool[:60]


def test_only_the_top_pool_by_fusion_score_reaches_the_cross_encoder(
        svc, candidates, monkeypatch):
    seen: list[list[str]] = []

    def fake_rerank(query, pairs, model):
        seen.append([p[1].chunk_id for p in pairs])
        # score in reverse of arrival, so rerank order is observably not fusion
        return [(float(i), p[1]) for i, p in enumerate(pairs)]

    monkeypatch.setattr(svc_mod, "RERANK_POOL_MAX", 50)
    monkeypatch.setattr("vecgrep.backend.rerank.rerank", fake_rerank)
    svc._apply_rerank("q", list(candidates), 10, None)

    assert len(seen) == 1, "the cross-encoder must run exactly once"
    scored = seen[0]
    assert len(scored) == 50
    # the 50 highest fusion scores are c0..c49, whatever order they arrived in
    assert set(scored) == {f"c{i}" for i in range(50)}


def test_the_unscored_tail_is_still_returned_below_the_reranked_head(
        svc, candidates, monkeypatch):
    def fake_rerank(query, pairs, model):
        return [(float(i), p[1]) for i, p in enumerate(pairs)]

    monkeypatch.setattr(svc_mod, "RERANK_POOL_MAX", 50)
    monkeypatch.setattr("vecgrep.backend.rerank.rerank", fake_rerank)
    out = svc._apply_rerank("q", list(candidates), 80, None)

    ids = [r.chunk_id for r in out]
    # A latency cap must not shrink the result set the caller asked for.
    assert len(ids) == 80
    head, tail = ids[:50], ids[50:]
    assert set(head) <= {f"c{i}" for i in range(50)}, "head is the reranked pool"
    assert all(t not in head for t in tail)
    # the tail keeps fusion order: c50, c51, ... (descending fusion score)
    assert tail == [f"c{50 + i}" for i in range(len(tail))]


def test_the_tail_is_not_marked_reranked_and_keeps_its_fusion_score(
        svc, candidates, monkeypatch):
    def fake_rerank(query, pairs, model):
        return [(0.9, p[1]) for p in pairs]

    monkeypatch.setattr(svc_mod, "RERANK_POOL_MAX", 50)
    monkeypatch.setattr("vecgrep.backend.rerank.rerank", fake_rerank)
    out = svc._apply_rerank("q", list(candidates), 80, None)

    by_id = {r.chunk_id: r for r in out}
    tail = by_id["c60"]
    assert "rerank" not in tail.matched_by, \
        "a chunk the model never scored must not claim it was reranked"
    assert tail.similarity_pct == pytest.approx(40.0), \
        "unscored tail keeps its fusion score, not a rerank-calibrated one"
    assert "rerank_score" not in (tail.explain or {})
    head = by_id["c0"]
    assert "rerank" in head.matched_by


def test_a_pool_smaller_than_the_cap_is_reranked_whole(svc, monkeypatch):
    small = [_result(i, 100.0 - i) for i in range(12)]
    seen: list[int] = []

    def fake_rerank(query, pairs, model):
        seen.append(len(pairs))
        return [(float(len(pairs) - i), p[1]) for i, p in enumerate(pairs)]

    monkeypatch.setattr(svc_mod, "RERANK_POOL_MAX", 50)
    monkeypatch.setattr("vecgrep.backend.rerank.rerank", fake_rerank)
    out = svc._apply_rerank("q", small, 10, None)
    assert seen == [12], "no truncation when the pool already fits"
    assert all("rerank" in r.matched_by for r in out)


def test_cap_is_env_tunable_and_defaults_to_the_candidate_pool(svc):
    assert svc_mod.RERANK_POOL_MAX == svc_mod.CANDIDATE_POOL == 50


def test_a_zero_or_negative_cap_disables_truncation(svc, candidates, monkeypatch):
    """An operator turning the cap off must get the old whole-pool behaviour,
    not an empty rerank."""
    seen: list[int] = []

    def fake_rerank(query, pairs, model):
        seen.append(len(pairs))
        return [(float(i), p[1]) for i, p in enumerate(pairs)]

    monkeypatch.setattr(svc_mod, "RERANK_POOL_MAX", 0)
    monkeypatch.setattr("vecgrep.backend.rerank.rerank", fake_rerank)
    svc._apply_rerank("q", list(candidates), 10, None)
    assert seen == [120]

# ── test_rerank_score_cache ──

def _candidates(n: int) -> list[tuple[str, dict]]:
    return [(f"chunk text {i}", {"id": i}) for i in range(n)]


def test_a_repeat_query_scores_nothing_again(monkeypatch):
    seen: list[list[str]] = []

    def fake_predict(query, texts, model_name):
        seen.append(list(texts))
        return [float(len(t)) for t in texts]

    monkeypatch.setattr(rerank, "RERANK_WORKER_ENABLED", True)
    monkeypatch.setattr(rerank, "_worker_predict", fake_predict)

    first = rerank.rerank("q", _candidates(4))
    second = rerank.rerank("q", _candidates(4))

    assert len(seen) == 1, "the model was asked to score the same pairs twice"
    assert seen[0] == [f"chunk text {i}" for i in range(4)]
    assert first == second, "a cached rerank must return the same ranking"


def test_only_the_misses_are_sent(monkeypatch):
    sent: list[list[str]] = []

    def fake_predict(query, texts, model_name):
        sent.append(list(texts))
        return [1.0] * len(texts)

    monkeypatch.setattr(rerank, "RERANK_WORKER_ENABLED", True)
    monkeypatch.setattr(rerank, "_worker_predict", fake_predict)

    rerank.rerank("q", _candidates(3))
    rerank.rerank("q", _candidates(5))          # 3 known, 2 new

    assert sent[1] == ["chunk text 3", "chunk text 4"], sent[1]


def test_scores_are_placed_back_on_the_right_candidate(monkeypatch):
    """A partial hit reassembles a mixed list; putting a score on the wrong
    payload would be silent and wrong."""
    def fake_predict(query, texts, model_name):
        return [float(t.rsplit(" ", 1)[1]) for t in texts]

    monkeypatch.setattr(rerank, "RERANK_WORKER_ENABLED", True)
    monkeypatch.setattr(rerank, "_worker_predict", fake_predict)

    warm = rerank.rerank("q", _candidates(6))
    mixed = rerank.rerank("q", _candidates(9))
    by_id = {p["id"]: s for s, p in mixed}
    assert [by_id[i] for i in range(9)] == sorted(by_id[i] for i in range(9))
    for score, payload in warm:
        assert by_id[payload["id"]] == pytest.approx(score)


def test_a_different_query_is_a_different_key(monkeypatch):
    sent: list[list[str]] = []
    monkeypatch.setattr(rerank, "RERANK_WORKER_ENABLED", True)
    monkeypatch.setattr(rerank, "_worker_predict",
                        lambda q, t, m: sent.append(list(t)) or [1.0] * len(t))
    rerank.rerank("one", _candidates(2))
    rerank.rerank("two", _candidates(2))
    assert len(sent) == 2, "two different questions shared an answer"


def test_a_different_model_is_a_different_key(monkeypatch):
    sent: list[list[str]] = []
    monkeypatch.setattr(rerank, "RERANK_WORKER_ENABLED", True)
    monkeypatch.setattr(rerank, "_worker_predict",
                        lambda q, t, m: sent.append(list(t)) or [1.0] * len(t))
    rerank.rerank("q", _candidates(2), model_name="model-a")
    rerank.rerank("q", _candidates(2), model_name="model-b")
    assert len(sent) == 2


def test_the_cache_is_bounded_and_evicts_oldest_first(monkeypatch):
    monkeypatch.setattr(rerank, "RERANK_SCORE_CACHE_SIZE", 4)
    rerank.clear_score_cache()
    calls: list[list[str]] = []
    monkeypatch.setattr(rerank, "RERANK_WORKER_ENABLED", True)
    monkeypatch.setattr(rerank, "_worker_predict",
                        lambda q, t, m: calls.append(list(t)) or [1.0] * len(t))

    rerank.rerank("q", _candidates(4))            # fills it
    rerank.rerank("q", [("chunk text 99", {})])   # evicts chunk text 0
    rerank.rerank("q", [("chunk text 0", {})])    # must be a miss again

    assert rerank.score_cache_size() <= 4
    assert calls[-1] == ["chunk text 0"]


def test_a_zero_size_cache_is_off(monkeypatch):
    monkeypatch.setattr(rerank, "RERANK_SCORE_CACHE_SIZE", 0)
    rerank.clear_score_cache()
    calls: list[int] = []
    monkeypatch.setattr(rerank, "RERANK_WORKER_ENABLED", True)
    monkeypatch.setattr(rerank, "_worker_predict",
                        lambda q, t, m: calls.append(len(t)) or [1.0] * len(t))
    rerank.rerank("q", _candidates(3))
    rerank.rerank("q", _candidates(3))
    assert calls == [3, 3] and rerank.score_cache_size() == 0


def test_everything_cached_means_the_model_is_never_called(monkeypatch):
    """The hit path must not wake the worker at all — waking it is most of the
    cost this is here to avoid."""
    monkeypatch.setattr(rerank, "RERANK_WORKER_ENABLED", True)
    monkeypatch.setattr(rerank, "_worker_predict",
                        lambda q, t, m: [1.0] * len(t))
    rerank.rerank("q", _candidates(3))

    def explode(*a, **kw):
        raise AssertionError("the model was called on a full cache hit")

    monkeypatch.setattr(rerank, "_worker_predict", explode)
    monkeypatch.setattr(rerank, "_load", explode)
    assert len(rerank.rerank("q", _candidates(3))) == 3

# ── test_rerank_warm ──

@pytest.fixture(autouse=True)
def _clean_state():
    R._cache.clear()
    R._warming.clear()
    R._failed_at.clear()
    yield
    R._cache.clear()
    R._warming.clear()
    R._failed_at.clear()


def test_wait_ready_is_false_while_the_model_is_still_loading(monkeypatch):
    """The whole point: a cold model reports not-ready fast instead of
    blocking the caller for the length of the load."""
    release = threading.Event()

    def _slow(name):
        release.wait(10)
        R._cache[name] = object()
        return R._cache[name]

    monkeypatch.setattr(R, "_load", _slow)
    started = time.monotonic()
    assert R.wait_ready("m", timeout=0.2) is False
    assert time.monotonic() - started < 5, "wait_ready blocked on the load"
    release.set()


def test_wait_ready_is_true_once_warm_finishes(monkeypatch):
    monkeypatch.setattr(R, "_load", lambda name: R._cache.setdefault(name, object()))
    assert R.ensure_warm("m").wait(5)
    assert R.is_ready("m") is True
    assert R.wait_ready("m", timeout=0) is True


def test_warm_is_idempotent(monkeypatch):
    """Called per request, so a second call must not start a second load."""
    calls = []

    def _count(name):
        calls.append(name)
        time.sleep(0.05)
        R._cache[name] = object()
        return R._cache[name]

    monkeypatch.setattr(R, "_load", _count)
    events = [R.ensure_warm("m") for _ in range(5)]
    for ev in events:
        assert ev.wait(5)
    assert len(calls) == 1


def test_a_failed_load_does_not_retry_on_every_call(monkeypatch):
    """Retrying a failing download per search would be its own outage."""
    calls = []

    def _boom(name):
        calls.append(name)
        raise R.RerankerError("no extra installed")

    monkeypatch.setattr(R, "_load", _boom)
    for _ in range(5):
        assert R.wait_ready("m", timeout=1) is False
    assert len(calls) == 1


def test_a_failed_load_retries_after_the_cooldown(monkeypatch):
    """...but it must not stay off until someone restarts the process."""
    calls = []

    def _boom(name):
        calls.append(name)
        raise R.RerankerError("transient")

    monkeypatch.setattr(R, "_load", _boom)
    assert R.wait_ready("m", timeout=1) is False
    assert len(calls) == 1
    R._failed_at["m"] = time.monotonic() - (R._RETRY_AFTER_S + 1)
    assert R.wait_ready("m", timeout=1) is False
    assert len(calls) == 2


def test_rerank_still_loads_synchronously_for_direct_callers(monkeypatch):
    """The CLI and the eval harness want the model, however long it takes —
    only the SERVER needs the bounded path."""
    monkeypatch.setattr(R, "_load", lambda name: _FakeModel_rerank_warm())
    out = R.rerank("q", [("a", {"id": 1}), ("b", {"id": 2})], model_name="m")
    assert [p["id"] for _, p in out] == [2, 1]


def test_cross_encoder_loads_in_float16_when_cuda_is_available(monkeypatch):
    captured = {}
    float16 = object()

    class _FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            captured["name"] = name
            captured["kwargs"] = kwargs

    fake_torch = types.SimpleNamespace(
        float16=float16,
        cuda=types.SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "sentence_transformers",
        types.SimpleNamespace(CrossEncoder=_FakeCrossEncoder),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    model = R._load("reranker")

    assert isinstance(model, _FakeCrossEncoder)
    assert captured == {
        "name": "reranker",
        "kwargs": {"model_kwargs": {"torch_dtype": float16}},
    }


def test_cross_encoder_keeps_default_dtype_without_cuda(monkeypatch):
    captured = {}

    class _FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            captured["name"] = name
            captured["kwargs"] = kwargs

    fake_torch = types.SimpleNamespace(
        float16=object(),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "sentence_transformers",
        types.SimpleNamespace(CrossEncoder=_FakeCrossEncoder),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    R._load("reranker")

    assert captured == {"name": "reranker", "kwargs": {}}


class _FakeModel_rerank_warm:
    def predict(self, pairs, **kwargs):  # kwargs: batch_size, see rerank.RERANK_BATCH
        return [float(i) for i in range(len(pairs))]
