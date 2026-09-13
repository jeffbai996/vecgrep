"""Scoring the same (query, chunk) twice is arithmetic we already did.

A cross-encoder score is a pure function of (model, query, chunk text): same
inputs, same logit, every time. So a repeat is free to serve from memory and
the results are IDENTICAL, which is the whole point -- Jeff 2026-09-12 asked
for "fast without quality trade".

Measured before this: a scoped reranked search ran 2.83s against 0.52s
unreranked, and running the SAME query again cost the full 2.83s over.

Deliberately in memory, not sqlite like the embed cache. A score is eight
bytes and there are fifty per search; putting that on disk buys persistence
across restarts and costs SSD writes on every miss, on a box where write
volume has been a live problem all week. Bounded by entry count so it cannot
grow into a second problem to remember.

What it does NOT do: help a query nobody has asked before. The query is part
of the key because the score depends on it. There is no lossless way to make a
novel rerank cheaper -- the model has to read the text.
"""
from __future__ import annotations

import pytest

from vecgrep.backend import rerank


@pytest.fixture(autouse=True)
def _clean_cache():
    rerank.clear_score_cache()
    yield
    rerank.clear_score_cache()


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

    def explode(*a, **kw):  # noqa: ANN002
        raise AssertionError("the model was called on a full cache hit")

    monkeypatch.setattr(rerank, "_worker_predict", explode)
    monkeypatch.setattr(rerank, "_load", explode)
    assert len(rerank.rerank("q", _candidates(3))) == 3
