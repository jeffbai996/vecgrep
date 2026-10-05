"""Long candidates must not multiply the padding cost of short candidates."""
from __future__ import annotations

import pytest

from vecgrep.backend import rerank as rr


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
