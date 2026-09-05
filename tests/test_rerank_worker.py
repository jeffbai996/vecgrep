"""Process isolation bounds the reranker's retained Torch heap.

The worker owns CUDA and receives only query/text primitives.  The API process
keeps result objects and ranking logic, so recycling a worker cannot change
search semantics or require the service itself to restart.
"""
from __future__ import annotations

import math
import time

import pytest

from vecgrep.backend import rerank as rr


class _Connection:
    def __init__(self, incoming):
        self.incoming = iter(incoming)
        self.sent = []
        self.closed = False

    def recv(self):
        return next(self.incoming)

    def send(self, value):
        self.sent.append(value)

    def close(self):
        self.closed = True


class _Model:
    def __init__(self):
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return [-1.0, 2.0]


class _Process:
    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        pass

    def terminate(self):
        self.alive = False


class _TimeoutConnection:
    def __init__(self):
        self.sent = []
        self.closed = False

    def send(self, value):
        self.sent.append(value)

    def poll(self, timeout=None):
        return False

    def close(self):
        self.closed = True


def test_worker_scores_primitive_texts_and_releases_cuda(monkeypatch):
    conn = _Connection([
        ("predict", "needle", ["first", "second"]),
        ("stop",),
    ])
    model = _Model()
    released = []
    monkeypatch.setattr(rr, "_construct_model", lambda _name: model)
    monkeypatch.setattr(rr, "_release_cuda_cache", lambda: released.append(True))
    monkeypatch.setattr(rr, "_process_committed_bytes", lambda: 123)

    rr._worker_main(conn, "model", batch_size=8, max_jobs=10, max_bytes=1000)

    assert conn.sent == [
        ("ready",),
        ("result", [-1.0, 2.0], 123, False),
    ]
    assert model.calls == [
        ([('needle', 'first'), ('needle', 'second')], {"batch_size": 8})
    ]
    assert released == [True]
    assert conn.closed is True


def test_worker_recycles_only_after_returning_the_completed_result(monkeypatch):
    conn = _Connection([("predict", "q", ["a", "b"])])
    monkeypatch.setattr(rr, "_construct_model", lambda _name: _Model())
    monkeypatch.setattr(rr, "_release_cuda_cache", lambda: None)
    monkeypatch.setattr(rr, "_process_committed_bytes", lambda: 3000)

    rr._worker_main(conn, "model", batch_size=8, max_jobs=50, max_bytes=2000)

    assert conn.sent[-1] == ("result", [-1.0, 2.0], 3000, True)
    assert conn.closed is True


def test_isolated_rerank_keeps_payloads_in_parent_and_preserves_scoring(monkeypatch):
    payloads = [{"id": 1}, {"id": 2}]
    captured = {}

    def predict(query, texts, model_name):
        captured.update(query=query, texts=texts, model_name=model_name)
        return [-1.0, 2.0]

    monkeypatch.setattr(rr, "RERANK_WORKER_ENABLED", True)
    monkeypatch.setattr(rr, "_worker_predict", predict)
    result = rr.rerank(
        "needle",
        [("first", payloads[0]), ("second", payloads[1])],
        model_name="model",
    )

    assert captured == {
        "query": "needle",
        "texts": ["first", "second"],
        "model_name": "model",
    }
    assert [payload["id"] for _, payload in result] == [2, 1]
    assert result[0][0] == 1 / (1 + math.exp(-2.0))


def test_committed_memory_counts_resident_and_swapped_pages(tmp_path):
    smaps = tmp_path / "smaps_rollup"
    smaps.write_text(
        "Rss:                1200 kB\n"
        "Pss:                1000 kB\n"
        "Swap:                300 kB\n",
        encoding="ascii",
    )

    assert rr._process_committed_bytes(smaps) == 1500 * 1024


def test_prediction_timeout_opens_circuit_before_retry(monkeypatch):
    monkeypatch.setattr(rr, "RERANK_WORKER_FAILURE_COOLDOWN_S", 60.0)
    monkeypatch.setattr(rr, "_worker_pressure_active", lambda: False)
    worker = rr._WorkerClient("model")
    worker._process = _Process()
    worker._connection = _TimeoutConnection()
    worker.ready.set()
    worker.attempt_done.set()
    started = []
    monkeypatch.setattr(worker, "_start", lambda: started.append(True))

    with pytest.raises(rr.RerankerError, match="prediction timed out"):
        worker.predict("q", ["candidate"])

    assert worker._failed_at is not None
    assert worker.is_ready() is False
    worker.ensure_started()
    time.sleep(0.02)
    assert started == []

    worker._failed_at -= 61.0
    worker.ensure_started()
    time.sleep(0.05)
    assert started == [True]


@pytest.mark.parametrize("exit_on", ["join", "terminate", "kill", "never"])
@pytest.mark.parametrize("force", [False, True])
def test_retirement_keeps_ownership_until_child_exits(monkeypatch, exit_on, force):
    worker = rr._WorkerClient("model")
    events = []

    class Child:
        alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout=None):
            assert timeout == 2
            events.append("join")
            # A concurrent request must not replace a child during retirement.
            worker.ensure_started()
            if exit_on == "join":
                self.alive = False

        def terminate(self):
            events.append("terminate")
            if exit_on == "terminate":
                self.alive = False

        def kill(self):
            events.append("kill")
            if exit_on == "kill":
                self.alive = False

    child = Child()
    worker._process = child
    worker._connection = _Connection([])
    worker.ready.set()
    started = []
    monkeypatch.setattr(worker, "_start", lambda: started.append(True))
    monkeypatch.setattr(rr, "_worker_pressure_active", lambda: False)
    worker._retire(terminate=force)

    assert started == []
    assert not worker.ready.is_set()
    if exit_on == "never":
        assert worker._process is child
        worker.ensure_started()
        assert started == []
    else:
        assert worker._process is None
        assert not child.is_alive()
    assert ("terminate" in events) == (force or exit_on != "join")
    assert ("kill" in events) == (exit_on in {"kill", "never"})
