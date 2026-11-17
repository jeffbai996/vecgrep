from __future__ import annotations

import json
import math
import time
from typing import Any, cast

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

    def start(self):
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
    worker._process = cast(Any, _Process())
    worker._connection = cast(Any, _TimeoutConnection())
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
    worker._process = cast(Any, child)
    worker._connection = cast(Any, _Connection([]))
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


def test_start_owns_child_before_unused_pipe_close_can_fail(monkeypatch):
    import multiprocessing

    worker = rr._WorkerClient("model")
    worker._starting = True
    process = _Process()
    process.start = lambda: None
    parent = _Connection([])

    class BrokenChildPipe:
        def close(self):
            raise OSError("unused pipe close failed")

    class Context:
        def Pipe(self):
            return parent, BrokenChildPipe()

        def Process(self, **kwargs):
            return process

    monkeypatch.setattr(multiprocessing, "get_context", lambda method: Context())
    worker._start()

    assert not process.is_alive()
    assert worker._process is None
    assert not worker._starting
    assert not worker.ready.is_set()
    assert worker._failed_at is not None
    assert parent.closed

# ── test_rerank_worker_idle ──

class _Process_rerank_worker_idle:
    def __init__(self):
        self.alive = True
        self.terminated = False
        self.pid = 4242

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.alive = False

    def terminate(self):
        self.terminated = True
        self.alive = False


class _Connection_rerank_worker_idle:
    def __init__(self, responses=()):
        self.responses = list(responses)
        self.sent = []
        self.closed = False

    def send(self, value):
        self.sent.append(value)

    def poll(self, timeout=None):
        return bool(self.responses)

    def recv(self):
        return self.responses.pop(0)

    def close(self):
        self.closed = True


def _ready_worker(conn):
    worker = rr._WorkerClient("model")
    worker._process = cast(Any, _Process_rerank_worker_idle())
    worker._connection = conn
    worker.ready.set()
    worker.attempt_done.set()
    return worker


def test_worker_retires_after_idle_timeout(monkeypatch):
    monkeypatch.setattr(rr, "RERANK_WORKER_IDLE_S", 0.05)
    conn = _Connection_rerank_worker_idle()
    worker = _ready_worker(conn)
    process = worker._process
    assert isinstance(process, _Process_rerank_worker_idle)

    worker._mark_ready()          # what _start does once the child says ready
    time.sleep(0.3)

    assert worker.is_ready() is False
    assert conn.sent == [("stop",)]       # graceful stop, not a kill
    assert conn.closed is True
    assert process.terminated is False


def test_worker_respawns_on_next_rerank_after_idle(monkeypatch):
    monkeypatch.setattr(rr, "RERANK_WORKER_IDLE_S", 0.05)
    worker = _ready_worker(_Connection_rerank_worker_idle())
    worker._mark_ready()
    time.sleep(0.3)
    assert worker.is_ready() is False

    started = []
    monkeypatch.setattr(worker, "_start", lambda: started.append(True))
    worker.ensure_started()
    assert started == [True]


def test_predict_resets_the_idle_clock(monkeypatch):
    monkeypatch.setattr(rr, "RERANK_WORKER_IDLE_S", 0.25)
    conn = _Connection_rerank_worker_idle([("result", [0.5], 100, False), ("result", [0.5], 100, False)])
    worker = _ready_worker(conn)
    worker._mark_ready()
    time.sleep(0.15)
    worker.predict("q", ["a"])
    time.sleep(0.15)
    assert worker.is_ready() is True     # 0.3 s since ready, but only 0.15 s idle
    time.sleep(0.3)
    assert worker.is_ready() is False


def test_idle_zero_disables_retirement(monkeypatch):
    monkeypatch.setattr(rr, "RERANK_WORKER_IDLE_S", 0.0)
    monkeypatch.setattr(rr, "RERANK_WORKER_PRESSURE_FILE", None)
    worker = _ready_worker(_Connection_rerank_worker_idle())
    worker._mark_ready()
    time.sleep(0.2)
    assert worker.is_ready() is True


def test_boot_warm_is_skipped_when_worker_is_lazy(monkeypatch):
    monkeypatch.setattr(rr, "RERANK_WORKER_ENABLED", True)
    monkeypatch.setattr(rr, "RERANK_WORKER_IDLE_S", 600.0)
    assert rr.warm_at_boot() is False

    monkeypatch.setattr(rr, "RERANK_WORKER_IDLE_S", 0.0)
    assert rr.warm_at_boot() is True     # a worker that never retires is worth pre-warming

    monkeypatch.setattr(rr, "RERANK_WORKER_ENABLED", False)
    assert rr.warm_at_boot() is True     # in-process model: boot warm stays as it was


def test_fresh_pressure_signal_is_fail_open_for_missing_stale_or_bad_data(
    tmp_path, monkeypatch
):
    signal = tmp_path / "pressure.json"
    monkeypatch.setattr(rr, "RERANK_WORKER_PRESSURE_MAX_AGE_S", 120.0)

    assert rr._worker_pressure_active(signal) is False
    signal.write_text("not-json", encoding="utf-8")
    assert rr._worker_pressure_active(signal) is False
    signal.write_text(json.dumps({"level": "protect"}), encoding="utf-8")
    mtime = signal.stat().st_mtime
    assert rr._worker_pressure_active(signal, now=mtime + 1) is True
    assert rr._worker_pressure_active(signal, now=mtime + 121) is False
    signal.write_text(json.dumps({"level": "warn"}), encoding="utf-8")
    assert rr._worker_pressure_active(signal) is False
    signal.write_text(json.dumps({"pressure": True}), encoding="utf-8")
    assert rr._worker_pressure_active(signal) is True


def test_pressure_retires_ready_worker_without_waiting_full_idle(monkeypatch):
    monkeypatch.setattr(rr, "RERANK_WORKER_IDLE_S", 2700.0)
    monkeypatch.setattr(rr, "RERANK_WORKER_PRESSURE_FILE", "/pressure.json")
    monkeypatch.setattr(rr, "RERANK_WORKER_PRESSURE_POLL_S", 0.02)
    monkeypatch.setattr(rr, "_worker_pressure_active", lambda: True)
    conn = _Connection_rerank_worker_idle()
    worker = _ready_worker(conn)
    process = worker._process
    assert isinstance(process, _Process_rerank_worker_idle)

    worker._mark_ready()
    time.sleep(0.2)

    assert worker.is_ready() is False
    assert conn.sent == [("stop",)]
    assert process.terminated is False


def test_pressure_blocks_cold_worker_spawn(monkeypatch):
    monkeypatch.setattr(rr, "_worker_pressure_active", lambda: True)
    worker = rr._WorkerClient("model")
    started = []
    monkeypatch.setattr(worker, "_start", lambda: started.append(True))

    event = worker.ensure_started()

    assert event.is_set()
    assert started == []
    assert worker.is_ready() is False


def test_pressure_never_retires_an_inflight_prediction(monkeypatch):
    monkeypatch.setattr(rr, "RERANK_WORKER_IDLE_S", 2700.0)
    monkeypatch.setattr(rr, "RERANK_WORKER_PRESSURE_FILE", "/pressure.json")
    monkeypatch.setattr(rr, "RERANK_WORKER_PRESSURE_POLL_S", 60.0)
    monkeypatch.setattr(rr, "_worker_pressure_active", lambda: True)
    conn = _Connection_rerank_worker_idle()
    worker = _ready_worker(conn)

    worker._call_lock.acquire()
    try:
        worker._retire_if_idle()
        assert worker.is_ready() is True
        assert conn.sent == []
    finally:
        worker._call_lock.release()
        if worker._idle_timer is not None:
            worker._idle_timer.cancel()
