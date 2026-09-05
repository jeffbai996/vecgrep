"""The reranker worker is a 1.2 GB process for a feature that is idle most of
the day. It should exist only while it is being used: retire after a quiet
spell, come back on the next rerank, and never be spawned at boot just in case.
"""
from __future__ import annotations

import json
import time

from vecgrep.backend import rerank as rr


class _Process:
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


class _Connection:
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
    worker._process = _Process()
    worker._connection = conn
    worker.ready.set()
    worker.attempt_done.set()
    return worker


def test_worker_retires_after_idle_timeout(monkeypatch):
    monkeypatch.setattr(rr, "RERANK_WORKER_IDLE_S", 0.05)
    conn = _Connection()
    worker = _ready_worker(conn)
    process = worker._process

    worker._mark_ready()          # what _start does once the child says ready
    time.sleep(0.3)

    assert worker.is_ready() is False
    assert conn.sent == [("stop",)]       # graceful stop, not a kill
    assert conn.closed is True
    assert process.terminated is False


def test_worker_respawns_on_next_rerank_after_idle(monkeypatch):
    monkeypatch.setattr(rr, "RERANK_WORKER_IDLE_S", 0.05)
    worker = _ready_worker(_Connection())
    worker._mark_ready()
    time.sleep(0.3)
    assert worker.is_ready() is False

    started = []
    monkeypatch.setattr(worker, "_start", lambda: started.append(True))
    worker.ensure_started()
    assert started == [True]


def test_predict_resets_the_idle_clock(monkeypatch):
    monkeypatch.setattr(rr, "RERANK_WORKER_IDLE_S", 0.25)
    conn = _Connection([("result", [0.5], 100, False), ("result", [0.5], 100, False)])
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
    worker = _ready_worker(_Connection())
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
    conn = _Connection()
    worker = _ready_worker(conn)
    process = worker._process

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
    conn = _Connection()
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
