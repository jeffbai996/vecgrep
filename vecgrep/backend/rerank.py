"""Cross-encoder reranking.

Off by default. When enabled, takes the candidate pool from any retrieval
mode (hybrid/vector/bm25) and rescores each (query, chunk) pair with a
cross-encoder. Cross-encoders are slow but much more accurate than the
bi-encoder used for the initial vector retrieval — running them only on
top-50 keeps latency reasonable.

Lazy import: sentence-transformers pulls in torch (~hundreds of MB) and
is gated behind the optional `vecgrep[rerank]` extra. We import it only
when the user actually asks for reranking.
"""
from __future__ import annotations

import atexit
import json
import logging
import math
import os
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)

# Round 3 (2026-08-18, 119 gold cases / 26 negatives, docs/STORAGE_RETRIEVAL):
#   base   hit@3 76.3  mrr .681  neg FP 11.5%  +250 ms
#   v2-m3  hit@3 82.8  mrr .702  neg FP  3.8%  +1.4 s
#   large  hit@3 79.6  mrr .697  neg FP  3.8%  +3.5 s
# v2-m3 is the first reranker that does not demote answers the pool already
# had (base lost hit@3 vs unreranked; v2-m3 gains it) and it calibrates the
# negatives, so it is the default as of 2026-08-18. It was held back until now
# for a serve-path reason, not a quality one: loading it took 5+ minutes inside
# a request, which held one of eight sync-threadpool slots and starved the
# health route until squad-watchdog (strikes=1) restarted the server
# mid-request. ensure_warm() + wait_ready() below removed that — the model
# loads in a background thread at startup and a search that arrives before it
# is ready returns fusion order instead of waiting. Fall back per install with
# VECGREP_RERANKER=BAAI/bge-reranker-base.
DEFAULT_RERANKER = os.environ.get("VECGREP_RERANKER", "BAAI/bge-reranker-v2-m3")


# How long a search waits for a cold reranker before giving up and returning
# fusion order. Loading a cross-encoder imports torch and, the first time,
# downloads weights -- minutes, not milliseconds. Search runs in FastAPI's sync
# threadpool, capped at 8 (VECGREP_THREAD_POOL_SIZE), so a blocking load does
# not merely slow one request: it holds one of eight threads for the whole load
# and starves every other sync route, health check included. That is how the
# v2-m3 first load took the server down on 2026-08-18 -- squad-watchdog saw the
# timeout and at strikes=1 restarted vecgrep-serve mid-request. Waiting a few
# seconds is fine; waiting minutes is an outage.
RERANK_WAIT_S = float(os.environ.get("VECGREP_RERANK_WAIT_S", "5"))

# A load that fails (no extra installed, no network for the weights) must not
# be retried on every single search -- but it must not disable reranking until
# someone restarts the process either. Retry no more often than this.
_RETRY_AFTER_S = 300.0

# How many (query, chunk) pairs go through the cross-encoder at once.
# sentence-transformers defaults to 32, and a batch is padded to its longest
# member -- so 32 transcript chunks (up to 4000 chars each) become one enormous
# padded tensor. On 2026-08-20 a single reranked search on the transcript
# corpus took the 24 GB card from 5.0 GB to 17.5 GB; three of those filled it
# and the next allocation came back as `CUDA error: unknown error`, which is
# what made the card look wedged that morning. Batch size bounds the peak and
# changes nothing about the scores: each pair is scored independently, so the
# only thing that moves is how many are in flight. Raise it on a card with
# room to spare.
RERANK_BATCH = int(os.environ.get("VECGREP_RERANK_BATCH", "8"))


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Process isolation is opt-in so library/CLI callers keep the direct behavior
# they have always had. Long-running servers should enable it: exiting the
# child is the only reliable way to return every retained Torch/Python page to
# the OS after a high-water workload.
RERANK_WORKER_ENABLED = _env_enabled("VECGREP_RERANK_WORKER")
RERANK_WORKER_MAX_BYTES = max(
    0, int(os.environ.get("VECGREP_RERANK_WORKER_MAX_MB", "2300")) * 1024 * 1024
)
RERANK_WORKER_MAX_JOBS = max(
    0, int(os.environ.get("VECGREP_RERANK_WORKER_MAX_JOBS", "64"))
)
RERANK_WORKER_START_TIMEOUT_S = float(
    os.environ.get("VECGREP_RERANK_WORKER_START_TIMEOUT_S", "600")
)
RERANK_WORKER_CALL_TIMEOUT_S = float(
    os.environ.get("VECGREP_RERANK_WORKER_CALL_TIMEOUT_S", "60")
)
# A timed-out or crashed prediction usually means the accelerator is wedged or
# unavailable, not that immediately loading another 1.2 GB worker will help.
# Keep automatic callers on the already-computed fusion order for a bounded
# cooldown, then let the next request probe again. Scores and candidate sets
# are untouched; this only stops a failed enhancement from becoming a respawn
# loop.
RERANK_WORKER_FAILURE_COOLDOWN_S = max(
    0.0,
    float(os.environ.get("VECGREP_RERANK_WORKER_FAILURE_COOLDOWN_S", "300")),
)
# Retire the worker after this many seconds without a predict. The
# cross-encoder is ~1.2 GB resident for a feature most searches never touch;
# on a shared box that is the difference between the host having headroom
# and not. The next rerank respawns it (searches that arrive while it loads
# already fall back to fusion order). 0 keeps the worker for the life of the
# server, which was the behaviour before this knob existed.
RERANK_WORKER_IDLE_S = max(
    0.0, float(os.environ.get("VECGREP_RERANK_WORKER_IDLE_S", "600"))
)

# Optional host-pressure signal. The production watcher writes a tiny JSON
# document containing either ``level`` (warn/protect/hard/critical) or a
# boolean ``pressure``. This remains opt-in so vecgrep has no dependency on a
# particular host governor. A stale or malformed signal fails open: retrieval
# still works, and the ordinary idle/max-bytes bounds remain in force.
RERANK_WORKER_PRESSURE_FILE = os.environ.get("VECGREP_RERANK_WORKER_PRESSURE_FILE")
RERANK_WORKER_PRESSURE_POLL_S = max(
    1.0, float(os.environ.get("VECGREP_RERANK_WORKER_PRESSURE_POLL_S", "15"))
)
RERANK_WORKER_PRESSURE_MAX_AGE_S = max(
    1.0,
    float(os.environ.get("VECGREP_RERANK_WORKER_PRESSURE_MAX_AGE_S", "120")),
)
RERANK_WORKER_PRESSURE_LEVELS = frozenset(
    level.strip().lower()
    for level in os.environ.get(
        "VECGREP_RERANK_WORKER_PRESSURE_LEVELS", "protect,hard,critical"
    ).split(",")
    if level.strip()
)


def _worker_pressure_active(
    path: str | os.PathLike[str] | None = None,
    *,
    now: float | None = None,
) -> bool:
    """Return whether a fresh, valid host signal asks heavy workers to yield."""
    raw_path = path if path is not None else RERANK_WORKER_PRESSURE_FILE
    if not raw_path:
        return False
    signal = Path(raw_path)
    try:
        stat = signal.stat()
        current = time.time() if now is None else now
        if current - stat.st_mtime > RERANK_WORKER_PRESSURE_MAX_AGE_S:
            return False
        payload = json.loads(signal.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("pressure") is True:
        return True
    return str(payload.get("level") or "").strip().lower() in RERANK_WORKER_PRESSURE_LEVELS


def warm_at_boot() -> bool:
    """Should the server pre-load the reranker at startup?

    Only when it would stay loaded anyway. A worker that retires when idle
    would be spawned at boot just to be retired ten minutes later — 1.2 GB of
    churn for nothing, on the one occasion (startup) the box is busiest.
    """
    return not (RERANK_WORKER_ENABLED and RERANK_WORKER_IDLE_S > 0)


def _release_cuda_cache() -> None:
    """Hand torch's reserve back to the driver.

    The caching allocator keeps every block it has ever used, so the high-water
    mark of one unlucky batch stays charged to the card for the life of the
    process. That is fine when torch owns the GPU; here it shares a 24 GB card
    with the transcription model and the desktop, and an idle search server
    holding 19 GB starves both. Costs a sync (tens of ms) against a rerank that
    already takes over a second.

    Best-effort by design: reranking is an optional extra, so torch may be
    absent entirely, and on a CPU-only box there is nothing to release.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 -- releasing memory must never fail a search
        pass


class RerankerError(RuntimeError):
    pass


_cache: dict[str, object] = {}
_lock = threading.Lock()
_warming: dict[str, threading.Event] = {}
_failed_at: dict[str, float] = {}


def _process_committed_bytes(
    path: Path = Path("/proc/self/smaps_rollup"),
) -> int:
    """Resident plus swapped pages owned by this process.

    RSS alone falls when Linux swaps a retained allocator arena out, which can
    make a bloated process look healthy immediately before it faults the same
    pages back in. Counting both is the meaningful recycle threshold. On a
    non-Linux host the metric is unavailable and the fixed job limit remains
    the fallback bound.
    """
    values = {"Rss": 0, "Swap": 0}
    try:
        for line in path.read_text(encoding="ascii").splitlines():
            key = line.split(":", 1)[0]
            if key in values:
                values[key] = int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return 0
    return values["Rss"] + values["Swap"]


def _construct_model(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise RerankerError(
            "Reranking requires the 'rerank' extra. "
            "Install with `pip install vecgrep[rerank]`."
        ) from e
    try:
        import torch

        kwargs = {}
        if torch.cuda.is_available():
            # The v2-m3 checkpoint is stored as 2.2 GB of FP32 weights. Loading
            # directly in FP16 halves the steady-state model allocation and is
            # the native fast path on CUDA. CPU keeps the upstream default.
            kwargs = {"model_kwargs": {"torch_dtype": torch.float16}}
        return CrossEncoder(model_name, **kwargs)
    except Exception as e:
        raise RerankerError(
            f"Failed to load cross-encoder '{model_name}': {e}"
        ) from e


def _worker_main(
    connection,
    model_name: str,
    *,
    batch_size: int,
    max_jobs: int,
    max_bytes: int,
) -> None:
    """Own the model and CUDA context until the configured recycle point."""
    try:
        try:
            model = _construct_model(model_name)
        except Exception as exc:
            connection.send(("load-error", type(exc).__name__, str(exc)))
            return
        connection.send(("ready",))
        jobs = 0
        while True:
            try:
                request = connection.recv()
            except EOFError:
                return
            if not request or request[0] == "stop":
                return
            if request[0] != "predict":
                connection.send(("error", "ProtocolError", "unknown command"))
                return
            _, query, texts = request
            pairs = [(query, text) for text in texts]
            try:
                raw = model.predict(pairs, batch_size=batch_size)
                scores = [float(score) for score in raw]
            except Exception as exc:
                connection.send(("error", type(exc).__name__, str(exc)))
                return
            finally:
                _release_cuda_cache()
            jobs += 1
            committed = _process_committed_bytes()
            recycle = (
                (max_jobs > 0 and jobs >= max_jobs)
                or (max_bytes > 0 and committed >= max_bytes)
            )
            connection.send(("result", scores, committed, recycle))
            if recycle:
                return
    finally:
        try:
            connection.close()
        except OSError:
            pass


class _WorkerClient:
    """Thread-safe parent-side owner of one spawned reranker process."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.ready = threading.Event()
        self.attempt_done = threading.Event()
        self._state_lock = threading.Lock()
        self._call_lock = threading.Lock()
        self._process = None
        self._connection = None
        self._starting = False
        self._retiring = False
        self._failed_at: float | None = None
        self._last_used = time.monotonic()
        self._idle_timer: threading.Timer | None = None
        self._pressure_logged = False

    def _mark_ready(self) -> None:
        with self._state_lock:
            self._starting = False
            self._failed_at = None
            self._last_used = time.monotonic()
            self.ready.set()
            self.attempt_done.set()
        self._arm_idle_timer()

    def _arm_idle_timer(self) -> None:
        idle = RERANK_WORKER_IDLE_S
        delays = []
        if idle > 0:
            delays.append(idle)
        if RERANK_WORKER_PRESSURE_FILE:
            delays.append(RERANK_WORKER_PRESSURE_POLL_S)
        if not delays:
            return
        with self._state_lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
            timer = threading.Timer(min(delays), self._retire_if_idle)
            timer.daemon = True
            timer.name = "vecgrep-rerank-worker-idle"
            self._idle_timer = timer
        timer.start()

    def _retire_if_idle(self) -> None:
        # A predict may have landed just before the timer fired; if so, the
        # clock was reset and this firing is stale — re-arm for the remainder.
        idle = RERANK_WORKER_IDLE_S
        pressure = _worker_pressure_active()
        if not self._call_lock.acquire(blocking=False):
            self._arm_idle_timer()
            return
        try:
            with self._state_lock:
                elapsed = time.monotonic() - self._last_used
            if pressure and self.is_ready():
                logger.info(
                    "retiring reranker worker %s under host memory pressure",
                    self.model_name,
                )
                self._retire(terminate=False)
                return
            if idle > 0 and elapsed < idle:
                self._arm_idle_timer()
                return
            if idle <= 0:
                self._arm_idle_timer()
                return
            if not self.is_ready():
                return
            logger.info(
                "retiring idle reranker worker %s after %.0fs", self.model_name, elapsed
            )
            self._retire(terminate=False)
        finally:
            self._call_lock.release()

    def is_ready(self) -> bool:
        with self._state_lock:
            process = self._process
            ready = self.ready.is_set() and process is not None and process.is_alive()
            if not ready:
                self.ready.clear()
            return ready

    def ensure_started(self) -> threading.Event:
        stale_connection = None
        with self._state_lock:
            process = self._process
            if self.ready.is_set() and process is not None and process.is_alive():
                self.attempt_done.set()
                return self.attempt_done
            if self._starting or self._retiring:
                return self.attempt_done
            if process is not None and process.is_alive():
                # A failed retirement still owns this model process.
                self.attempt_done.set()
                return self.attempt_done
            if (
                self._failed_at is not None
                and time.monotonic() - self._failed_at
                < RERANK_WORKER_FAILURE_COOLDOWN_S
            ):
                self.attempt_done.set()
                return self.attempt_done
        pressure = _worker_pressure_active()
        with self._state_lock:
            # The pressure-file read happens outside the lock. Recheck the
            # state so two cold callers cannot both decide to spawn.
            process = self._process
            if self.ready.is_set() and process is not None and process.is_alive():
                self.attempt_done.set()
                return self.attempt_done
            if self._starting or self._retiring:
                return self.attempt_done
            if process is not None and process.is_alive():
                # A failed retirement still owns this model process.
                self.attempt_done.set()
                return self.attempt_done
            if (
                self._failed_at is not None
                and time.monotonic() - self._failed_at
                < RERANK_WORKER_FAILURE_COOLDOWN_S
            ):
                self.attempt_done.set()
                return self.attempt_done
            if pressure:
                if not self._pressure_logged:
                    logger.info(
                        "reranker worker %s suppressed under host memory pressure",
                        self.model_name,
                    )
                    self._pressure_logged = True
                self.attempt_done.set()
                return self.attempt_done
            self._pressure_logged = False
            stale_connection = self._connection
            self._process = None
            self._connection = None
            self._starting = True
            self.ready.clear()
            self.attempt_done.clear()
        if stale_connection is not None:
            try:
                stale_connection.close()
            except OSError:
                pass
        threading.Thread(
            target=self._start,
            name="vecgrep-rerank-worker-warm",
            daemon=True,
        ).start()
        return self.attempt_done

    def _start(self) -> None:
        import multiprocessing

        process = None
        parent_connection = None
        try:
            context = multiprocessing.get_context("spawn")
            parent_connection, child_connection = context.Pipe()
            process = context.Process(
                target=_worker_main,
                args=(child_connection, self.model_name),
                kwargs={
                    "batch_size": RERANK_BATCH,
                    "max_jobs": RERANK_WORKER_MAX_JOBS,
                    "max_bytes": RERANK_WORKER_MAX_BYTES,
                },
                name="vecgrep-reranker",
                daemon=True,
            )
            process.start()
            # Own the live child before any fallible post-start operation so
            # startup cleanup can always find and retire it.
            with self._state_lock:
                self._process = process
                self._connection = parent_connection
            child_connection.close()
            if not parent_connection.poll(RERANK_WORKER_START_TIMEOUT_S):
                raise TimeoutError("reranker worker load timed out")
            response = parent_connection.recv()
            if not response or response[0] != "ready":
                detail = ": ".join(str(part) for part in response[1:])
                raise RerankerError(detail or "reranker worker failed to load")
            self._mark_ready()
            logger.info("reranker worker %s ready pid=%s", self.model_name, process.pid)
        except Exception as exc:
            logger.warning("reranker worker %s failed to start: %s", self.model_name, exc)
            if parent_connection is not None:
                try:
                    parent_connection.close()
                except OSError:
                    pass
            if process is not None:
                self._retire(terminate=True)
            with self._state_lock:
                self._starting = False
                self._failed_at = time.monotonic()
                self.ready.clear()
                self.attempt_done.set()

    def predict(self, query: str, texts: list[str]) -> list[float]:
        with self._call_lock:
            if not self.is_ready():
                raise RerankerError(f"reranker worker '{self.model_name}' is not ready")
            with self._state_lock:
                connection = self._connection
            if connection is None:
                raise RerankerError(f"reranker worker '{self.model_name}' is unavailable")
            try:
                connection.send(("predict", query, texts))
                if not connection.poll(RERANK_WORKER_CALL_TIMEOUT_S):
                    raise TimeoutError("reranker worker prediction timed out")
                response = connection.recv()
            except Exception as exc:
                self._retire(terminate=True)
                self._open_circuit(exc)
                raise RerankerError(str(exc)) from exc
            if not response or response[0] != "result":
                detail = ": ".join(str(part) for part in response[1:])
                self._retire(terminate=True)
                error = RerankerError(detail or "reranker worker prediction failed")
                self._open_circuit(error)
                raise error
            _, scores, committed, recycle = response
            with self._state_lock:
                self._last_used = time.monotonic()
            if recycle:
                logger.info(
                    "recycling reranker worker %s after committed_mb=%.1f",
                    self.model_name,
                    committed / 1024 / 1024,
                )
                self._retire(terminate=False)
                self.ensure_started()
            return [float(score) for score in scores]

    def _open_circuit(self, exc: Exception) -> None:
        with self._state_lock:
            self._failed_at = time.monotonic()
            self.attempt_done.set()
        logger.warning(
            "reranker worker %s circuit open for %.0fs after %s",
            self.model_name,
            RERANK_WORKER_FAILURE_COOLDOWN_S,
            type(exc).__name__,
        )

    def _retire(self, *, terminate: bool) -> None:
        with self._state_lock:
            process = self._process
            connection = self._connection
            self._retiring = True
            self._connection = None
            self.ready.clear()
            self.attempt_done.set()
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
        if connection is not None:
            if not terminate:
                try:
                    connection.send(("stop",))
                except (BrokenPipeError, EOFError, OSError):
                    pass
            try:
                connection.close()
            except OSError:
                pass
        try:
            if process is not None:
                if not terminate:
                    process.join(timeout=2)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
                if process.is_alive():
                    logger.error("reranker worker %s survived shutdown", self.model_name)
        finally:
            with self._state_lock:
                # Even failed OS termination must not abandon ownership and
                # permit another model allocation alongside the survivor.
                if self._process is process and (process is None or not process.is_alive()):
                    self._process = None
                self._retiring = False

    def stop(self) -> None:
        with self._call_lock:
            self._retire(terminate=False)


_workers: dict[str, _WorkerClient] = {}
_workers_lock = threading.Lock()


def _worker_for(model_name: str) -> _WorkerClient:
    with _workers_lock:
        worker = _workers.get(model_name)
        if worker is None:
            worker = _WorkerClient(model_name)
            _workers[model_name] = worker
        return worker


def _worker_predict(query: str, texts: list[str], model_name: str) -> list[float]:
    return _worker_for(model_name).predict(query, texts)


def shutdown_workers() -> None:
    with _workers_lock:
        workers = list(_workers.values())
        _workers.clear()
    for worker in workers:
        worker.stop()


atexit.register(shutdown_workers)


def ensure_warm(model_name: str = DEFAULT_RERANKER) -> threading.Event:
    """Load `model_name` in the background if it is neither loaded nor already
    loading. Returns an Event set once the attempt finishes, successfully or
    not. Idempotent, so calling it per request is free after the first."""
    if RERANK_WORKER_ENABLED:
        return _worker_for(model_name).ensure_started()
    with _lock:
        if model_name in _cache:
            done = threading.Event()
            done.set()
            return done
        ev = _warming.get(model_name)
        if ev is not None:
            failed = _failed_at.get(model_name)
            if failed is None or (time.monotonic() - failed) < _RETRY_AFTER_S:
                return ev
            # Past the cooldown after a failure: drop the old attempt and retry.
            _failed_at.pop(model_name, None)
        ev = threading.Event()
        _warming[model_name] = ev

    def _run() -> None:
        try:
            _load(model_name)
            logger.info("reranker %s ready", model_name)
        except Exception as exc:
            with _lock:
                _failed_at[model_name] = time.monotonic()
            logger.warning("reranker %s failed to load: %s", model_name, exc)
        finally:
            ev.set()

    threading.Thread(
        target=_run, name="vecgrep-rerank-warm", daemon=True
    ).start()
    return ev


def is_ready(model_name: str = DEFAULT_RERANKER) -> bool:
    """True when the model is loaded and predict() will not block on I/O."""
    if RERANK_WORKER_ENABLED:
        return _worker_for(model_name).is_ready()
    with _lock:
        return model_name in _cache


def wait_ready(
    model_name: str = DEFAULT_RERANKER, timeout: float | None = None
) -> bool:
    """Warm if needed, wait a BOUNDED time, and report whether it is usable.

    False is not an error — it means "not yet", and the caller should fall
    back to unreranked order rather than hold a threadpool slot open."""
    if is_ready(model_name):
        return True
    ev = ensure_warm(model_name)
    ev.wait(RERANK_WAIT_S if timeout is None else timeout)
    return is_ready(model_name)


def _load(model_name: str):
    with _lock:
        cached = _cache.get(model_name)
    if cached is not None:
        return cached
    model = _construct_model(model_name)
    with _lock:
        _cache[model_name] = model
    return model


def rerank(
    query: str,
    candidates: list[tuple[str, dict]],
    model_name: str = DEFAULT_RERANKER,
) -> list[tuple[float, dict]]:
    """Score (query, chunk_text) for each candidate. Returns (score, payload)
    pairs sorted descending. Scores are sigmoid-mapped 0..1 for downstream
    display.
    """
    if not candidates:
        return []
    texts = [text for text, _ in candidates]
    if RERANK_WORKER_ENABLED:
        raw = _worker_predict(query, texts, model_name)
    else:
        model = _load(model_name)
        pairs = [(query, text) for text in texts]
        try:
            raw = model.predict(pairs, batch_size=RERANK_BATCH)  # numpy logits
        finally:
            # Release on the failure path too: a rerank that dies mid-batch is
            # exactly when the card is most full and the next caller needs room.
            _release_cuda_cache()

    # bge-reranker emits raw logits; squashing through sigmoid puts them in
    # 0..1 which is more useful for percentage display than raw values.
    scored = [
        (1 / (1 + math.exp(-float(s))), payload)
        for s, (_, payload) in zip(raw, candidates)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored
