"""Cross-thread and cross-process admission for corpus state.

Vecgrep keeps one logical corpus in three physical places: Qdrant, the BM25
sidecar, and ``corpora.json``.  Every process that can touch those surfaces
(API, CLI, MCP, and watchers) uses the same lock files here.  The in-process
reader/writer lock matters because ``flock`` alone is a poor thread scheduler;
the file lock extends the same contract across processes.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal


LockMode = Literal["read", "write"]


class LockUpgradeError(RuntimeError):
    """Raised instead of deadlocking when code tries read -> write nesting."""


class LockTimeout(TimeoutError):
    """Raised when admission does not arrive inside the caller's deadline.

    Callers that can degrade -- search, above all -- pass a timeout and treat
    this as "that corpus is busy" rather than waiting out whatever is holding
    it.  Waiting was the old behaviour and it had no ceiling: one reader
    stalled on disk plus one queued writer left a corpus unsearchable, with no
    error and no log line, until the process was restarted.
    """


class _ThreadRWLock:
    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._readers: dict[int, int] = {}
        self._writer: int | None = None
        self._writer_depth = 0
        self._waiting_writers = 0

    def _wait(self, deadline: float | None) -> bool:
        """Wait to be notified. False once the deadline has passed."""
        if deadline is None:
            self._condition.wait()
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        self._condition.wait(remaining)
        return True

    def acquire(self, mode: LockMode, timeout: float | None = None) -> None:
        ident = threading.get_ident()
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            if self._writer == ident:
                self._writer_depth += 1
                return
            if mode == "read":
                # A thread that already reads re-enters without queueing. The
                # writer it would otherwise defer to is itself waiting for
                # this thread's reader count to reach zero, so deferring here
                # never returns -- and CorpusLocks.hold supports nested reads,
                # so this is a path real callers take.
                if self._readers.get(ident):
                    self._readers[ident] += 1
                    return
                # Prefer queued writers so a steady search stream cannot starve
                # an index/delete indefinitely.
                while self._writer is not None or self._waiting_writers:
                    if not self._wait(deadline):
                        raise LockTimeout("timed out waiting to read")
                self._readers[ident] = 1
                return
            if self._readers.get(ident):
                raise LockUpgradeError(
                    "cannot upgrade a corpus read lock to write; release read first"
                )
            self._waiting_writers += 1
            try:
                while self._writer is not None or self._readers:
                    if not self._wait(deadline):
                        raise LockTimeout("timed out waiting to write")
                self._writer = ident
                self._writer_depth = 1
            finally:
                self._waiting_writers -= 1
                # A writer that gave up may be the only reason readers are
                # parked; nothing else will wake them.
                self._condition.notify_all()

    def release(self, mode: LockMode) -> None:
        ident = threading.get_ident()
        with self._condition:
            if self._writer == ident:
                self._writer_depth -= 1
                if self._writer_depth == 0:
                    self._writer = None
                    self._condition.notify_all()
                return
            if mode != "read" or not self._readers.get(ident):
                raise RuntimeError("corpus lock released by a non-owner")
            remaining = self._readers[ident] - 1
            if remaining:
                self._readers[ident] = remaining
            else:
                del self._readers[ident]
                if not self._readers:
                    self._condition.notify_all()


_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, _ThreadRWLock] = {}
_LOCAL = threading.local()


def _thread_lock(path: Path) -> _ThreadRWLock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, _ThreadRWLock())


def _flock_until(handle, mode: LockMode, deadline: float | None) -> None:
    """Take the file lock, giving up at `deadline` instead of waiting forever."""
    op = fcntl.LOCK_SH if mode == "read" else fcntl.LOCK_EX
    if deadline is None:
        fcntl.flock(handle.fileno(), op)
        return
    while True:
        try:
            fcntl.flock(handle.fileno(), op | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LockTimeout(f"timed out waiting for the {mode} file lock")
            time.sleep(min(0.05, remaining))


def _held() -> dict[str, tuple[LockMode, int, object]]:
    state = getattr(_LOCAL, "held", None)
    if state is None:
        state = {}
        _LOCAL.held = state
    return state


class CorpusLocks:
    """Reader/writer locks keyed by corpus, with deterministic file names."""

    def __init__(self, root: Path | None) -> None:
        self.root = root
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path | None:
        if self.root is None:
            return None
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return self.root / f"{digest}.lock"

    @contextmanager
    def hold(
        self, key: str, mode: LockMode, timeout: float | None = None
    ) -> Iterator[None]:
        path = self._path(key)
        deadline = None if timeout is None else time.monotonic() + timeout
        # Ephemeral services have no shared disk state, but still need thread
        # admission because one service instance may be called concurrently.
        thread_key = path or Path(f"/__vecgrep_ephemeral__/{id(self)}/{key}")
        lock = _thread_lock(thread_key)
        state_key = str(thread_key)
        held = _held()
        prior = held.get(state_key)
        if prior is not None:
            prior_mode, depth, handle = prior
            if prior_mode == "read" and mode == "write":
                raise LockUpgradeError(
                    f"cannot upgrade corpus {key!r} from read to write"
                )
            lock.acquire(mode, timeout=timeout)
            held[state_key] = (prior_mode, depth + 1, handle)
            try:
                yield
            finally:
                old_mode, old_depth, old_handle = held[state_key]
                if old_depth == 1:
                    held.pop(state_key, None)
                else:
                    held[state_key] = (old_mode, old_depth - 1, old_handle)
                lock.release(mode)
            return

        lock.acquire(mode, timeout=timeout)
        handle = None
        try:
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = open(path, "a+")
                # A file-lock timeout unwinds through this same finally, so the
                # thread lock taken above is released and nothing stays half
                # held.
                _flock_until(handle, mode, deadline)
            held[state_key] = (mode, 1, handle)
            yield
        finally:
            held.pop(state_key, None)
            if handle is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
            lock.release(mode)

    def read(self, corpus: str, timeout: float | None = None):
        return self.hold(f"corpus:{corpus}", "read", timeout=timeout)

    def write(self, corpus: str, timeout: float | None = None):
        return self.hold(f"corpus:{corpus}", "write", timeout=timeout)

    def registry_read(self, timeout: float | None = None):
        return self.hold("registry", "read", timeout=timeout)

    def registry_write(self, timeout: float | None = None):
        return self.hold("registry", "write", timeout=timeout)
