"""Bound failed owner-approval code attempts on the unlock form, per client."""
from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import deque
from typing import Callable, Deque

from starlette.types import Scope


def client_key(scope: Scope) -> str:
    """Key attempts by the address tailscaled reports, not the socket peer.

    Funnel and Serve requests all arrive from loopback. tailscaled discards any
    X-Forwarded-For the client sent and sets its own, so the first address is
    the real client. A direct connection has no such header and falls back to
    the socket peer.
    """
    forwarded = ""
    for name, value in scope.get("headers") or ():
        if name == b"x-forwarded-for":
            forwarded = value.decode("latin-1").split(",", 1)[0].strip()
            break
    client = scope.get("client") or ("", 0)
    address = forwarded or str(client[0] or "unknown")
    return hashlib.sha256(address.encode()).hexdigest()[:32]


class FailureWindow:
    """Sliding window of failures per key.

    Once a key has `limit` failures inside `window_s`, it stays locked until
    the oldest of them ages out. Locked attempts are not recorded, so a lock
    cannot be extended by hammering it.
    """

    def __init__(self, *, limit: int = 10, window_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic,
                 max_keys: int = 10_000) -> None:
        self._limit = max(1, int(limit))
        self._window_s = float(window_s)
        self._clock = clock
        self._max_keys = max_keys
        self._lock = threading.Lock()
        self._failures: dict[str, Deque[float]] = {}

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()

    def retry_after(self, key: str) -> int:
        """Whole seconds until `key` may try again; 0 when it is not locked."""
        with self._lock:
            now = self._clock()
            recent = self._prune(key, now)
            if len(recent) < self._limit:
                return 0
            return max(1, math.ceil(self._window_s - (now - recent[0])))

    def record_failure(self, key: str) -> None:
        with self._lock:
            now = self._clock()
            recent = self._prune(key, now)
            recent.append(now)
            self._failures[key] = recent
            if len(self._failures) > self._max_keys:
                # Evict the oldest-inserted key so cycling source addresses
                # cannot grow this table without bound.
                oldest = next(iter(self._failures))
                if oldest != key:
                    self._failures.pop(oldest, None)

    def _prune(self, key: str, now: float) -> Deque[float]:
        recent = self._failures.get(key)
        if recent is None:
            return deque()
        while recent and now - recent[0] >= self._window_s:
            recent.popleft()
        if not recent:
            del self._failures[key]
        return recent


UNLOCK_FAILURES = FailureWindow()
