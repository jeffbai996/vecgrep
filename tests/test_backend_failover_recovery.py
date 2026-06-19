"""Backend cache must re-resolve when its embedder dies mid-session.

A long-lived `serve` process resolves + CACHES an embed backend at startup.
When the primary Ollama dies while the service is running, the cached backend
keeps pointing at the dead endpoint and every embed raises — search dies and
never re-probes for the fallback until a restart. The service must drop a dead
backend and re-resolve (→ failover) on the fly, not rely on a restart.
"""
from __future__ import annotations

import pytest

from vecgrep.backend.embed.base import EmbedBackend, EmbedBackendError
from vecgrep.backend.service import VecgrepService


class _DyingBackend(EmbedBackend):
    """Embeds fine until `die_after` calls, then raises like a dead Ollama."""
    name = "dying"
    model = "stub-1"
    dim = 16

    def __init__(self, die_after: int = 0):
        self._calls = 0
        self._die_after = die_after

    def embed(self, texts):
        self._calls += 1
        if self._calls > self._die_after:
            raise EmbedBackendError("Could not reach Ollama at http://dead:11434")
        return [[0.1] * self.dim for _ in texts]


class _HealthyBackend(EmbedBackend):
    name = "healthy"
    model = "stub-1"
    dim = 16

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[0.2] * self.dim for _ in texts]


def test_embed_reresolves_to_fallback_when_cached_backend_dies(vg_home, monkeypatch):
    """A cached backend that starts raising → service drops it, re-resolves via
    get_embed_backend (the failover path), and the embed succeeds on the healthy
    fallback. No restart required."""
    svc = VecgrepService(ephemeral=False)
    dead = _DyingBackend(die_after=0)        # raises on first embed
    healthy = _HealthyBackend()
    svc._backend_cache["auto"] = dead

    # When the service re-resolves, hand it the healthy backend (simulating the
    # factory now probing primary-dead → fallback-alive).
    monkeypatch.setattr(
        "vecgrep.backend.service.get_embed_backend",
        lambda *a, **k: healthy,
    )

    vec = svc._embed_query_with_failover(None, "hello")
    assert vec == [0.2] * 16          # came from the healthy fallback
    assert healthy.calls == 1
    # Cache was refreshed off the dead backend (may be wrapped in CachedBackend).
    assert svc._backend_cache.get("auto") is not dead


def test_failure_propagates_when_reresolve_also_dead(vg_home, monkeypatch):
    """If the re-resolve ALSO yields a dead backend, the error surfaces — one
    retry, not an infinite loop."""
    svc = VecgrepService(ephemeral=False)
    svc._backend_cache["auto"] = _DyingBackend(die_after=0)
    monkeypatch.setattr(
        "vecgrep.backend.service.get_embed_backend",
        lambda *a, **k: _DyingBackend(die_after=0),
    )
    with pytest.raises(EmbedBackendError):
        svc._embed_query_with_failover(None, "hello")


def test_healthy_backend_not_reresolved(vg_home, monkeypatch):
    """A working backend embeds normally and is NOT evicted — no cache thrash."""
    svc = VecgrepService(ephemeral=False)
    healthy = _HealthyBackend()
    svc._backend_cache["auto"] = healthy
    calls = {"n": 0}

    def _should_not_run(*a, **k):
        calls["n"] += 1
        return _HealthyBackend()
    monkeypatch.setattr("vecgrep.backend.service.get_embed_backend", _should_not_run)

    vec = svc._embed_query_with_failover(None, "hello")
    assert vec == [0.2] * 16
    assert calls["n"] == 0           # never re-resolved
    assert svc._backend_cache["auto"] is healthy
