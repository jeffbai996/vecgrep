from __future__ import annotations

import pytest

from vecgrep.backend.config import Settings
from vecgrep.backend.embed import factory
from vecgrep.backend.embed.base import EmbedBackend, EmbedBackendError
from vecgrep.backend.embed.factory import get_embed_backend
from vecgrep.backend.embed.ollama import OllamaBackend
from vecgrep.backend.service import VecgrepService

# ── test_embed_factory ──

def _settings(**kw) -> Settings:
    s = Settings()
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_primary_alive_uses_primary(monkeypatch):
    s = _settings(ollama_url="http://primary:11434",
                  ollama_fallback_url="http://fallback:11434")
    monkeypatch.setattr(factory, "_ollama_alive", lambda url: url == "http://primary:11434")
    b = factory.get_embed_backend(s)
    assert isinstance(b, OllamaBackend)
    assert b.base_url == "http://primary:11434"


def test_ollama_num_batch_reaches_backend(monkeypatch):
    s = _settings(
        ollama_url="http://primary:11434",
        ollama_fallback_url=None,
        ollama_num_batch=2048,
    )
    monkeypatch.setattr(factory, "_ollama_alive", lambda url: True)

    b = factory.get_embed_backend(s)

    assert isinstance(b, OllamaBackend)
    assert b.num_batch == 2048


def test_primary_dead_falls_back_to_secondary_ollama(monkeypatch):
    s = _settings(ollama_url="http://primary:11434",
                  ollama_fallback_url="http://fallback:11434")
    # primary dead, fallback alive
    monkeypatch.setattr(factory, "_ollama_alive", lambda url: url == "http://fallback:11434")
    b = factory.get_embed_backend(s)
    assert isinstance(b, OllamaBackend)
    assert b.base_url == "http://fallback:11434"


def test_both_ollama_dead_never_reaches_for_openai(monkeypatch):
    # This used to select OpenAI. It no longer does, with or without a key:
    # OpenAI embeds in its own vector space, so writing those vectors into a
    # corpus built with bge-m3 corrupts it with no error anywhere — and it is
    # a metered call nobody asked for. Only an explicit pin reaches OpenAI.
    monkeypatch.setattr(factory, "OpenAIBackend",
                        lambda *a, **k: pytest.fail("reached OpenAI"))
    s = _settings(ollama_url="http://primary:11434",
                  ollama_fallback_url="http://fallback:11434",
                  openai_api_key="sk-test")  # a key being present is the trap
    monkeypatch.setattr(factory, "_ollama_alive", lambda url: False)
    with pytest.raises(factory.EmbedBackendError, match="Ollama not reachable") as exc:
        factory.get_embed_backend(s)
    # The old message advised exporting a key. That advice was the bug.
    assert "OPENAI_API_KEY" not in str(exc.value)


def test_both_ollama_dead_no_key_raises(monkeypatch):
    s = _settings(ollama_url="http://primary:11434",
                  ollama_fallback_url="http://fallback:11434")
    monkeypatch.setattr(factory, "_ollama_alive", lambda url: False)
    with pytest.raises(factory.EmbedBackendError):
        factory.get_embed_backend(s)


def test_no_fallback_url_also_raises_rather_than_billing(monkeypatch):
    # Unset fallback: one dead probe, then the same loud failure.
    monkeypatch.setattr(factory, "OpenAIBackend",
                        lambda *a, **k: pytest.fail("reached OpenAI"))
    s = _settings(ollama_url="http://primary:11434", ollama_fallback_url=None,
                  openai_api_key="sk-test")  # a key being present is the trap
    monkeypatch.setattr(factory, "_ollama_alive", lambda url: False)
    with pytest.raises(factory.EmbedBackendError):
        factory.get_embed_backend(s)


def test_explicit_openai_pin_still_works(monkeypatch):
    """Only the AUTOMATIC path is gone; pinning is a decision, not a surprise."""
    sentinel = object()
    monkeypatch.setattr(factory, "OpenAIBackend", lambda *a, **k: sentinel)
    s = _settings(ollama_url="http://primary:11434", openai_api_key="sk-test")
    monkeypatch.setattr(factory, "_ollama_alive", lambda url: False)
    assert factory.get_embed_backend(s, prefer="openai") is sentinel


def test_prefer_ollama_uses_primary_when_alive(monkeypatch):
    # A corpus pinned prefer='ollama' uses the primary when it's alive.
    s = _settings(ollama_url="http://primary:11434",
                  ollama_fallback_url="http://fallback:11434")
    monkeypatch.setattr(factory, "_ollama_alive", lambda url: True)
    b = factory.get_embed_backend(s, prefer="ollama")
    assert isinstance(b, OllamaBackend)
    assert b.base_url == "http://primary:11434"


def test_prefer_ollama_FAILS_OVER_when_primary_dead(monkeypatch):
    # The bug: a pinned prefer='ollama' corpus used the primary url blindly,
    # bypassing the failover — so a dead primary never fell over. It MUST honor
    # the same primary->fallback probe as the unpinned path.
    s = _settings(ollama_url="http://primary:11434",
                  ollama_fallback_url="http://fallback:11434")
    monkeypatch.setattr(factory, "_ollama_alive", lambda url: url == "http://fallback:11434")
    b = factory.get_embed_backend(s, prefer="ollama")
    assert isinstance(b, OllamaBackend)
    assert b.base_url == "http://fallback:11434"


def test_fallback_only_probed_when_primary_dead(monkeypatch):
    # Don't waste a probe on the fallback if the primary is alive.
    probed = []
    monkeypatch.setattr(factory, "_ollama_alive",
                        lambda url: probed.append(url) or (url == "http://primary:11434"))
    s = _settings(ollama_url="http://primary:11434",
                  ollama_fallback_url="http://fallback:11434")
    factory.get_embed_backend(s)
    assert probed == ["http://primary:11434"]  # fallback never probed

# ── test_mixed_model_backend ──

_OLLAMA_URL = "http://localhost:11434"


def test_get_embed_backend_honors_explicit_model(monkeypatch):
    # Model selection is independent of whether the developer runs Ollama.
    monkeypatch.setattr("vecgrep.backend.embed.factory._ollama_alive", lambda url: True)
    s = Settings(ollama_url=_OLLAMA_URL, embed_model="nomic-embed-text")
    # Pin a different model than the settings default; backend must use it.
    b = get_embed_backend(s, prefer="ollama", model="bge-m3")
    assert b.model == "bge-m3"
    # Without an override, falls back to the settings default.
    b2 = get_embed_backend(s, prefer="ollama")
    assert b2.model == "nomic-embed-text"


def test_backend_for_keys_cache_by_model(svc, monkeypatch):
    """_backend_for must return DISTINCT backends for corpora with different
    models, not reuse one global backend (which caused the old mismatch error)."""
    from tests.conftest import StubEmbed
    from vecgrep.backend.store.corpora import Corpus

    # Resolve a fake backend per (prefer, model) so we don't hit Ollama.
    def fake_get_embed_backend(settings, prefer=None, model=None):
        return StubEmbed(name=prefer or "ollama", model=model or "default", dim=16)

    monkeypatch.setattr(
        "vecgrep.backend.service.get_embed_backend", fake_get_embed_backend
    )
    # Clear the pre-injected 'auto' stub so resolution goes through our fake.
    svc._backend_cache.clear()

    c_nomic = Corpus(name="a", embed_backend="ollama", embed_model="nomic-embed-text", dim=16)
    c_bge = Corpus(name="b", embed_backend="ollama", embed_model="bge-m3", dim=16)

    b_nomic = svc._backend_for(c_nomic)
    b_bge = svc._backend_for(c_bge)

    assert b_nomic.model == "nomic-embed-text"
    assert b_bge.model == "bge-m3"
    assert b_nomic is not b_bge  # distinct backends, cached separately

# ── test_backend_failover_recovery ──

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
