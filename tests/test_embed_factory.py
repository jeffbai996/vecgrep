"""Embed backend selection — primary Ollama, then a SECOND Ollama fallback,
then OpenAI.

Adds an ollama_fallback_url so a deployment can run a primary Ollama (e.g. a
beefy GPU box) and transparently fall back to a secondary Ollama (e.g. the
local host) when the primary is unreachable — without forcing OpenAI. The
liveness probe is the same `_ollama_alive` check already used for primary.
"""
from __future__ import annotations

import pytest

from vecgrep.backend.config import Settings
from vecgrep.backend.embed import factory
from vecgrep.backend.embed.ollama import OllamaBackend
from vecgrep.backend.embed.openai import OpenAIBackend


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


def test_primary_dead_falls_back_to_secondary_ollama(monkeypatch):
    s = _settings(ollama_url="http://primary:11434",
                  ollama_fallback_url="http://fallback:11434")
    # primary dead, fallback alive
    monkeypatch.setattr(factory, "_ollama_alive", lambda url: url == "http://fallback:11434")
    b = factory.get_embed_backend(s)
    assert isinstance(b, OllamaBackend)
    assert b.base_url == "http://fallback:11434"


def test_both_ollama_dead_falls_to_openai_when_key_set(monkeypatch):
    # Stub OpenAIBackend so the test doesn't need the openai package installed —
    # we're asserting the SELECTION (OpenAI chosen), not the client itself.
    sentinel = object()
    monkeypatch.setattr(factory, "OpenAIBackend", lambda *a, **k: sentinel)
    s = _settings(ollama_url="http://primary:11434",
                  ollama_fallback_url="http://fallback:11434",
                  openai_api_key="sk-test")
    monkeypatch.setattr(factory, "_ollama_alive", lambda url: False)
    assert factory.get_embed_backend(s) is sentinel


def test_both_ollama_dead_no_key_raises(monkeypatch):
    s = _settings(ollama_url="http://primary:11434",
                  ollama_fallback_url="http://fallback:11434")
    monkeypatch.setattr(factory, "_ollama_alive", lambda url: False)
    with pytest.raises(factory.EmbedBackendError):
        factory.get_embed_backend(s)


def test_no_fallback_url_preserves_old_behavior(monkeypatch):
    # Unset fallback → behave exactly as before: primary probe, then OpenAI.
    sentinel = object()
    monkeypatch.setattr(factory, "OpenAIBackend", lambda *a, **k: sentinel)
    s = _settings(ollama_url="http://primary:11434", ollama_fallback_url=None,
                  openai_api_key="sk-test")
    monkeypatch.setattr(factory, "_ollama_alive", lambda url: False)
    assert factory.get_embed_backend(s) is sentinel


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
