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
