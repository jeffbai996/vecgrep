"""Per-corpus embed-backend resolution (mixed-model serving).

get_embed_backend now takes an explicit `model`, and _backend_for resolves it
from the corpus's pinned model rather than the global env. This is what lets one
service query corpora embedded with different models without the old hard-error.
"""
from __future__ import annotations

from vecgrep.backend.config import Settings
from vecgrep.backend.embed.factory import get_embed_backend


def test_get_embed_backend_honors_explicit_model():
    s = Settings(ollama_url="http://localhost:11434", embed_model="nomic-embed-text")
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
