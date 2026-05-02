"""Pick an embedding backend based on settings.

Default: Ollama. If Ollama is unreachable AND OPENAI_API_KEY is set,
fall back to OpenAI. If Ollama is unreachable and no key is set, raise
with a clear message.
"""
from __future__ import annotations

import httpx

from ..config import Settings
from .base import EmbedBackend, EmbedBackendError
from .ollama import OllamaBackend
from .openai import OpenAIBackend


def _ollama_alive(base_url: str) -> bool:
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=2.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def get_embed_backend(settings: Settings, prefer: str | None = None) -> EmbedBackend:
    """Return an embed backend.

    `prefer` lets a corpus pin its backend ('ollama' or 'openai').
    If pinned, we honor it or fail loudly.
    If unpinned, prefer Ollama, fall back to OpenAI when key is set.
    """
    if prefer == "openai":
        if not settings.openai_api_key:
            raise EmbedBackendError(
                "Corpus is pinned to OpenAI but OPENAI_API_KEY is not set."
            )
        return OpenAIBackend(settings.openai_api_key, settings.openai_embed_model)

    if prefer == "ollama":
        return OllamaBackend(settings.ollama_url, settings.embed_model)

    if _ollama_alive(settings.ollama_url):
        return OllamaBackend(settings.ollama_url, settings.embed_model)

    if settings.openai_api_key:
        return OpenAIBackend(settings.openai_api_key, settings.openai_embed_model)

    raise EmbedBackendError(
        f"Ollama not reachable at {settings.ollama_url} and OPENAI_API_KEY is not set. "
        f"Either start Ollama (`ollama serve` and `ollama pull {settings.embed_model}`) "
        "or export OPENAI_API_KEY to use OpenAI as a fallback."
    )
