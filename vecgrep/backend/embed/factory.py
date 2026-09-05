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


def _resolve_ollama(settings, model: str | None):
    """Primary Ollama if alive, else the configured fallback Ollama if alive,
    else None. Shared by the pinned-ollama and the unpinned/auto paths so a
    corpus pin can never bypass failover. Same model/dim either way, so vectors
    stay corpus-compatible. The fallback is only probed after the primary fails,
    so a healthy primary costs exactly one probe."""
    chosen = model or settings.embed_model
    if _ollama_alive(settings.ollama_url):
        return OllamaBackend(
            settings.ollama_url,
            chosen,
            num_batch=settings.ollama_num_batch,
        )
    fallback = settings.ollama_fallback_url
    if fallback and _ollama_alive(fallback):
        return OllamaBackend(
            fallback,
            chosen,
            num_batch=settings.ollama_num_batch,
        )
    return None


def get_embed_backend(
    settings: Settings, prefer: str | None = None, model: str | None = None
) -> EmbedBackend:
    """Return an embed backend.

    `prefer` lets a corpus pin its backend ('ollama' or 'openai').
    `model` lets a corpus pin its exact embed model, overriding the settings
    default — this is what lets the engine serve corpora embedded with
    DIFFERENT models simultaneously (each corpus queries with its own model)
    instead of forcing one global model and erroring on mismatch.
    Pinned or not, Ollama is the only backend reached automatically. OpenAI is
    metered and embeds in its own vector space, so it is used only when a
    corpus pins it. Every other path fails loudly instead.
    """
    if prefer == "openai":
        if not settings.openai_api_key:
            raise EmbedBackendError(
                "Corpus is pinned to OpenAI but OPENAI_API_KEY is not set."
            )
        return OpenAIBackend(settings.openai_api_key, model or settings.openai_embed_model)

    # Pinned to ollama: still honor primary->fallback. (A corpus pin must NOT
    # bypass failover — that was the bug where a pinned corpus used the dead
    # primary url blindly and never fell over.) Pinned means "don't use OpenAI",
    # so a dead primary + no live fallback raises rather than silently switching
    # embed models (which would break vector compatibility for that corpus).
    if prefer == "ollama":
        ob = _resolve_ollama(settings, model)
        if ob is not None:
            return ob
        raise EmbedBackendError(
            f"Corpus pinned to Ollama but neither primary ({settings.ollama_url}) "
            f"nor fallback ({settings.ollama_fallback_url}) is reachable."
        )

    ob = _resolve_ollama(settings, model)
    if ob is not None:
        return ob

    # Both Ollama endpoints are down. We do NOT reach for OpenAI here, for the
    # reason the pinned branch above already gives: a different provider means a
    # different embed model, and writing those vectors into a corpus embedded
    # with another one silently breaks its vector space. It is also a metered
    # call nobody asked for. OpenAI is reachable only by pinning a corpus to it
    # (`prefer="openai"`), which is a decision someone made on purpose.
    fallback_hint = (
        f" (fallback {settings.ollama_fallback_url} also unreachable)"
        if settings.ollama_fallback_url else ""
    )
    raise EmbedBackendError(
        f"Ollama not reachable at {settings.ollama_url}{fallback_hint}. Start "
        f"it (`ollama serve` and `ollama pull {settings.embed_model}`), or pin "
        f"this corpus to a backend that is up."
    )
