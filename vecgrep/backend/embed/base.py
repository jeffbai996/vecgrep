"""Embedding backend interface.

Two impls: Ollama (default), OpenAI (fallback). Both expose embed() that
returns a list of float vectors. Backend identity (name + dim) is pinned
into corpus metadata so we reject cross-model mixing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class EmbedBackendError(RuntimeError):
    """Raised when an embedding call fails for a user-actionable reason.

    The message should tell the user what to do (e.g., start Ollama,
    pull the model, set OPENAI_API_KEY).
    """


class EmbedBackend(ABC):
    name: str
    model: str
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def identity(self) -> str:
        return f"{self.name}:{self.model}"
