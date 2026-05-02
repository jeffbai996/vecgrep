from __future__ import annotations

from .base import EmbedBackend, EmbedBackendError

# OpenAI's text-embedding-3-small returns 1536 by default.
_KNOWN_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIBackend(EmbedBackend):
    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise EmbedBackendError(
                "openai package not installed. Install with `pip install vecgrep[openai]`."
            ) from e

        self.model = model
        self._client = OpenAI(api_key=api_key)
        self.dim = _KNOWN_DIMS.get(model) or self._probe_dim()

    def _probe_dim(self) -> int:
        return len(self.embed_one("probe"))

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = self._client.embeddings.create(model=self.model, input=texts)
        except Exception as e:
            raise EmbedBackendError(f"OpenAI embeddings request failed: {e}") from e
        return [item.embedding for item in resp.data]
