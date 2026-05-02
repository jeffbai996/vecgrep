from __future__ import annotations

import httpx

from .base import EmbedBackend, EmbedBackendError

# Known dimensions so we don't have to do a probe call to set up the collection.
# If the user picks an unknown model, we fall back to a one-shot probe.
_KNOWN_DIMS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
}


class OllamaBackend(EmbedBackend):
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=timeout)
        self.dim = _KNOWN_DIMS.get(model) or self._probe_dim()

    def _probe_dim(self) -> int:
        vec = self.embed_one("probe")
        return len(vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            try:
                r = self._client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": t},
                )
            except httpx.ConnectError as e:
                raise EmbedBackendError(
                    f"Could not reach Ollama at {self.base_url}. "
                    "Start it with `ollama serve` (or set VECGREP_OLLAMA_URL)."
                ) from e
            except httpx.HTTPError as e:
                raise EmbedBackendError(f"Ollama request failed: {e}") from e

            if r.status_code == 404:
                raise EmbedBackendError(
                    f"Ollama model '{self.model}' is not available. "
                    f"Pull it with `ollama pull {self.model}`."
                )
            if r.status_code >= 400:
                raise EmbedBackendError(
                    f"Ollama returned {r.status_code}: {r.text[:200]}"
                )

            data = r.json()
            if "embedding" not in data:
                raise EmbedBackendError(
                    f"Ollama response missing 'embedding' field: {data}"
                )
            out.append(data["embedding"])
        return out
