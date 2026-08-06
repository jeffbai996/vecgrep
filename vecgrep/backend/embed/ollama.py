from __future__ import annotations

import logging
import math

import httpx

from .base import EmbedBackend, EmbedBackendError

logger = logging.getLogger(__name__)

# Known dimensions so we don't have to do a probe call to set up the collection.
# If the user picks an unknown model, we fall back to a one-shot probe.
_KNOWN_DIMS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "bge-m3": 1024,
    "all-minilm": 384,
}


def _is_finite_vector(vec: list[float]) -> bool:
    """True only if every component is a finite number (no NaN/inf)."""
    return bool(vec) and all(math.isfinite(x) for x in vec)


class OllamaBackend(EmbedBackend):
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        # Phase-split timeout: a SHORT connect timeout so an unreachable host
        # (asleep / no service) fails over fast instead of hanging the full
        # read budget, but a LONG read timeout so a slow-but-alive embed (big
        # batch, cold model load) still completes. Without this, a dead primary
        # behind a primary->fallback chain stalls every embed up to `timeout`
        # before failover triggers.
        self._client = self._make_client(read=timeout)
        self.dim = _KNOWN_DIMS.get(model) or self._probe_dim()

    @staticmethod
    def _make_client(connect: float = 2.0, read: float = 60.0) -> httpx.Client:
        """httpx client with a fast connect + generous read timeout."""
        return httpx.Client(
            timeout=httpx.Timeout(read, connect=connect)
        )

    def _probe_dim(self) -> int:
        vec = self.embed_one("probe")
        return len(vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            out.append(self._embed_one_resilient(t))
        return out

    def _embed_one_resilient(self, text: str) -> list[float]:
        """Embed one chunk, tolerating per-chunk NaN failures.

        Some models (notably bge-m3) make Ollama 500 with "unsupported value:
        NaN" on specific inputs, or return a vector containing NaN/inf. A naive
        loop aborts the whole document on the first bad chunk. Instead we retry
        once, then fall back to a zero vector so the chunk is still STORED (BM25
        keeps it keyword-findable) but can never win on cosine similarity —
        degraded to keyword-only rather than silently dropped or falsely ranked.

        Connection/model-not-found errors still hard-fail: those are
        whole-backend problems the user must fix, not per-chunk noise.
        """
        last_reason = ""
        for attempt in range(2):
            try:
                # /api/embed, NOT the legacy /api/embeddings: the legacy
                # endpoint IGNORES `truncate` and returns HTTP 500 ("the input
                # length exceeds the context length") for any chunk longer than
                # the model context. Those 500s then fell through to the
                # zero-vector path below, so oversized chunks were stored but
                # unreachable by semantic search (observed at scale against a
                # local bge-m3 deployment). /api/embed honours truncate and
                # clips the input to the window instead.
                r = self._client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": text, "truncate": True},
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

            # A NaN failure surfaces as a 500 whose body mentions NaN. Treat that
            # (and any other 5xx) as a transient per-chunk fault: retry, then
            # fall back. Non-NaN 4xx (bad request) is a real error — raise it.
            if 400 <= r.status_code < 500:
                raise EmbedBackendError(f"Ollama returned {r.status_code}: {r.text[:200]}")
            if r.status_code >= 500:
                last_reason = f"HTTP {r.status_code}: {r.text[:120]}"
                continue

            data = r.json()
            # /api/embed nests under "embeddings" (a list, one row per input);
            # the legacy endpoint used a flat "embedding". Read both so a
            # mixed-version Ollama cannot silently zero-vector every chunk.
            vec = data.get("embedding")
            if vec is None:
                rows = data.get("embeddings")
                if isinstance(rows, list) and rows:
                    vec = rows[0]
            if vec is None:
                last_reason = f"response missing 'embeddings' field: {str(data)[:120]}"
                continue
            if not _is_finite_vector(vec):
                last_reason = "embedding contained NaN/inf"
                continue
            return vec

        logger.warning(
            "Ollama embed failed for a chunk after retry (%s); using zero vector "
            "so the chunk stays keyword-searchable but never wins on cosine. "
            "Chunk head: %r",
            last_reason,
            text[:80],
        )
        return [0.0] * self.dim
