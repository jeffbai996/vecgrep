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


# Chunks per /api/embed request. Big enough that round-trip latency stops
# dominating, small enough that one poison chunk only costs a re-embed of its
# own window on the fallback path rather than the whole document.
_MAX_BATCH = 64


def _is_finite_vector(vec: list[float]) -> bool:
    """True only if every component is a finite number (no NaN/inf)."""
    return bool(vec) and all(math.isfinite(x) for x in vec)


class OllamaBackend(EmbedBackend):
    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        num_batch: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_batch = num_batch
        # Phase-split timeout: a SHORT connect timeout so an unreachable host
        # (asleep / no service) fails over fast instead of hanging the full
        # read budget, but a LONG read timeout so a slow-but-alive embed (big
        # batch, cold model load) still completes. Without this, a dead primary
        # behind a primary->fallback chain stalls every embed up to `timeout`
        # before failover triggers.
        self._client = self._make_client(read=timeout)
        self.dim = _KNOWN_DIMS.get(model) or self._probe_dim()

    def _request_payload(self, input_: str | list[str]) -> dict:
        payload = {"model": self.model, "input": input_, "truncate": True}
        if self.num_batch is not None:
            payload["options"] = {"num_batch": self.num_batch}
        return payload

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
        """Embed a batch, falling back to one-at-a-time only when it fails.

        This used to be a plain per-chunk loop. /api/embed accepts an array, so
        that spent one HTTP round trip per chunk: measured against a live
        bge-m3, the endpoint served 32 inputs in 0.5s (~67/s) while a corpus
        rebuild driven by the loop moved ~5.7/s. The ~12x gap was round-trip
        overhead, not the GPU -- a full rebuild spent a night on work the
        backend could do in an hour.

        The per-chunk path is still the fallback, because it is the only thing
        that isolates a poison chunk (bge-m3 500s on some inputs). Fast path
        when the batch is clean, slow path when it is not.
        """
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), _MAX_BATCH):
            window = texts[i : i + _MAX_BATCH]
            rows = self._embed_batch(window)
            if rows is None:
                rows = [self._embed_one_resilient(t) for t in window]
            out.extend(rows)
        return out

    def _embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        """One request for the whole window. None means "use the slow path".

        Anything short of a clean, correctly-sized, all-finite answer returns
        None rather than guessing. In particular a row count that does not
        match the input count is NEVER zipped onto the inputs: pairing 3
        returned vectors with 4 chunks would attach the wrong embedding to the
        wrong text, which raises no error and only shows up much later as
        nonsense search results. Re-embedding per chunk is cheap next to
        silently corrupting a corpus.
        """
        try:
            r = self._client.post(
                f"{self.base_url}/api/embed",
                json=self._request_payload(texts),
            )
        except httpx.ConnectError as e:
            # An unreachable backend is a whole-backend problem, not a batch
            # quirk — surface it now instead of retrying it len(texts) times.
            raise EmbedBackendError(
                f"Could not reach Ollama at {self.base_url}. "
                "Start it with `ollama serve` (or set VECGREP_OLLAMA_URL)."
            ) from e
        except httpx.HTTPError:
            return None
        if r.status_code != 200:
            # Let the per-chunk path decide: it already distinguishes a fatal
            # 404/4xx from a transient 5xx, so error messages stay identical.
            return None
        try:
            rows = r.json().get("embeddings")
        except ValueError:
            return None
        if not isinstance(rows, list) or len(rows) != len(texts):
            return None
        if not all(_is_finite_vector(v) for v in rows):
            return None
        return rows

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
                    json=self._request_payload(text),
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
