"""Cross-encoder reranking.

Off by default. When enabled, takes the candidate pool from any retrieval
mode (hybrid/vector/bm25) and rescores each (query, chunk) pair with a
cross-encoder. Cross-encoders are slow but much more accurate than the
bi-encoder used for the initial vector retrieval — running them only on
top-50 keeps latency reasonable.

Lazy import: sentence-transformers pulls in torch (~hundreds of MB) and
is gated behind the optional `vecgrep[rerank]` extra. We import it only
when the user actually asks for reranking.
"""
from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

# Round 3 (2026-08-18, 119 gold cases / 26 negatives, docs/STORAGE_RETRIEVAL):
#   base   hit@3 76.3  mrr .681  neg FP 11.5%  +250 ms
#   v2-m3  hit@3 82.8  mrr .702  neg FP  3.8%  +1.4 s
#   large  hit@3 79.6  mrr .697  neg FP  3.8%  +3.5 s
# v2-m3 is the first reranker that does not demote answers the pool already
# had (base lost hit@3 vs unreranked; v2-m3 gains it) and it calibrates the
# negatives. It is NOT the default yet: rerank runs synchronously inside the
# serve request path, and the first v2-m3 load+predict in that process blocked
# the event loop for 5+ minutes on 2026-08-18, long enough for squad-watchdog
# (strikes=1) to kill and restart the server mid-request. Until model loading
# is warmed at startup (or off the event loop), the smaller model stays.
# Override per install with VECGREP_RERANKER=BAAI/bge-reranker-v2-m3.
DEFAULT_RERANKER = os.environ.get("VECGREP_RERANKER", "BAAI/bge-reranker-base")


# How long a search waits for a cold reranker before giving up and returning
# fusion order. Loading a cross-encoder imports torch and, the first time,
# downloads weights -- minutes, not milliseconds. Search runs in FastAPI's sync
# threadpool, capped at 8 (VECGREP_THREAD_POOL_SIZE), so a blocking load does
# not merely slow one request: it holds one of eight threads for the whole load
# and starves every other sync route, health check included. That is how the
# v2-m3 first load took the server down on 2026-08-18 -- squad-watchdog saw the
# timeout and at strikes=1 restarted vecgrep-serve mid-request. Waiting a few
# seconds is fine; waiting minutes is an outage.
RERANK_WAIT_S = float(os.environ.get("VECGREP_RERANK_WAIT_S", "5"))

# A load that fails (no extra installed, no network for the weights) must not
# be retried on every single search -- but it must not disable reranking until
# someone restarts the process either. Retry no more often than this.
_RETRY_AFTER_S = 300.0


class RerankerError(RuntimeError):
    pass


_cache: dict[str, object] = {}
_lock = threading.Lock()
_warming: dict[str, threading.Event] = {}
_failed_at: dict[str, float] = {}


def ensure_warm(model_name: str = DEFAULT_RERANKER) -> threading.Event:
    """Load `model_name` in the background if it is neither loaded nor already
    loading. Returns an Event set once the attempt finishes, successfully or
    not. Idempotent, so calling it per request is free after the first."""
    with _lock:
        if model_name in _cache:
            done = threading.Event()
            done.set()
            return done
        ev = _warming.get(model_name)
        if ev is not None:
            failed = _failed_at.get(model_name)
            if failed is None or (time.monotonic() - failed) < _RETRY_AFTER_S:
                return ev
            # Past the cooldown after a failure: drop the old attempt and retry.
            _failed_at.pop(model_name, None)
        ev = threading.Event()
        _warming[model_name] = ev

    def _run() -> None:
        try:
            _load(model_name)
            logger.info("reranker %s ready", model_name)
        except Exception as exc:
            with _lock:
                _failed_at[model_name] = time.monotonic()
            logger.warning("reranker %s failed to load: %s", model_name, exc)
        finally:
            ev.set()

    threading.Thread(
        target=_run, name="vecgrep-rerank-warm", daemon=True
    ).start()
    return ev


def is_ready(model_name: str = DEFAULT_RERANKER) -> bool:
    """True when the model is loaded and predict() will not block on I/O."""
    with _lock:
        return model_name in _cache


def wait_ready(
    model_name: str = DEFAULT_RERANKER, timeout: float | None = None
) -> bool:
    """Warm if needed, wait a BOUNDED time, and report whether it is usable.

    False is not an error — it means "not yet", and the caller should fall
    back to unreranked order rather than hold a threadpool slot open."""
    if is_ready(model_name):
        return True
    ev = ensure_warm(model_name)
    ev.wait(RERANK_WAIT_S if timeout is None else timeout)
    return is_ready(model_name)


def _load(model_name: str):
    with _lock:
        cached = _cache.get(model_name)
    if cached is not None:
        return cached
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise RerankerError(
            "Reranking requires the 'rerank' extra. "
            "Install with `pip install vecgrep[rerank]`."
        ) from e
    try:
        model = CrossEncoder(model_name)
    except Exception as e:
        raise RerankerError(
            f"Failed to load cross-encoder '{model_name}': {e}"
        ) from e
    with _lock:
        _cache[model_name] = model
    return model


def rerank(
    query: str,
    candidates: list[tuple[str, dict]],
    model_name: str = DEFAULT_RERANKER,
) -> list[tuple[float, dict]]:
    """Score (query, chunk_text) for each candidate. Returns (score, payload)
    pairs sorted descending. Scores are sigmoid-mapped 0..1 for downstream
    display.
    """
    if not candidates:
        return []
    model = _load(model_name)
    pairs = [(query, text) for text, _ in candidates]
    raw = model.predict(pairs)  # numpy array of logits

    # bge-reranker emits raw logits; squashing through sigmoid puts them in
    # 0..1 which is more useful for percentage display than raw values.
    import math

    scored = [
        (1 / (1 + math.exp(-float(s))), payload)
        for s, (_, payload) in zip(raw, candidates)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored
