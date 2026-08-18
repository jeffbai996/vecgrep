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

import os

# Round 3 (2026-08-18, 119 gold cases / 26 negatives, docs/STORAGE_RETRIEVAL):
#   base   hit@3 76.3  mrr .681  neg FP 11.5%  +250 ms
#   v2-m3  hit@3 82.8  mrr .702  neg FP  3.8%  +1.4 s
#   large  hit@3 79.6  mrr .697  neg FP  3.8%  +3.5 s
# v2-m3 is the first reranker that does not demote answers the pool already
# had (base lost hit@3 vs unreranked; v2-m3 gains it) and it calibrates the
# negatives. Override per install with VECGREP_RERANKER.
DEFAULT_RERANKER = os.environ.get("VECGREP_RERANKER", "BAAI/bge-reranker-v2-m3")


class RerankerError(RuntimeError):
    pass


_cache: dict[str, object] = {}


def _load(model_name: str):
    if model_name in _cache:
        return _cache[model_name]
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
