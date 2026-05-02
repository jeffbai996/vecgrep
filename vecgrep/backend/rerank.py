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

DEFAULT_RERANKER = "BAAI/bge-reranker-base"


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
