"""Result assembly: near-duplicate collapse + diversity-aware selection.

The retrieval core (hybrid vector+BM25, fusion, rerank) stays untouched — this
module improves what happens AFTER scoring: which of the scored candidates
actually make it into the returned set.

Two stages, run by service.search():

1. `dedup_near_duplicates` — the hard collapse. Extends the old span-overlap
   dedup with a text-clone check, so the transcript-native duplicate class
   (the same message body re-surfacing at a different char span: bot alert
   spam, quoted replies, re-posted lines) collapses too. Same-source only —
   the same text in two different files is evidence of propagation, not a dup.

2. `mmr_select` — greedy maximal-marginal-relevance selection of top_k:
   `lambda * relevance - (1 - lambda) * max_similarity_to_selected`, with a
   near-clone exclusion: a candidate >= EXCLUDE_SIM similar to something
   already selected is pure redundancy and is skipped while distinct
   candidates remain (it can still backfill if the pool runs dry). On corpora
   with no near-duplicates every pairwise similarity is ~0 and selection
   degrades to plain score order — non-transcript corpora are unaffected.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover - import cycle guard (service imports us)
    from .service import SearchResult

# Diversity/relevance trade-off for MMR. 0.7 keeps relevance dominant; the
# diversity term mostly breaks ties between near-equal-scored siblings.
MMR_LAMBDA = 0.7
# Similarity at/above which a candidate is treated as a clone of an already
# selected result and skipped during MMR (still eligible as backfill).
EXCLUDE_SIM = 0.92
# Text similarity at/above which the HARD dedup collapses same-source chunks
# even when their spans don't overlap.
CLONE_THRESHOLD = 0.92
# Chunks shorter than this (normalized) carry no usable clone signal — a tiny
# text matching another tiny text is coincidence, not duplication.
MIN_CLONE_CHARS = 40
# Two same-source chunks sharing a verbatim run of this many tokens (and at
# least MIN_CLONE_CHARS chars) contain the same message — treat as near-clone.
SHINGLE_TOKENS = 8
SHINGLE_SIM = 0.95

# ── result budget (full tier + stub tail) ────────────────────────────────────
# The assistant gets breadth without a blown context: the head of the ranking
# keeps context windows ("full"), the tail degrades to one-line stubs emitted
# until a token ceiling. 10 + 90 = 100 (Jeff 2026-07-05, up from 8+72=80).
# max_total is a CAP, not a target — token_ceiling stays the real limit, so a
# dense result set legitimately returns fewer stubs.
DEFAULT_FULL_K = 10
DEFAULT_MAX_TOTAL = 100
DEFAULT_STUB_TOKEN_CEILING = 4000
_SNIPPET_CHARS = 160
# Cheap deterministic token estimate (~4 chars/token for EN/code, safely
# conservative for CJK) — a real tokenizer is a heavy dep for a budget knob.
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class ResultStub:
    """A one-line result reference: enough to spot a pattern and decide to
    expand, cheap enough to return by the dozen. NO context windows — the
    assistant re-queries the full window via chunk_id (`/api/chunk`, MCP
    `get_chunk`) when a stub looks interesting."""

    chunk_id: str
    corpus: str
    source_id: str
    doc_timestamp: float | None
    snippet: str
    score: float
    similarity_pct: float


def estimate_tokens(text: str) -> int:
    """Deterministic ~4-chars/token estimate for budget accounting."""
    return len(text) // _CHARS_PER_TOKEN


def _stub_of(r: "SearchResult") -> ResultStub:
    snippet = " ".join(r.chunk.split())[:_SNIPPET_CHARS]
    return ResultStub(
        chunk_id=r.chunk_id,
        corpus=r.corpus,
        source_id=r.source_id,
        doc_timestamp=r.doc_timestamp,
        snippet=snippet,
        score=r.score,
        similarity_pct=r.similarity_pct,
    )


def split_full_and_stubs(
    results: list["SearchResult"],
    full_k: int = DEFAULT_FULL_K,
    max_total: int = DEFAULT_MAX_TOTAL,
    token_ceiling: int = DEFAULT_STUB_TOKEN_CEILING,
) -> tuple[list["SearchResult"], list[ResultStub]]:
    """Split a ranked result list into (full head, stub tail).

    The head keeps full context; the tail is stubbed and emitted in rank
    order until either max_total results or the stub token ceiling is
    reached — whichever bites first. The ceiling is checked after appending,
    so it can be crossed by at most one stub (never under-fills on a
    generous budget).
    """
    full = results[:full_k]
    stubs: list[ResultStub] = []
    spent = 0
    for r in results[full_k:max_total]:
        s = _stub_of(r)
        stubs.append(s)
        spent += estimate_tokens(s.snippet + s.source_id)
        if spent >= token_ceiling:
            break
    return full, stubs


def _normalized(r: "SearchResult") -> str:
    """Whitespace-collapsed lowercase chunk text, cached on the result."""
    cached = getattr(r, "_asm_norm", None)
    if cached is None:
        cached = " ".join(r.chunk.lower().split())
        r._asm_norm = cached  # type: ignore[attr-defined]
    return cached


def _shingles(r: "SearchResult") -> frozenset:
    """Token 8-gram shingles of the normalized text (cached). Only shingles
    long enough to be distinctive (>= MIN_CLONE_CHARS chars) are kept."""
    cached = getattr(r, "_asm_shingles", None)
    if cached is None:
        toks = re.findall(r"\S+", _normalized(r))
        grams = set()
        for i in range(max(0, len(toks) - SHINGLE_TOKENS + 1)):
            g = " ".join(toks[i : i + SHINGLE_TOKENS])
            if len(g) >= MIN_CLONE_CHARS:
                grams.add(g)
        cached = frozenset(grams)
        r._asm_shingles = cached  # type: ignore[attr-defined]
    return cached


def span_overlap_ratio(a: "SearchResult", b: "SearchResult") -> float:
    """Char-span overlap as a fraction of the shorter span (same source only)."""
    lo = max(a.chunk_start, b.chunk_start)
    hi = min(a.chunk_end, b.chunk_end)
    shorter = min(a.chunk_end - a.chunk_start, b.chunk_end - b.chunk_start)
    if shorter <= 0:
        return 0.0
    return max(0, hi - lo) / shorter


def _text_similarity(a: "SearchResult", b: "SearchResult") -> float:
    """difflib ratio on normalized text; 0.0 when either side is too short to
    judge. quick_ratio bounds the real ratio from above, so use it as a cheap
    bail-out before the quadratic comparison."""
    na, nb = _normalized(a), _normalized(b)
    if len(na) < MIN_CLONE_CHARS or len(nb) < MIN_CLONE_CHARS:
        return 0.0
    sm = difflib.SequenceMatcher(None, na, nb)
    if sm.real_quick_ratio() < 0.5 or sm.quick_ratio() < 0.5:
        return 0.0
    return sm.ratio()


def pair_similarity(a: "SearchResult", b: "SearchResult") -> float:
    """How redundant is b given a (symmetric)? 0.0 across sources — different
    files are never duplicates of each other. Within a source: the max of
    span overlap, text-clone similarity, and the shared-verbatim-message
    signal (a shared long shingle ~= the same message body re-surfacing)."""
    if a.corpus != b.corpus or a.source_id != b.source_id:
        return 0.0
    sim = span_overlap_ratio(a, b)
    if sim < SHINGLE_SIM and _shingles(a) & _shingles(b):
        sim = max(sim, SHINGLE_SIM)
    if sim < 1.0:
        sim = max(sim, _text_similarity(a, b))
    return sim


def dedup_near_duplicates(
    results: list["SearchResult"],
    min_overlap: float = 0.5,
    clone_threshold: float = CLONE_THRESHOLD,
) -> list["SearchResult"]:
    """Hard collapse of same-source near-duplicates, keeping the higher score.

    Collides on char-span overlap >= min_overlap of the shorter span (the
    v0.7.0 behavior) OR normalized text similarity >= clone_threshold (the
    repeated-message class the span check can't see). Preserves input order
    otherwise — callers sort afterwards.
    """
    kept: list[SearchResult] = []
    for r in results:
        dup_idx = None
        for i, k in enumerate(kept):
            if k.corpus != r.corpus or k.source_id != r.source_id:
                continue
            if span_overlap_ratio(k, r) >= min_overlap:
                dup_idx = i
                break
            if _text_similarity(k, r) >= clone_threshold:
                dup_idx = i
                break
        if dup_idx is None:
            kept.append(r)
        elif r.score > kept[dup_idx].score:
            kept[dup_idx] = r
    return kept


def mmr_select(
    results: list["SearchResult"],
    k: int,
    lambda_: float = MMR_LAMBDA,
    exclude_sim: float = EXCLUDE_SIM,
    key: Callable[["SearchResult"], float] | None = None,
) -> list["SearchResult"]:
    """Greedy MMR selection of up to k results.

    Relevance is `key(r)` (default: fused score) min-max normalized within the
    candidate pool. Each round picks argmax(lambda * rel - (1-lambda) *
    max_sim_to_selected), skipping candidates whose max similarity to the
    selected set is >= exclude_sim (near-clones add nothing) while distinct
    candidates remain; if only near-clones are left and slots remain, the
    best-scoring near-clones backfill so k is still honored.
    """
    if not results or k <= 0:
        return []
    key = key or (lambda r: r.score)
    scores = [key(r) for r in results]
    lo, hi = min(scores), max(scores)
    spread = (hi - lo) or 1.0
    rel = {id(r): (key(r) - lo) / spread for r in results}

    selected: list[SearchResult] = []
    remaining = list(results)
    # max similarity of each remaining candidate to the selected set,
    # updated incrementally as we select (avoids recomputing pairs).
    max_sim: dict[int, float] = {id(r): 0.0 for r in results}

    while remaining and len(selected) < k:
        best = best_mmr = None
        best_clone = best_clone_mmr = None
        for r in remaining:
            s = max_sim[id(r)]
            mmr = lambda_ * rel[id(r)] - (1.0 - lambda_) * s
            if s >= exclude_sim:
                if best_clone is None or mmr > best_clone_mmr:
                    best_clone, best_clone_mmr = r, mmr
            elif best is None or mmr > best_mmr:
                best, best_mmr = r, mmr
        pick = best if best is not None else best_clone
        selected.append(pick)
        remaining.remove(pick)
        for r in remaining:
            sim = pair_similarity(pick, r)
            if sim > max_sim[id(r)]:
                max_sim[id(r)] = sim
    return selected
