"""High-level service: index and search.

Sits between the API/CLI and the store/embed/adapter layers. Owns the
corpus lifecycle so callers don't have to coordinate four subsystems.
"""
from __future__ import annotations

import fnmatch
import hashlib
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import Settings, get_settings
from .embed import EmbedBackend, EmbedBackendError, get_embed_backend
from .embed.cache import CachedBackend, EmbedCache
from .ingestion.adapters import (
    AdapterError,
    Document,
    detect_adapter,
)
from .ingestion.chunkers import (
    Chunker,
    FixedTokenChunker,
    MarkdownSectionChunker,
    SentenceWindowChunker,
)
from .store import (
    BM25Store,
    Corpus,
    CorpusError,
    CorpusRegistry,
    EPHEMERAL_NAME,
    QdrantStore,
    StoredHit,
)


CHUNKERS: dict[str, type[Chunker]] = {
    "sentence_window": SentenceWindowChunker,
    "fixed_token": FixedTokenChunker,
    "markdown_section": MarkdownSectionChunker,
}

SearchMode = Literal["hybrid", "vector", "bm25"]
DEFAULT_MODE: SearchMode = "hybrid"

# Reciprocal Rank Fusion constant. 60 is the canonical value from the
# original RRF paper; we expose it here as a single knob.
RRF_K = 60

# BM25 weight in hybrid fusion. >1 boosts literal-keyword hits over
# semantic-only matches, which matters because nomic-embed-text floors
# around 70-75% similarity for *any* English query — the vector retriever
# almost always returns a noisy top-50 even when nothing matches
# semantically. Without a BM25 boost, that noise crowds out genuine
# keyword hits in the fused ranking. 1.5 chosen empirically: enough to
# float exact-match hits to the top of short queries, not so much that
# it dominates long conceptual queries where vector should win.
# Override via env var VECGREP_BM25_WEIGHT.
BM25_WEIGHT = float(os.environ.get("VECGREP_BM25_WEIGHT", "1.5"))

# Floor + headroom for displaying BM25-only hit pct.
#
# BM25 scores are unbounded positive numbers and corpus-relative, so the raw
# value can't map directly to a meaningful percentage. We rescale within the
# result set: the strongest BM25 hit for this query reads at BM25_DISPLAY_TOP,
# weaker hits taper toward BM25_DISPLAY_FLOOR. The display is "rank-relative
# confidence", not absolute. Underlying ranking uses raw RRF scores and is
# unaffected.
#
# Calibration matches the cosine sigmoid: floor at 25% (visible but clearly
# weak), top at 90% (strong but not "certain"). Anything below 25% gets
# clipped — if BM25 didn't find it strongly, vector probably should be the
# voice that speaks.
BM25_DISPLAY_FLOOR = 25.0
BM25_DISPLAY_TOP = 90.0

# How many candidates each retriever returns before fusion. Larger pool
# = better recall, marginal cost. 50 is a good default for small corpora.
CANDIDATE_POOL = 50

# Vector noise floor. The vector retriever returns a full top-50 even when
# nothing matches semantically -- those sub-noise hits then flood RRF. Drop any
# vector hit whose cosine sits MARGIN below the model's calibration center,
# before fusion. The margin is deliberately gentle (0.10): for bge-m3
# (center 0.55) that floors at ~0.45, which displays around 8% -- clearly junk,
# nothing a user would want -- while leaving the real signal band untouched.
# Raise VECGREP_COSINE_FLOOR_MARGIN to filter harder, set it high (>=1) to
# disable. (Conservative on purpose: a too-aggressive floor hides weak-but-real
# hits; the calibration + optional rerank already de-emphasize them.)
COSINE_FLOOR_MARGIN = float(os.environ.get("VECGREP_COSINE_FLOOR_MARGIN", "0.10"))

# Stable namespace for chunk IDs. Lets BM25 and Qdrant reference the same
# chunks deterministically — re-indexing a source with the same content
# regenerates the same IDs.
_ID_NAMESPACE = uuid.UUID("3a7d9e5f-0c1b-4a2e-9f4d-abcdef000001")

_SECONDS_PER_DAY = 86400.0


def _recency_factor(doc_ts: float | None, half_life_days: float | None, now: float) -> float:
    """Multiplier in (0, 1] applied to a hit's fused score for recency decay.

    `0.5 ** (age_days / half_life)`: a chunk one half-life old scores as if
    half as relevant, two half-lives as a quarter, etc. Returns 1.0 (no decay)
    when the corpus has no half-life configured or the chunk has no timestamp —
    so undated content is never penalized, only de-prioritized relative to
    dated-and-fresh content. Future-dated chunks (clock skew) clamp to 1.0.
    """
    if not half_life_days or half_life_days <= 0 or doc_ts is None:
        return 1.0
    age_days = max(0.0, (now - doc_ts) / _SECONDS_PER_DAY)
    return 0.5 ** (age_days / half_life_days)


@dataclass
class SearchResult:
    score: float
    similarity_pct: float
    chunk: str
    chunk_start: int
    chunk_end: int
    context_before: str
    context_after: str
    source_id: str
    corpus: str
    metadata: dict
    # Deterministic chunk id (uuid5 of corpus|source_id|chunk_index). Used by
    # the /api/chunk endpoint to fetch expanded context on demand.
    chunk_id: str
    # Which retrievers placed this result. "vector", "bm25", or both.
    matched_by: list[str]
    # Per-retriever score breakdown — populated when --explain is on.
    # Empty dict otherwise. Keys: vector_cosine, vector_rank, bm25_score,
    # bm25_rank, rrf, rerank_score (when reranked).
    explain: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.explain is None:
            self.explain = {}


class VecgrepService:
    def __init__(self, settings: Settings | None = None, ephemeral: bool = False) -> None:
        self.settings = settings or get_settings()
        self.ephemeral = ephemeral
        self.registry = CorpusRegistry(self.settings.corpora_file)
        self.store = QdrantStore(
            None if ephemeral else self.settings.qdrant_path,
            url=None if ephemeral else self.settings.qdrant_url,
        )
        self.bm25 = BM25Store(None if ephemeral else self.settings.home / "bm25")
        self._backend_cache: dict[str, EmbedBackend] = {}
        # Embedding cache lives on disk except in ephemeral mode. Wrapping
        # is opt-in per backend in _backend_for() so tests / mocks can
        # bypass it by sticking a backend directly into _backend_cache.
        self._embed_cache: EmbedCache | None = (
            None if ephemeral else EmbedCache(self.settings.home / "embed_cache.db")
        )

    # ----- backend resolution ---------------------------------------------------
    def _backend_for(self, corpus: Corpus | None) -> EmbedBackend:
        prefer = corpus.embed_backend if corpus else None
        # Resolve to the corpus's OWN pinned model, not the global env model.
        # This is what lets one running service query corpora embedded with
        # different models — each gets its own backend, keyed by (backend,model).
        model = corpus.embed_model if corpus else None
        cache_key = f"{prefer}:{model}" if (prefer and model) else (prefer or "auto")
        if cache_key not in self._backend_cache:
            # Reuse an already-resolved 'auto' backend if it happens to match
            # the corpus's pinned backend AND model — avoids a redundant live
            # resolve (and lets tests inject just one mock).
            auto = self._backend_cache.get("auto")
            if auto is not None and auto.name == prefer and (model is None or auto.model == model):
                self._backend_cache[cache_key] = auto
            else:
                raw = get_embed_backend(self.settings, prefer=prefer, model=model)
                self._backend_cache[cache_key] = (
                    CachedBackend(raw, self._embed_cache) if self._embed_cache else raw
                )
        backend = self._backend_cache[cache_key]
        # Dim is the only hard invariant left: a model mismatch can't happen now
        # (we resolved by the corpus's model), but a dim mismatch would mean the
        # stored vectors are incompatible — recreate is the only fix.
        if corpus and backend.dim != corpus.dim:
            raise EmbedBackendError(
                f"Corpus '{corpus.name}' has dim={corpus.dim} but backend "
                f"{backend.name}/{backend.model} produces dim={backend.dim}. "
                "Recreate the corpus."
            )
        return backend

    def _cache_key_for(self, corpus: Corpus | None) -> str:
        prefer = corpus.embed_backend if corpus else None
        model = corpus.embed_model if corpus else None
        return f"{prefer}:{model}" if (prefer and model) else (prefer or "auto")

    def _embed_query_with_failover(self, corpus: Corpus | None, query: str) -> list[float]:
        """Embed a query, recovering from a backend that died mid-session.

        The resolved backend is cached for the life of the service. If its
        embedder dies (the primary Ollama goes down while we're running), the
        cached object keeps raising. Catch that once, EVICT the stale cache
        entry, and re-resolve via get_embed_backend — which re-runs the
        primary→fallback probe, so a downed primary transparently fails over to
        the secondary Ollama without a service restart. One retry only (a dead
        re-resolve re-raises rather than looping)."""
        backend = self._backend_for(corpus)
        try:
            return backend.embed_one(query)
        except EmbedBackendError:
            # Drop the dead backend so _backend_for re-resolves from scratch
            # (re-probing primary→fallback). Clearing 'auto' too forces a true
            # re-resolve rather than reusing the same dead object.
            key = self._cache_key_for(corpus)
            self._backend_cache.pop(key, None)
            self._backend_cache.pop("auto", None)
            backend = self._backend_for(corpus)
            return backend.embed_one(query)

    # ----- chunkers -------------------------------------------------------------
    @staticmethod
    def chunker(name: str) -> Chunker:
        if name not in CHUNKERS:
            raise CorpusError(f"Unknown chunker: {name}")
        return CHUNKERS[name]()

    # ----- indexing -------------------------------------------------------------
    def index(
        self,
        source: str,
        corpus_name: str,
        chunker_name: str = "sentence_window",
        force: bool = False,
        include: str | None = None,
    ) -> tuple[int, int, int]:
        """Index a source into a corpus. Returns (docs, chunks, skipped).

        Incremental: per-document content hashes are persisted in the corpus
        metadata. A second index() call against the same source skips
        embedding when the content hash hasn't changed. Pass force=True to
        re-embed unconditionally (e.g. after a chunker change you want to
        replay).

        `include` is an optional filename glob applied when `source` is a
        directory — e.g. `*.md` to index only markdown and skip sibling raws.
        """
        if corpus_name == EPHEMERAL_NAME and not self.ephemeral:
            raise CorpusError("Use --ephemeral to write to the ephemeral corpus.")

        # Directories don't have a single adapter — they walk and dispatch
        # per-file in _expand. Only resolve an adapter for non-directory sources.
        is_dir = (
            not source.startswith(("http://", "https://"))
            and Path(source).is_dir()
        )
        adapter = None if is_dir else detect_adapter(source)()

        # Resolve corpus or create on the fly using current backend.
        if self.registry.has(corpus_name):
            corpus = self.registry.get(corpus_name)
            if corpus.chunker != chunker_name:
                raise CorpusError(
                    f"Corpus '{corpus_name}' uses chunker '{corpus.chunker}', "
                    f"requested '{chunker_name}'. Stay consistent or recreate the corpus."
                )
        else:
            backend = self._backend_for(None)
            corpus = Corpus(
                name=corpus_name,
                embed_backend=backend.name,
                embed_model=backend.model,
                dim=backend.dim,
                chunker=chunker_name,
                created_at=time.time(),
                updated_at=time.time(),
            )
            self.store.ensure_collection(_collection_for(corpus_name), corpus.dim)

        backend = self._backend_for(corpus)
        chunker = self.chunker(chunker_name)
        collection = _collection_for(corpus_name)

        # --force should also bust the embedding cache for these texts;
        # otherwise we just re-write identical vectors. The wrapped backend
        # has a `bypass` flag we toggle for the duration of this call.
        prev_bypass = getattr(backend, "bypass", None)
        if force and hasattr(backend, "bypass"):
            backend.bypass = True

        total_docs = 0
        total_chunks = 0
        # When a source is being re-indexed, subtract its old chunk count so
        # the corpus total stays accurate after we replace it.
        chunks_freed = 0
        skipped = 0
        for doc in _expand(source, adapter, include=include):
            chunks = chunker.chunk(doc.text)
            if not chunks:
                continue

            doc_hash = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
            if not force and corpus.source_hashes.get(doc.source_id) == doc_hash:
                # Already indexed at this exact content — skip embed call.
                skipped += 1
                continue

            chunks_freed += _count_chunks_for_source(self.bm25, corpus_name, doc.source_id)

            # Wipe any prior version of this source so re-indexing is idempotent.
            self.store.delete_by_source(collection, doc.source_id)
            self.bm25.delete_by_source(corpus_name, doc.source_id)

            ids = [_chunk_id(corpus_name, doc.source_id, c.index) for c in chunks]
            vectors = backend.embed([c.text for c in chunks])
            payloads = [
                {
                    "corpus": corpus_name,
                    "source_id": doc.source_id,
                    # Store full source text on every chunk so any hit can
                    # reconstruct surrounding context without a second lookup.
                    # Storage overhead is acceptable at MVP scale.
                    "source_text": doc.text,
                    "chunk_index": c.index,
                    "chunk_start": c.start,
                    "chunk_end": c.end,
                    "text": c.text,
                    "metadata": doc.metadata,
                    # Document's own date (epoch seconds) when discoverable.
                    # Powers optional recency decay at search time. None is fine.
                    "doc_timestamp": doc.timestamp,
                }
                for c in chunks
            ]
            self.store.ensure_collection(collection, corpus.dim)
            self.store.upsert(collection, ids, vectors, payloads)
            self.bm25.upsert(corpus_name, ids, [c.text for c in chunks], payloads)

            total_docs += 1
            total_chunks += len(chunks)
            if doc.source_id not in corpus.sources:
                corpus.sources.append(doc.source_id)
            corpus.source_hashes[doc.source_id] = doc_hash

        corpus.doc_count = len(corpus.sources)
        corpus.chunk_count = corpus.chunk_count - chunks_freed + total_chunks
        corpus.updated_at = time.time()
        # Persist via upsert (reload-merge-save) so a concurrent writer's other
        # corpora aren't clobbered. Ephemeral runs only touch the in-memory map.
        if self.ephemeral:
            self.registry._corpora[corpus.name] = corpus
        else:
            self.registry.upsert(corpus)

        # Restore the bypass flag for subsequent calls (other corpora etc.)
        if prev_bypass is not None and hasattr(backend, "bypass"):
            backend.bypass = prev_bypass

        return total_docs, total_chunks, skipped

    # ----- search ---------------------------------------------------------------
    def search(
        self,
        query: str,
        corpus_name: str | None = None,
        top_k: int | None = None,
        mode: SearchMode = DEFAULT_MODE,
        rerank: bool = False,
        rerank_model: str | None = None,
        filters: list[str] | None = None,
        explain: bool = False,
        include_superseded: bool = False,
    ) -> list[SearchResult]:
        top_k = top_k or self.settings.default_top_k
        if corpus_name:
            corpora = [self.registry.get(corpus_name)]
        else:
            corpora = self.registry.list()
        if not corpora:
            return []

        # Pull a wider pool than top_k whenever post-retrieval steps can shrink
        # the set — filtering, reranking, OR dedup — so we don't return fewer
        # than top_k results just because the top hits got filtered/collapsed.
        per_corpus_k = CANDIDATE_POOL

        results: list[SearchResult] = []
        for c in corpora:
            results.extend(self._search_one(c, query, per_corpus_k, mode, explain=explain))

        # Default to active-only retrieval (write-tool status schema): a
        # superseded version never surfaces as current truth. Caller opts out
        # with include_superseded=True (audit trail) or by passing its own
        # meta.status= filter explicitly. Legacy chunks (no status) stay visible
        # via default_active back-compat in _passes_filters.
        filters = list(filters or [])
        has_explicit_status = any(f.startswith("meta.status=") for f in filters)
        if not include_superseded and not has_explicit_status:
            filters.append("meta.status=active")
        if filters:
            results = [r for r in results if _passes_filters(r, filters, default_active=True)]

        # Collapse near-duplicate overlapping chunks from the same source before
        # truncating, so adjacent sentence-windows don't each eat a top_k slot.
        results = _dedup_overlapping(results)

        if rerank:
            results = self._apply_rerank(query, results, top_k, rerank_model, explain=explain)
        else:
            # Display order matches the displayed similarity_pct so the user
            # can trust their eyes (a higher % is always above a lower %).
            # `r.score` is the underlying RRF fused score — used for selection
            # of the top_k pool, not for visible ranking.
            results.sort(key=lambda r: r.score, reverse=True)
            results = results[:top_k]
            results.sort(key=lambda r: r.similarity_pct, reverse=True)
        return results

    def _apply_rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
        model_name: str | None,
        explain: bool = False,
    ) -> list[SearchResult]:
        from .rerank import DEFAULT_RERANKER, rerank as _rerank

        if not candidates:
            return []
        # rerank() takes (text, payload-ish) pairs. We pass each candidate's
        # chunk text + a dict that lets us reconstruct the SearchResult.
        pairs = [(c.chunk, c) for c in candidates]
        scored = _rerank(query, pairs, model_name or DEFAULT_RERANKER)
        out: list[SearchResult] = []
        for score, original in scored[:top_k]:
            r: SearchResult = original  # type: ignore[assignment]
            # The cross-encoder score IS the canonical relevance signal when
            # reranking is on: it's the single number that best approximates
            # P(relevant), computed identically for every hit regardless of
            # which retriever found it. So the displayed pct comes from it via
            # one calibrated sigmoid (_rerank_to_pct), replacing the per-hit
            # cosine/BM25 mix. The earlier "everything lands at 50%" problem is
            # solved by the steep, properly-centered calibration (center 0.57,
            # slope 35) instead of the naive score*100. Raw rerank score is
            # kept in explain for inspection; sort order is the rerank order.
            r.similarity_pct = _rerank_to_pct(float(score))
            r.explain = {**(r.explain or {}), "rerank_score": float(score)}
            if "rerank" not in r.matched_by:
                r.matched_by = [*r.matched_by, "rerank"]
            out.append(r)
        # `scored` is sorted by reranker score desc, and we appended in that
        # order -- so `out` already reflects the reranked ordering. Do NOT
        # re-sort by similarity_pct here (that would undo the rerank).
        return out

    def _search_one(
        self,
        corpus: Corpus,
        query: str,
        top_k: int,
        mode: SearchMode,
        explain: bool = False,
    ) -> list[SearchResult]:
        collection = _collection_for(corpus.name)

        vector_hits: list[StoredHit] = []
        bm25_hits: list[tuple[str, float, dict]] = []

        if mode in ("hybrid", "vector"):
            # Use the failover-aware embed so a backend that died mid-session
            # (primary Ollama down) re-resolves to the fallback instead of
            # raising forever on a stale cached backend.
            qv = self._embed_query_with_failover(corpus, query)
            vector_hits = self.store.search(collection, qv, top_k=CANDIDATE_POOL)
            # Drop sub-noise vector hits before they reach fusion (see
            # COSINE_FLOOR_MARGIN). Gentle by default; keeps the real signal band.
            floor = _cosine_floor(corpus.embed_model)
            if floor > 0.0:
                vector_hits = [h for h in vector_hits if h.score >= floor]

        if mode in ("hybrid", "bm25"):
            bm25_hits = self.bm25.search(corpus.name, query, top_k=CANDIDATE_POOL)

        if mode == "vector":
            out: list[SearchResult] = []
            for rank, h in enumerate(vector_hits[:top_k]):
                r = _hit_to_result(h, ["vector"], model=corpus.embed_model)
                # Always include raw vector_cosine + rank so the UI can
                # re-derive display % under user-tuned calibration without
                # re-querying. Cheap, never sensitive.
                r.explain = {"vector_cosine": float(h.score), "vector_rank": rank + 1}
                out.append(r)
            return out

        if mode == "bm25":
            out = []
            max_score = max((s for _, s, _ in bm25_hits), default=0.0)
            for rank, (cid, score, payload) in enumerate(bm25_hits[:top_k]):
                r = _bm25_to_result(corpus.name, cid, score, payload, ["bm25"], max_score=max_score)
                r.explain = {
                    "bm25_score": float(score),
                    "bm25_rank": rank + 1,
                    "bm25_max": float(max_score),
                }
                out.append(r)
            return out

        # mode == "hybrid": Reciprocal Rank Fusion.
        # RRF score = sum_over_retrievers(1 / (k + rank)). Identity is the
        # chunk id, which both retrievers share thanks to deterministic IDs.
        rrf: dict[str, float] = {}
        sources: dict[str, list[str]] = {}
        payloads_by_id: dict[str, dict] = {}
        vector_score_by_id: dict[str, float] = {}
        vector_rank_by_id: dict[str, int] = {}
        bm25_score_by_id: dict[str, float] = {}
        bm25_rank_by_id: dict[str, int] = {}

        for rank, hit in enumerate(vector_hits):
            cid = _id_for(hit)
            if not cid:
                continue
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            sources.setdefault(cid, []).append("vector")
            payloads_by_id[cid] = _hit_payload(hit)
            vector_score_by_id[cid] = hit.score
            vector_rank_by_id[cid] = rank + 1

        for rank, (cid, score, payload) in enumerate(bm25_hits):
            rrf[cid] = rrf.get(cid, 0.0) + BM25_WEIGHT / (RRF_K + rank + 1)
            sources.setdefault(cid, []).append("bm25")
            payloads_by_id.setdefault(cid, payload)
            bm25_score_by_id[cid] = score
            bm25_rank_by_id[cid] = rank + 1

        # Recency decay: multiply each candidate's RRF score by its decay
        # factor BEFORE truncating to top_k, so a fresh chunk just outside the
        # window can be rescued above a stale one and lexical closeness can't
        # float stale content to the top. No-op when the corpus has no
        # half-life or the chunk has no timestamp (factor 1.0).
        half_life = corpus.decay_half_life_days
        now = time.time()
        decay_by_id: dict[str, float] = {}
        decayed: dict[str, float] = {}
        for cid, raw in rrf.items():
            ts = payloads_by_id.get(cid, {}).get("doc_timestamp")
            factor = _recency_factor(ts, half_life, now)
            decay_by_id[cid] = factor
            decayed[cid] = raw * factor

        fused = sorted(decayed.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        # For BM25-only display: rescale per-query so the top BM25 hit reads
        # at BM25_DISPLAY_TOP (~90%) and weaker BM25 hits taper toward
        # BM25_DISPLAY_FLOOR. The raw fused RRF score is unchanged for ranking.
        max_bm25 = max(bm25_score_by_id.values()) if bm25_score_by_id else 0.0

        out = []
        for cid, fused_score in fused:
            payload = payloads_by_id[cid]
            matched_by = sources.get(cid, [])
            # similarity_pct: pick the most informative signal for display.
            # When vector saw it, the calibrated cosine pct (after sigmoid)
            # already reflects semantic relevance. When only BM25 saw it,
            # use rank-relative scaling so a strong keyword hit doesn't read
            # as "1.6% noise" (the raw RRF score for a BM25-only hit).
            # When BOTH retrievers fired, we take the higher of the two —
            # confirmation across modalities should boost confidence, not
            # average it down.
            cos_pct = (
                _cosine_to_pct(vector_score_by_id[cid], model=corpus.embed_model)
                if cid in vector_score_by_id
                else None
            )
            bm_pct = None
            if max_bm25 > 0 and cid in bm25_score_by_id:
                ratio = bm25_score_by_id[cid] / max_bm25
                bm_pct = BM25_DISPLAY_FLOOR + (BM25_DISPLAY_TOP - BM25_DISPLAY_FLOOR) * ratio
            if cos_pct is not None and bm_pct is not None:
                pct = max(cos_pct, bm_pct)
            elif cos_pct is not None:
                pct = cos_pct
            elif bm_pct is not None:
                pct = bm_pct
            else:
                # Should not happen — fused candidate must come from at
                # least one retriever. Defensive fallback at the floor.
                pct = BM25_DISPLAY_FLOOR
            r = _payload_to_result(payload, fused_score, pct, matched_by)
            # Always emit raw scores so the UI can re-derive display % under
            # user-tuned calibration. `rrf` is the pre-decay fusion score;
            # `decay` is the recency multiplier (1.0 when off); the result's
            # `score` is rrf * decay, i.e. the value that actually ranked it.
            r.explain = {"rrf": float(rrf[cid]), "decay": float(decay_by_id[cid])}
            if cid in vector_score_by_id:
                r.explain["vector_cosine"] = float(vector_score_by_id[cid])
                r.explain["vector_rank"] = vector_rank_by_id[cid]
            if cid in bm25_score_by_id:
                r.explain["bm25_score"] = float(bm25_score_by_id[cid])
                r.explain["bm25_rank"] = bm25_rank_by_id[cid]
                r.explain["bm25_max"] = float(max_bm25)
            out.append(r)
        # Final display sort happens in search() — _search_one returns RRF
        # order so the outer caller can fuse multi-corpus results sensibly.
        return out

    # ----- corpus management ----------------------------------------------------
    def delete_corpus(self, name: str) -> None:
        corpus = self.registry.get(name)
        self.store.drop_collection(_collection_for(corpus.name))
        self.bm25.drop(corpus.name)
        self.registry.delete(name)

    def set_decay(self, name: str, half_life_days: float | None) -> Corpus:
        """Set (or clear, with None) a corpus's recency-decay half-life in days.

        No re-index needed — decay is applied at search time from the
        per-chunk doc_timestamp already in each payload.
        """
        corpus = self.registry.get(name)
        if half_life_days is not None and half_life_days <= 0:
            raise CorpusError("half-life must be positive (or omit to disable decay)")
        corpus.decay_half_life_days = half_life_days
        self.registry.upsert(corpus)
        return corpus

    def get_chunk_window(
        self,
        corpus_name: str,
        chunk_id: str,
        window: int,
    ) -> dict | None:
        """Return an expanded context window around a chunk.

        `window` is the number of chars on each side. Pass -1 for the full
        source. Returns None if the chunk can't be found in either store.
        """
        corpus = self.registry.get(corpus_name)
        payload = self.store.get_by_id(_collection_for(corpus.name), chunk_id)
        if payload is None:
            # Vector store missed — try BM25 in case this chunk only lives
            # there (shouldn't happen under normal indexing, but be safe).
            payload = self.bm25.get_by_id(corpus.name, chunk_id)
        if payload is None:
            return None
        source_text = payload.get("source_text", "") or ""
        chunk_start = int(payload.get("chunk_start", 0))
        chunk_end = int(payload.get("chunk_end", 0))
        if window < 0:
            before_start = 0
            after_end = len(source_text)
        else:
            before_start = max(0, chunk_start - window)
            after_end = min(len(source_text), chunk_end + window)
        return {
            "corpus": corpus.name,
            "chunk_id": chunk_id,
            "source_id": payload.get("source_id", ""),
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "before": source_text[before_start:chunk_start],
            "chunk": payload.get("text", "") or source_text[chunk_start:chunk_end],
            "after": source_text[chunk_end:after_end],
            "source_length": len(source_text),
            "window": window,
        }

    def delete_source(self, corpus_name: str, source_id: str) -> None:
        corpus = self.registry.get(corpus_name)
        collection = _collection_for(corpus.name)
        chunks_freed = _count_chunks_for_source(self.bm25, corpus.name, source_id)
        self.store.delete_by_source(collection, source_id)
        self.bm25.delete_by_source(corpus.name, source_id)
        if source_id in corpus.sources:
            corpus.sources.remove(source_id)
        corpus.source_hashes.pop(source_id, None)
        corpus.doc_count = len(corpus.sources)
        corpus.chunk_count = max(0, corpus.chunk_count - chunks_freed)
        corpus.updated_at = time.time()
        if self.ephemeral:
            self.registry._corpora[corpus.name] = corpus
        else:
            self.registry.upsert(corpus)

    def list_corpora(self) -> list[Corpus]:
        return self.registry.list()

    def filterable_fields(self, corpus_name: str, max_values: int = 12) -> dict:
        """Describe what `filters` a caller can pass for this corpus.

        The `filters` param is otherwise a black box. This scans the corpus's
        stored payloads and reports the concrete filter expressions available:
        the always-present `source:` and `corpus:` filters, plus every
        `meta.<key>` discovered in chunk metadata with up to `max_values`
        observed sample values each. Lets a caller pre-filter by
        actor/channel/date before semantic ranking instead of guessing keys.
        """
        corpus = self.registry.get(corpus_name)  # raises CorpusError if absent
        idx = self.bm25._load(corpus.name)
        meta_values: dict[str, set] = {}
        for payload in idx.payloads:
            meta = payload.get("metadata") or {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    meta_values.setdefault(k, set())
                    if len(meta_values[k]) < max_values:
                        meta_values[k].add(v)
        has_timestamp = any(p.get("doc_timestamp") is not None for p in idx.payloads)
        return {
            "corpus": corpus.name,
            "filters": {
                "source": {"form": "source:GLOB", "description": "fnmatch on source_id"},
                "corpus": {"form": "corpus:NAME", "description": "exact corpus match"},
                "meta": {
                    "form": "meta.KEY=VALUE",
                    "keys": {
                        k: sorted(str(x) for x in vals) for k, vals in sorted(meta_values.items())
                    },
                },
            },
            "has_doc_timestamp": has_timestamp,
        }

    # ----- migration ------------------------------------------------------------
    def migrate_corpus(
        self,
        name: str,
        to_backend: str,
        to_model: str | None = None,
    ) -> Corpus:
        """Re-embed every chunk in `name` with a new backend / model.

        Strategy: re-index every original source under a temp corpus using
        the requested backend, then atomically swap names. If everything
        fails the original corpus is untouched (we created the temp first
        and never modified the original until success).

        Sources that no longer exist (deleted files, dead URLs) are skipped
        with a warning rather than failing the whole migration.
        """
        old = self.registry.get(name)
        if old.embed_backend == to_backend and (to_model is None or old.embed_model == to_model):
            raise CorpusError(
                f"Corpus '{name}' already uses {to_backend}"
                + (f"/{to_model}" if to_model else "")
                + " — nothing to migrate."
            )

        # Override the embed_model setting just long enough to resolve the
        # new backend; we don't replace self.settings, so other corpora are
        # untouched.
        prev_embed_model = self.settings.embed_model
        prev_openai_model = self.settings.openai_embed_model
        if to_model:
            if to_backend == "openai":
                self.settings.openai_embed_model = to_model
            else:
                self.settings.embed_model = to_model
        try:
            new_backend = get_embed_backend(self.settings, prefer=to_backend)
        finally:
            self.settings.embed_model = prev_embed_model
            self.settings.openai_embed_model = prev_openai_model

        if to_model and new_backend.model != to_model:
            raise EmbedBackendError(
                f"Requested model '{to_model}' did not resolve to the expected "
                f"backend model (got '{new_backend.model}'). Check env vars."
            )

        wrapped_new = (
            CachedBackend(new_backend, self._embed_cache) if self._embed_cache else new_backend
        )

        # Reserve a temp slot in the same backend cache so index() finds the
        # new backend when called against the temp corpus.
        temp_name = f"__migrate__{name}__"
        if self.registry.has(temp_name):
            self.delete_corpus(temp_name)
        # Stash the new backend under both 'auto' (for corpus creation) and
        # under the new backend's name (for subsequent _backend_for calls).
        prev_cache = dict(self._backend_cache)
        self._backend_cache["auto"] = wrapped_new
        self._backend_cache[wrapped_new.name] = wrapped_new

        skipped: list[str] = []
        try:
            for src in old.sources:
                try:
                    self.index(src, temp_name, chunker_name=old.chunker)
                except (AdapterError, EmbedBackendError) as e:
                    skipped.append(f"{src}: {e}")
        finally:
            # Restore prior backend cache so subsequent operations against
            # other corpora behave normally.
            self._backend_cache = prev_cache

        if not self.registry.has(temp_name):
            raise CorpusError(
                f"Migration produced an empty corpus — every source failed. "
                f"Original '{name}' left untouched. Errors: {skipped}"
            )

        new_corpus = self.registry.get(temp_name)
        new_corpus.name = name
        new_corpus.created_at = old.created_at

        # Drop the old corpus first (qdrant + bm25 + registry).
        self.delete_corpus(name)

        # Migrate temp -> final by copying points (Qdrant tracks collections
        # via an in-memory + meta.json registry; renaming dirs corrupts it).
        # Scroll points from the temp collection, upsert into the final
        # collection, then drop temp.
        temp_collection = _collection_for(temp_name)
        new_collection = _collection_for(name)
        self.store.ensure_collection(new_collection, new_corpus.dim)
        offset: object = None
        from qdrant_client.http import models as qm
        while True:
            points, offset = self.store.client.scroll(
                collection_name=temp_collection,
                with_payload=True,
                with_vectors=True,
                limit=256,
                offset=offset,
            )
            if not points:
                break
            self.store.client.upsert(
                collection_name=new_collection,
                points=[
                    qm.PointStruct(id=p.id, vector=p.vector, payload=p.payload or {})
                    for p in points
                ],
                wait=True,
            )
            if offset is None:
                break
        self.store.drop_collection(temp_collection)

        # BM25 pickle: rename file AND rewrite payload.corpus inside it.
        # Otherwise BM25 hits surface with the temp_name in their payload,
        # which leaks the implementation detail into search results and
        # breaks corpus filters.
        old_bm25 = self.settings.home / "bm25" / f"{temp_name}.pkl"
        new_bm25 = self.settings.home / "bm25" / f"{name}.pkl"
        if old_bm25.exists():
            if new_bm25.exists():
                new_bm25.unlink()
            old_bm25.rename(new_bm25)
        # Drop in-memory caches for BOTH names so the rename is visible.
        self.bm25._cache.pop(temp_name, None)
        self.bm25._cache.pop(name, None)
        # Now rewrite payloads under the new name.
        idx = self.bm25._load(name)
        for payload in idx.payloads:
            if payload.get("corpus") != name:
                payload["corpus"] = name
        self.bm25._persist(name)

        # Drop the temp registry entry; upsert under final name. We also
        # rewrite each chunk payload's "corpus" field to the final name —
        # otherwise filters / display would still show the temp name.
        # The simplest way: scroll-and-rewrite payload corpus key in-place.
        offset = None
        while True:
            points, offset = self.store.client.scroll(
                collection_name=new_collection,
                with_payload=True,
                limit=256,
                offset=offset,
            )
            if not points:
                break
            updates = []
            for p in points:
                payload = p.payload or {}
                if payload.get("corpus") != name:
                    payload["corpus"] = name
                    updates.append((p.id, payload))
            for pid, payload in updates:
                self.store.client.set_payload(
                    collection_name=new_collection,
                    payload=payload,
                    points=[pid],
                    wait=True,
                )
            if offset is None:
                break

        self.registry.delete(temp_name)
        self.registry.upsert(new_corpus)

        return new_corpus

    # ----- export / import ------------------------------------------------------
    def export_corpus(self, name: str, dest: Path) -> Path:
        """Write a portable .tar.gz containing this corpus' state.

        Bundle layout:
            corpus.json            metadata (name, embed model, dim, source list, hashes)
            qdrant/                collection storage (dir tree)
            bm25.pkl               inverted index pickle (if present)
        """
        import json
        import tarfile
        import tempfile

        corpus = self.registry.get(name)
        # Force an explicit close on the qdrant client so the storage dir is
        # in a consistent state before we tar it. Re-create on next use.
        self.store.client.close()

        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            (staging / "corpus.json").write_text(
                json.dumps(_corpus_to_dict(corpus), indent=2, sort_keys=True)
            )

            # Qdrant embedded mode stores everything in one folder; per-collection
            # data lives under that folder. Easiest portable export: ship the
            # whole Qdrant dir but only collections matching this corpus.
            collection = _collection_for(name)
            src_qdrant = self.settings.qdrant_path / "collection" / collection
            if src_qdrant.is_dir():
                _copytree(src_qdrant, staging / "qdrant" / "collection" / collection)
            meta_root = self.settings.qdrant_path / "meta.json"
            if meta_root.is_file():
                (staging / "qdrant" / "meta.json").parent.mkdir(parents=True, exist_ok=True)
                (staging / "qdrant" / "meta.json").write_bytes(meta_root.read_bytes())

            bm25_src = self.settings.home / "bm25" / f"{name}.pkl"
            if bm25_src.is_file():
                (staging / "bm25.pkl").write_bytes(bm25_src.read_bytes())

            dest = dest.expanduser().resolve()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(dest, "w:gz") as tar:
                for child in staging.iterdir():
                    tar.add(child, arcname=child.name)

        # Re-open the store so subsequent calls work.
        self.store = QdrantStore(
            None if self.ephemeral else self.settings.qdrant_path,
            url=None if self.ephemeral else self.settings.qdrant_url,
        )
        return dest

    def import_corpus(self, archive: Path, rename: str | None = None) -> Corpus:
        """Restore a corpus from a tarball produced by export_corpus()."""
        import json
        import tarfile
        import tempfile

        archive = archive.expanduser().resolve()
        if not archive.is_file():
            raise CorpusError(f"Archive not found: {archive}")

        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            with tarfile.open(archive, "r:gz") as tar:
                # SECURITY: 'data' filter blocks path-traversal (../, absolute
                # paths, device/symlink entries) from an attacker-authored tarball.
                tar.extractall(staging, filter="data")

            meta_path = staging / "corpus.json"
            if not meta_path.is_file():
                raise CorpusError(f"Archive missing corpus.json: {archive}")
            meta = json.loads(meta_path.read_text())
            target_name = rename or meta["name"]

            if self.registry.has(target_name):
                raise CorpusError(
                    f"Corpus '{target_name}' already exists. "
                    "Pass --rename to import under a different name."
                )

            # Close current store so we can write into its dir.
            self.store.client.close()

            # Restore qdrant collection. If we're renaming, the stored
            # directory still has the old name; rename it on copy.
            src_qdrant = staging / "qdrant" / "collection"
            if src_qdrant.is_dir():
                old_collection = _collection_for(meta["name"])
                new_collection = _collection_for(target_name)
                src_collection_dir = src_qdrant / old_collection
                if src_collection_dir.is_dir():
                    dest_collection_dir = (
                        self.settings.qdrant_path / "collection" / new_collection
                    )
                    dest_collection_dir.parent.mkdir(parents=True, exist_ok=True)
                    _copytree(src_collection_dir, dest_collection_dir)

            # SECURITY: do NOT import the tarball's bm25.pkl — loading an
            # attacker-supplied pickle is arbitrary code execution. The BM25
            # index is rebuilt from the imported documents on the next index
            # (bm25_store load is a no-op when the .pkl is absent), so skipping
            # the untrusted pickle is safe and closes the RCE vector.
            pass

            corpus = Corpus(
                name=target_name,
                embed_backend=meta["embed_backend"],
                embed_model=meta["embed_model"],
                dim=meta["dim"],
                chunker=meta.get("chunker", "sentence_window"),
                doc_count=meta.get("doc_count", 0),
                chunk_count=meta.get("chunk_count", 0),
                created_at=meta.get("created_at", time.time()),
                updated_at=time.time(),
                sources=list(meta.get("sources", [])),
                source_hashes=dict(meta.get("source_hashes", {})),
            )
            self.registry.upsert(corpus)

        # Re-open store so the new collection is visible.
        self.store = QdrantStore(
            None if self.ephemeral else self.settings.qdrant_path,
            url=None if self.ephemeral else self.settings.qdrant_url,
        )
        return corpus


# ----- helpers ------------------------------------------------------------------
def _collection_for(corpus_name: str) -> str:
    return f"vecgrep__{corpus_name}"


def _chunk_id(corpus_name: str, source_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, f"{corpus_name}|{source_id}|{chunk_index}"))


def _id_for(hit: StoredHit) -> str:
    # StoredHit doesn't carry the Qdrant point id, but we can regenerate it
    # from payload — same inputs to uuid5 produce the same id.
    return _chunk_id(hit.corpus, hit.source_id, hit.chunk_index)


def _hit_payload(hit: StoredHit) -> dict:
    return {
        "corpus": hit.corpus,
        "source_id": hit.source_id,
        "source_text": hit.source_text,
        "chunk_index": hit.chunk_index,
        "chunk_start": hit.chunk_start,
        "chunk_end": hit.chunk_end,
        "text": hit.chunk_text,
        "metadata": hit.metadata,
        "doc_timestamp": hit.doc_timestamp,
    }


def _corpus_to_dict(c: Corpus) -> dict:
    return {
        "name": c.name,
        "embed_backend": c.embed_backend,
        "embed_model": c.embed_model,
        "dim": c.dim,
        "chunker": c.chunker,
        "doc_count": c.doc_count,
        "chunk_count": c.chunk_count,
        "created_at": c.created_at,
        "updated_at": c.updated_at,
        "sources": list(c.sources),
        "source_hashes": dict(c.source_hashes),
    }


def _copytree(src: Path, dst: Path) -> None:
    import shutil

    shutil.copytree(src, dst, dirs_exist_ok=True)


def _passes_filters(
    result: SearchResult, filters: list[str], default_active: bool = False
) -> bool:
    """Apply --filter expressions. Supported forms:

        source:GLOB      — fnmatch against result.source_id
        corpus:NAME      — exact corpus name match
        meta.KEY=VALUE   — exact metadata field match (string compare)

    All filters AND together. A malformed filter is silently ignored.

    default_active: back-compat for the write-tool status schema. When True, a
    `meta.status=active` filter also passes a chunk that has NO status field at
    all (legacy chunks indexed before the schema existed) — so the default
    "show only active" retrieval doesn't make every pre-schema chunk vanish.
    Only relaxes the status=active case; every other filter stays strict.
    """
    for f in filters:
        if default_active and f == "meta.status=active":
            # Legacy chunk (no status key) counts as active; an explicit
            # non-active status still fails.
            if str(result.metadata.get("status", "active")) != "active":
                return False
            continue
        if ":" not in f and "=" not in f:
            continue
        if f.startswith("source:"):
            pat = f[len("source:") :]
            if not fnmatch.fnmatch(result.source_id, pat):
                return False
        elif f.startswith("corpus:"):
            if result.corpus != f[len("corpus:") :]:
                return False
        elif f.startswith("meta."):
            key_value = f[len("meta.") :]
            if "=" not in key_value:
                continue
            key, value = key_value.split("=", 1)
            if str(result.metadata.get(key, "")) != value:
                return False
    return True


def _count_chunks_for_source(bm25: BM25Store, corpus_name: str, source_id: str) -> int:
    # BM25 store carries the by_source map; cheaper than asking Qdrant.
    idx = bm25._load(corpus_name)
    return len(idx.by_source.get(source_id, []))


def _expand(source: str, adapter, include: str | None = None) -> list[Document]:
    """Adapters yield one or more docs. For directories, walk and re-detect.

    A directory path: walk recursively, dispatch each file through detect_adapter
    so a single index call can mix file types. `include` is an optional glob
    (fnmatch against the filename) to restrict which files are indexed — e.g.
    `*.md` to index only markdown and skip sibling `.jsonl` raws. None = all.
    """
    p = Path(source) if not source.startswith(("http://", "https://")) else None
    if p and p.is_dir():
        docs: list[Document] = []
        for sub in sorted(p.rglob("*")):
            if not sub.is_file():
                continue
            if include and not fnmatch.fnmatch(sub.name, include):
                continue
            try:
                sub_adapter_cls = detect_adapter(str(sub))
            except Exception:
                continue
            try:
                docs.extend(sub_adapter_cls().load(str(sub)))
            except Exception:
                continue
        return docs
    return list(adapter.load(source))


# Cosine-to-percentage calibration.
#
# The naive `(cos + 1) / 2 * 100` mapping is misleading for dense embedding
# models: with `nomic-embed-text` the empirical noise floor sits at cos~0.50
# (which reads as 75% under the naive map) and meaningful matches start at
# cos~0.65. Under the naive map the whole user-relevant range (50% → 95%
# usefulness) is squashed into the top ~20 display points and ~75% of the
# scale is wasted on noise that the user would never inspect.
#
# We map cos → pct via a sigmoid centered on the empirical
# inflection point so display percentages spread where they matter:
#
#   cos 0.40 → ~3%    (well below noise — irrelevant)
#   cos 0.55 → ~17%   (noise floor for unrelated queries)
#   cos 0.66 → 50%    (inflection)
#   cos 0.75 → ~75%   (clearly relevant)
#   cos 0.85 → ~91%   (strong match)
#   cos 0.92 → ~97%   (near-duplicate)
#
# Calibration is per-embedding-model because each model's cosine distribution
# differs: where its noise floor sits and where genuine matches start. A center
# tuned for one model reads pessimistically (or optimistically) on another.
#
# nomic-embed-text: noise floor ~0.50, strong matches ~0.65+. (center 0.66)
# bge-m3:           cosines run LOWER — noise floor ~0.52, strong matches ~0.60-0.66.
#                   Empirically: gibberish tops ~0.53; real hits 0.60-0.66. So
#                   center 0.55, slope 25 maps 0.50→~22%, 0.55→50%, 0.62→~85%,
#                   0.66→~94% — spreading display % across bge-m3's actual range
#                   instead of squashing everything below 50% (the old 0.66
#                   center sat ABOVE almost every bge-m3 cosine).
# Fallback (unknown model): the nomic-ish defaults.
CALIBRATION_CENTER = 0.66
CALIBRATION_SLOPE = 12.0

_MODEL_CALIBRATION: dict[str, tuple[float, float]] = {
    "nomic-embed-text": (0.66, 12.0),
    "bge-m3": (0.55, 25.0),
    "mxbai-embed-large": (0.55, 25.0),  # same family/range as bge-m3
}


def _calibration_for(model: str | None) -> tuple[float, float]:
    """(center, slope) for an embed model, falling back to module defaults."""
    if model and model in _MODEL_CALIBRATION:
        return _MODEL_CALIBRATION[model]
    return CALIBRATION_CENTER, CALIBRATION_SLOPE


def _cosine_floor(model: str | None) -> float:
    """The minimum cosine a vector hit must clear to enter fusion: the model's
    calibration center minus COSINE_FLOOR_MARGIN. <=0 means no floor."""
    center, _ = _calibration_for(model)
    return max(0.0, center - COSINE_FLOOR_MARGIN)


def _cosine_to_pct(
    score: float,
    center: float | None = None,
    slope: float | None = None,
    model: str | None = None,
) -> float:
    """Sigmoid-calibrated cosine → display percentage.

    `model` selects the per-model calibration (center/slope). Explicit `center`
    /`slope` args override it (used by the web-UI tuning page). With neither,
    falls back to the module defaults.
    """
    mc, ms = _calibration_for(model)
    c = mc if center is None else center
    s = ms if slope is None else slope
    cos = max(-1.0, min(1.0, score))
    x = s * (cos - c)
    # Guard against overflow for extreme x.
    if x > 60:
        return 100.0
    if x < -60:
        return 0.0
    return 100.0 / (1.0 + math.exp(-x))


# Cross-encoder-reranker → percentage calibration.
#
# When reranking is on, the cross-encoder's sigmoid-squashed score (0..1) is the
# single best proxy for P(relevant) — it's roughly query-independent, unlike raw
# cosine (whose absolute value drifts per query) or RRF (contaminated by which
# retrievers fired). So the displayed pct comes straight from it via one sigmoid,
# the SAME formula for every hit regardless of which retriever surfaced it.
#
# Empirically (bge-reranker-base on this corpus): the noise floor sits ~0.50-0.53,
# genuine-but-weak hits ~0.57, strong hits 0.66-0.73. Center 0.57, slope 35 maps:
#   0.51 → ~12%   (noise)
#   0.57 → 50%    (uncertain boundary)
#   0.66 → ~96%   (strong)
RERANK_CALIBRATION_CENTER = 0.57
RERANK_CALIBRATION_SLOPE = 35.0


def _rerank_to_pct(prob: float) -> float:
    """Sigmoid-calibrated cross-encoder score (0..1) → display percentage."""
    x = RERANK_CALIBRATION_SLOPE * (prob - RERANK_CALIBRATION_CENTER)
    if x > 60:
        return 100.0
    if x < -60:
        return 0.0
    return 100.0 / (1.0 + math.exp(-x))


def _dedup_overlapping(results: list[SearchResult], min_overlap: float = 0.5) -> list[SearchResult]:
    """Drop near-duplicate chunks from the same source with overlapping spans.

    The sentence-window chunker emits overlapping windows (stride < window), so
    one passage can surface as several hits at different ranks, wasting top_k
    slots. Two hits collide when they share a (corpus, source_id) and their
    char ranges overlap by >= `min_overlap` of the shorter span. We keep the
    higher-scoring hit of each colliding group and preserve input order
    otherwise (callers sort afterward).
    """
    kept: list[SearchResult] = []
    for r in results:
        dup_idx = None
        for i, k in enumerate(kept):
            if k.corpus != r.corpus or k.source_id != r.source_id:
                continue
            lo = max(k.chunk_start, r.chunk_start)
            hi = min(k.chunk_end, r.chunk_end)
            overlap = max(0, hi - lo)
            shorter = min(k.chunk_end - k.chunk_start, r.chunk_end - r.chunk_start)
            if shorter > 0 and overlap / shorter >= min_overlap:
                dup_idx = i
                break
        if dup_idx is None:
            kept.append(r)
        elif r.score > kept[dup_idx].score:
            kept[dup_idx] = r  # replace with the stronger of the pair
    return kept


def _hit_to_result(h: StoredHit, matched_by: list[str], model: str | None = None) -> SearchResult:
    return _payload_to_result(
        _hit_payload(h), h.score, _cosine_to_pct(h.score, model=model), matched_by
    )


def _bm25_to_result(
    corpus_name: str,
    cid: str,
    score: float,
    payload: dict,
    matched_by: list[str],
    max_score: float | None = None,
) -> SearchResult:
    """BM25 hit → SearchResult with rank-relative display percentage.

    If `max_score` is provided (caller has the full BM25 candidate set),
    pct = floor + (top - floor) * (score / max_score). Otherwise falls back
    to a soft cap at score=20 — only meaningful for a single hit in
    isolation, but at least bounded.
    """
    if max_score and max_score > 0:
        ratio = max(0.0, min(1.0, score / max_score))
        pct = BM25_DISPLAY_FLOOR + (BM25_DISPLAY_TOP - BM25_DISPLAY_FLOOR) * ratio
    else:
        pct = min(BM25_DISPLAY_TOP, score / 20 * BM25_DISPLAY_TOP)
    return _payload_to_result(payload, score, pct, matched_by)


def _payload_to_result(
    payload: dict,
    score: float,
    pct: float,
    matched_by: list[str],
) -> SearchResult:
    source_text = payload.get("source_text", "") or ""
    chunk_start = int(payload.get("chunk_start", 0))
    chunk_end = int(payload.get("chunk_end", 0))
    before = source_text[max(0, chunk_start - 400) : chunk_start] if source_text else ""
    after = source_text[chunk_end : chunk_end + 400] if source_text else ""
    corpus_name = payload.get("corpus", "") or ""
    source_id = payload.get("source_id", "") or ""
    chunk_index = int(payload.get("chunk_index", 0))
    cid = _chunk_id(corpus_name, source_id, chunk_index) if corpus_name and source_id else ""
    return SearchResult(
        score=score,
        similarity_pct=pct,
        chunk=payload.get("text", ""),
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        context_before=before,
        context_after=after,
        source_id=source_id,
        corpus=corpus_name,
        metadata=payload.get("metadata", {}) or {},
        chunk_id=cid,
        matched_by=matched_by,
    )
