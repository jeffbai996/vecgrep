"""High-level service: index and search.

Sits between the API/CLI and the store/embed/adapter layers. Owns the
corpus lifecycle so callers don't have to coordinate four subsystems.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import Settings, get_settings
from .embed import EmbedBackend, EmbedBackendError, get_embed_backend
from .ingestion.adapters import (
    Document,
    detect_adapter,
)
from .ingestion.chunkers import Chunker, FixedTokenChunker, SentenceWindowChunker
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
}

SearchMode = Literal["hybrid", "vector", "bm25"]
DEFAULT_MODE: SearchMode = "hybrid"

# Reciprocal Rank Fusion constant. 60 is the canonical value from the
# original RRF paper; we expose it here as a single knob.
RRF_K = 60

# How many candidates each retriever returns before fusion. Larger pool
# = better recall, marginal cost. 50 is a good default for small corpora.
CANDIDATE_POOL = 50

# Stable namespace for chunk IDs. Lets BM25 and Qdrant reference the same
# chunks deterministically — re-indexing a source with the same content
# regenerates the same IDs.
_ID_NAMESPACE = uuid.UUID("3a7d9e5f-0c1b-4a2e-9f4d-abcdef000001")


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
    # Which retrievers placed this result. "vector", "bm25", or both.
    matched_by: list[str]


class VecgrepService:
    def __init__(self, settings: Settings | None = None, ephemeral: bool = False) -> None:
        self.settings = settings or get_settings()
        self.ephemeral = ephemeral
        self.registry = CorpusRegistry(self.settings.corpora_file)
        self.store = QdrantStore(None if ephemeral else self.settings.qdrant_path)
        self.bm25 = BM25Store(None if ephemeral else self.settings.home / "bm25")
        self._backend_cache: dict[str, EmbedBackend] = {}

    # ----- backend resolution ---------------------------------------------------
    def _backend_for(self, corpus: Corpus | None) -> EmbedBackend:
        prefer = corpus.embed_backend if corpus else None
        cache_key = prefer or "auto"
        if cache_key not in self._backend_cache:
            self._backend_cache[cache_key] = get_embed_backend(self.settings, prefer=prefer)
        backend = self._backend_cache[cache_key]
        if corpus and (backend.model != corpus.embed_model or backend.dim != corpus.dim):
            raise EmbedBackendError(
                f"Corpus '{corpus.name}' was indexed with "
                f"{corpus.embed_backend}/{corpus.embed_model} (dim={corpus.dim}), "
                f"but current backend is {backend.name}/{backend.model} (dim={backend.dim}). "
                "Either set the matching model or recreate the corpus."
            )
        return backend

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
    ) -> tuple[int, int]:
        """Index a source into a corpus. Returns (docs, chunks)."""
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

        total_docs = 0
        total_chunks = 0
        # When a source is being re-indexed, subtract its old chunk count so
        # the corpus total stays accurate after we replace it.
        chunks_freed = 0
        for doc in _expand(source, adapter):
            chunks = chunker.chunk(doc.text)
            if not chunks:
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

        corpus.doc_count = len(corpus.sources)
        corpus.chunk_count = corpus.chunk_count - chunks_freed + total_chunks
        corpus.updated_at = time.time()
        # Always update the in-memory registry so search() can find the corpus
        # in the same process; only skip the file write when ephemeral.
        self.registry._corpora[corpus.name] = corpus
        if not self.ephemeral:
            self.registry._save()
        return total_docs, total_chunks

    # ----- search ---------------------------------------------------------------
    def search(
        self,
        query: str,
        corpus_name: str | None = None,
        top_k: int | None = None,
        mode: SearchMode = DEFAULT_MODE,
        rerank: bool = False,
        rerank_model: str | None = None,
    ) -> list[SearchResult]:
        top_k = top_k or self.settings.default_top_k
        if corpus_name:
            corpora = [self.registry.get(corpus_name)]
        else:
            corpora = self.registry.list()
        if not corpora:
            return []

        # When reranking, ask each retriever for the full candidate pool so
        # the reranker has more to work with; truncate to top_k after.
        per_corpus_k = CANDIDATE_POOL if rerank else top_k

        results: list[SearchResult] = []
        for c in corpora:
            results.extend(self._search_one(c, query, per_corpus_k, mode))

        if rerank:
            results = self._apply_rerank(query, results, top_k, rerank_model)
        else:
            results.sort(key=lambda r: r.score, reverse=True)
            results = results[:top_k]
        return results

    def _apply_rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
        model_name: str | None,
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
            # Replace the score and pct with reranker output. matched_by
            # gains 'rerank' so the UI can show that this hit was rerank-confirmed.
            r.score = float(score)
            r.similarity_pct = float(score) * 100
            if "rerank" not in r.matched_by:
                r.matched_by = [*r.matched_by, "rerank"]
            out.append(r)
        return out

    def _search_one(
        self,
        corpus: Corpus,
        query: str,
        top_k: int,
        mode: SearchMode,
    ) -> list[SearchResult]:
        collection = _collection_for(corpus.name)

        vector_hits: list[StoredHit] = []
        bm25_hits: list[tuple[str, float, dict]] = []

        if mode in ("hybrid", "vector"):
            backend = self._backend_for(corpus)
            qv = backend.embed_one(query)
            vector_hits = self.store.search(collection, qv, top_k=CANDIDATE_POOL)

        if mode in ("hybrid", "bm25"):
            bm25_hits = self.bm25.search(corpus.name, query, top_k=CANDIDATE_POOL)

        if mode == "vector":
            return [_hit_to_result(h, ["vector"]) for h in vector_hits[:top_k]]

        if mode == "bm25":
            return [_bm25_to_result(corpus.name, cid, score, payload, ["bm25"])
                    for cid, score, payload in bm25_hits[:top_k]]

        # mode == "hybrid": Reciprocal Rank Fusion.
        # RRF score = sum_over_retrievers(1 / (k + rank)). Identity is the
        # chunk id, which both retrievers share thanks to deterministic IDs.
        rrf: dict[str, float] = {}
        sources: dict[str, list[str]] = {}
        payloads_by_id: dict[str, dict] = {}
        vector_score_by_id: dict[str, float] = {}

        for rank, hit in enumerate(vector_hits):
            cid = _id_for(hit)
            if not cid:
                continue
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            sources.setdefault(cid, []).append("vector")
            payloads_by_id[cid] = _hit_payload(hit)
            vector_score_by_id[cid] = hit.score

        for rank, (cid, _score, payload) in enumerate(bm25_hits):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            sources.setdefault(cid, []).append("bm25")
            payloads_by_id.setdefault(cid, payload)

        fused = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out: list[SearchResult] = []
        for cid, fused_score in fused:
            payload = payloads_by_id[cid]
            matched_by = sources.get(cid, [])
            # similarity_pct displays the underlying vector cosine when the
            # vector retriever saw it; otherwise we display the RRF score
            # scaled to 0-100 for visual continuity. The numeric `score`
            # field always carries the fused RRF for ranking honesty.
            if "vector" in matched_by:
                pct = _cosine_to_pct(vector_score_by_id[cid])
            else:
                pct = min(100.0, fused_score * 100)
            out.append(_payload_to_result(payload, fused_score, pct, matched_by))
        return out

    # ----- corpus management ----------------------------------------------------
    def delete_corpus(self, name: str) -> None:
        corpus = self.registry.get(name)
        self.store.drop_collection(_collection_for(corpus.name))
        self.bm25.drop(corpus.name)
        self.registry.delete(name)

    def list_corpora(self) -> list[Corpus]:
        return self.registry.list()


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
    }


def _count_chunks_for_source(bm25: BM25Store, corpus_name: str, source_id: str) -> int:
    # BM25 store carries the by_source map; cheaper than asking Qdrant.
    idx = bm25._load(corpus_name)
    return len(idx.by_source.get(source_id, []))


def _expand(source: str, adapter) -> list[Document]:
    """Adapters yield one or more docs. For directories, walk and re-detect.

    A directory path: walk recursively, dispatch each file through detect_adapter
    so a single index call can mix file types.
    """
    p = Path(source) if not source.startswith(("http://", "https://")) else None
    if p and p.is_dir():
        docs: list[Document] = []
        for sub in sorted(p.rglob("*")):
            if not sub.is_file():
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


def _cosine_to_pct(score: float) -> float:
    return max(0.0, min(1.0, (score + 1) / 2)) * 100


def _hit_to_result(h: StoredHit, matched_by: list[str]) -> SearchResult:
    return _payload_to_result(_hit_payload(h), h.score, _cosine_to_pct(h.score), matched_by)


def _bm25_to_result(
    corpus_name: str,
    cid: str,
    score: float,
    payload: dict,
    matched_by: list[str],
) -> SearchResult:
    # BM25 scores are unbounded positives. Squash to 0-100 with a soft cap
    # at score=20 (already a strong match in BM25). Purely cosmetic — the
    # actual numeric `score` stays raw for callers who care.
    pct = min(100.0, score / 20 * 100)
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
    return SearchResult(
        score=score,
        similarity_pct=pct,
        chunk=payload.get("text", ""),
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        context_before=before,
        context_after=after,
        source_id=payload.get("source_id", ""),
        corpus=payload.get("corpus", ""),
        metadata=payload.get("metadata", {}) or {},
        matched_by=matched_by,
    )
