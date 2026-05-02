"""High-level service: index and search.

Sits between the API/CLI and the store/embed/adapter layers. Owns the
corpus lifecycle so callers don't have to coordinate four subsystems.
"""
from __future__ import annotations

import fnmatch
import hashlib
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
        self.store = QdrantStore(None if ephemeral else self.settings.qdrant_path)
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
        cache_key = prefer or "auto"
        if cache_key not in self._backend_cache:
            # Reuse an already-resolved 'auto' backend if it happens to match
            # the corpus's pinned backend — avoids a redundant live resolve
            # (and lets tests inject just one mock).
            auto = self._backend_cache.get("auto")
            if auto is not None and auto.name == prefer:
                self._backend_cache[cache_key] = auto
            else:
                raw = get_embed_backend(self.settings, prefer=prefer)
                self._backend_cache[cache_key] = (
                    CachedBackend(raw, self._embed_cache) if self._embed_cache else raw
                )
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
        force: bool = False,
    ) -> tuple[int, int, int]:
        """Index a source into a corpus. Returns (docs, chunks, skipped).

        Incremental: per-document content hashes are persisted in the corpus
        metadata. A second index() call against the same source skips
        embedding when the content hash hasn't changed. Pass force=True to
        re-embed unconditionally (e.g. after a chunker change you want to
        replay).
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
        for doc in _expand(source, adapter):
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
        # Always update the in-memory registry so search() can find the corpus
        # in the same process; only skip the file write when ephemeral.
        self.registry._corpora[corpus.name] = corpus
        if not self.ephemeral:
            self.registry._save()

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
    ) -> list[SearchResult]:
        top_k = top_k or self.settings.default_top_k
        if corpus_name:
            corpora = [self.registry.get(corpus_name)]
        else:
            corpora = self.registry.list()
        if not corpora:
            return []

        # Filtering shrinks the result set after retrieval. Pull a wider pool
        # when filters are active so we don't end up with empty results just
        # because the top hits failed the filter.
        wider = bool(filters) or rerank
        per_corpus_k = CANDIDATE_POOL if wider else top_k

        results: list[SearchResult] = []
        for c in corpora:
            results.extend(self._search_one(c, query, per_corpus_k, mode, explain=explain))

        if filters:
            results = [r for r in results if _passes_filters(r, filters)]

        if rerank:
            results = self._apply_rerank(query, results, top_k, rerank_model, explain=explain)
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
            # Replace the score and pct with reranker output. matched_by
            # gains 'rerank' so the UI can show that this hit was rerank-confirmed.
            if explain:
                r.explain = {**(r.explain or {}), "rerank_score": float(score)}
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
        explain: bool = False,
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
            out: list[SearchResult] = []
            for rank, h in enumerate(vector_hits[:top_k]):
                r = _hit_to_result(h, ["vector"])
                if explain:
                    r.explain = {"vector_cosine": h.score, "vector_rank": rank + 1}
                out.append(r)
            return out

        if mode == "bm25":
            out = []
            for rank, (cid, score, payload) in enumerate(bm25_hits[:top_k]):
                r = _bm25_to_result(corpus.name, cid, score, payload, ["bm25"])
                if explain:
                    r.explain = {"bm25_score": score, "bm25_rank": rank + 1}
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
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            sources.setdefault(cid, []).append("bm25")
            payloads_by_id.setdefault(cid, payload)
            bm25_score_by_id[cid] = score
            bm25_rank_by_id[cid] = rank + 1

        fused = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out = []
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
            r = _payload_to_result(payload, fused_score, pct, matched_by)
            if explain:
                r.explain = {"rrf": fused_score}
                if cid in vector_score_by_id:
                    r.explain["vector_cosine"] = vector_score_by_id[cid]
                    r.explain["vector_rank"] = vector_rank_by_id[cid]
                if cid in bm25_score_by_id:
                    r.explain["bm25_score"] = bm25_score_by_id[cid]
                    r.explain["bm25_rank"] = bm25_rank_by_id[cid]
            out.append(r)
        return out

    # ----- corpus management ----------------------------------------------------
    def delete_corpus(self, name: str) -> None:
        corpus = self.registry.get(name)
        self.store.drop_collection(_collection_for(corpus.name))
        self.bm25.drop(corpus.name)
        self.registry.delete(name)

    def list_corpora(self) -> list[Corpus]:
        return self.registry.list()

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
        self.store = QdrantStore(None if self.ephemeral else self.settings.qdrant_path)
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
                tar.extractall(staging)

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

            bm25_src = staging / "bm25.pkl"
            if bm25_src.is_file():
                bm25_dir = self.settings.home / "bm25"
                bm25_dir.mkdir(parents=True, exist_ok=True)
                (bm25_dir / f"{target_name}.pkl").write_bytes(bm25_src.read_bytes())

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
        self.store = QdrantStore(None if self.ephemeral else self.settings.qdrant_path)
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


def _passes_filters(result: SearchResult, filters: list[str]) -> bool:
    """Apply --filter expressions. Supported forms:

        source:GLOB      — fnmatch against result.source_id
        corpus:NAME      — exact corpus name match
        meta.KEY=VALUE   — exact metadata field match (string compare)

    All filters AND together. A malformed filter is silently ignored.
    """
    for f in filters:
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
