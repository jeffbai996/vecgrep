"""High-level service: index and search.

Sits between the API/CLI and the store/embed/adapter layers. Owns the
corpus lifecycle so callers don't have to coordinate four subsystems.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .config import Settings, get_settings
from .embed import EmbedBackend, EmbedBackendError, get_embed_backend
from .ingestion.adapters import (
    Document,
    detect_adapter,
)
from .ingestion.chunkers import Chunker, FixedTokenChunker, SentenceWindowChunker
from .store import (
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


class VecgrepService:
    def __init__(self, settings: Settings | None = None, ephemeral: bool = False) -> None:
        self.settings = settings or get_settings()
        self.ephemeral = ephemeral
        self.registry = CorpusRegistry(self.settings.corpora_file)
        self.store = QdrantStore(None if ephemeral else self.settings.qdrant_path)
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

        adapter_cls = detect_adapter(source)
        adapter = adapter_cls()

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

        total_docs = 0
        total_chunks = 0
        for doc in _expand(source, adapter):
            chunks = chunker.chunk(doc.text)
            if not chunks:
                continue

            # Wipe any prior version of this source so re-indexing is idempotent.
            self.store.delete_by_source(_collection_for(corpus_name), doc.source_id)

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
            self.store.ensure_collection(_collection_for(corpus_name), corpus.dim)
            self.store.upsert(_collection_for(corpus_name), vectors, payloads)

            total_docs += 1
            total_chunks += len(chunks)
            if doc.source_id not in corpus.sources:
                corpus.sources.append(doc.source_id)

        corpus.doc_count = len(corpus.sources)
        corpus.chunk_count += total_chunks
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
    ) -> list[SearchResult]:
        top_k = top_k or self.settings.default_top_k
        if corpus_name:
            corpora = [self.registry.get(corpus_name)]
        else:
            corpora = self.registry.list()
        if not corpora:
            return []

        all_hits: list[StoredHit] = []
        for c in corpora:
            backend = self._backend_for(c)
            qv = backend.embed_one(query)
            hits = self.store.search(_collection_for(c.name), qv, top_k=top_k)
            all_hits.extend(hits)

        all_hits.sort(key=lambda h: h.score, reverse=True)
        all_hits = all_hits[:top_k]

        return [_to_result(h) for h in all_hits]

    # ----- corpus management ----------------------------------------------------
    def delete_corpus(self, name: str) -> None:
        corpus = self.registry.get(name)
        self.store.drop_collection(_collection_for(corpus.name))
        self.registry.delete(name)

    def list_corpora(self) -> list[Corpus]:
        return self.registry.list()


# ----- helpers ------------------------------------------------------------------
def _collection_for(corpus_name: str) -> str:
    return f"vecgrep__{corpus_name}"


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


def _to_result(h: StoredHit) -> SearchResult:
    before = ""
    after = ""
    if h.source_text:
        before = h.source_text[max(0, h.chunk_start - 400) : h.chunk_start]
        after = h.source_text[h.chunk_end : h.chunk_end + 400]
    # Qdrant cosine similarity returns -1..1. Map to 0..100 for display.
    pct = max(0.0, min(1.0, (h.score + 1) / 2)) * 100
    return SearchResult(
        score=h.score,
        similarity_pct=pct,
        chunk=h.chunk_text,
        chunk_start=h.chunk_start,
        chunk_end=h.chunk_end,
        context_before=before,
        context_after=after,
        source_id=h.source_id,
        corpus=h.corpus,
        metadata=h.metadata,
    )
