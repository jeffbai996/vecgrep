"""High-level service: index and search.

Sits between the API/CLI and the store/embed/adapter layers. Owns the
corpus lifecycle so callers don't have to coordinate four subsystems.
"""
from __future__ import annotations

import contextlib

import fnmatch
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import uuid
import weakref
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .config import Settings, get_settings
from .embed import EmbedBackend, EmbedBackendError, get_embed_backend
from .aliases import aliases_path, expand_query, load_alias_map_cached
from .source_labels import apply_labels, load_source_labels_cached, source_labels_path
from .assembly import (
    DEFAULT_FULL_K,
    DEFAULT_MAX_TOTAL,
    DEFAULT_STUB_TOKEN_CEILING,
    ResultStub,
    dedup_near_duplicates,
    mmr_select,
    split_full_and_stubs,
)
from .embed.cache import CachedBackend, EmbedCache
from .explorer import ExplorerCatalog, build_catalog, list_catalog
from .explorer_store import CatalogGeneration, ExplorerStore
from .ingestion.enrich import chunk_enrichment
from .ingestion.adapters import (
    AdapterError,
    Document,
    detect_adapter,
)
from .ingestion.chunkers import (
    Chunker,
    FixedTokenChunker,
    MarkdownSectionChunker,
    CodeSymbolChunker,
    SentenceWindowChunker,
    TurnWindowChunker,
)
from .store import (
    BM25SqliteStore,
    BM25Store,
    Corpus,
    CorpusError,
    CorpusRegistry,
    EPHEMERAL_NAME,
    QdrantStore,
    StoredHit,
)
from .timeline import (
    ANCHOR_TOP_K,
    MAX_GROUPS,
    SLICE_PADDING,
    build_timeline,
    parse_events,
)
from .mutation import CorpusLocks
from .mutation_journal import MutationJournal


def _env_int(name: str, default: int | None) -> int | None:
    """An integer knob from the environment, or `default` when unset or junk.

    Unset and unparseable are treated the same on purpose: a typo in a unit
    file should fall back to the shipped default rather than crash the search
    server at import time."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default



logger = logging.getLogger(__name__)


CHUNKERS: dict[str, type[Chunker]] = {
    "sentence_window": SentenceWindowChunker,
    "fixed_token": FixedTokenChunker,
    "markdown_section": MarkdownSectionChunker,
    "code_symbol": CodeSymbolChunker,
    # Groups whole chat messages instead of cutting every N sentences:
    # a sentence window starts mid-answer and usually carries no speaker
    # line, so "who said what" is lost from the chunk that states it.
    "turn_window": TurnWindowChunker,
}

SearchMode = Literal["hybrid", "vector", "bm25"]
DEFAULT_MODE: SearchMode = "hybrid"


@dataclass(frozen=True)
class SearchWarning:
    corpus: str
    code: str
    message: str


@dataclass
class SearchOutcome:
    results: list["SearchResult"]
    warnings: list[SearchWarning]


class _SearchRuntime:
    """Process-shared resources for one immutable settings generation.

    REST and MCP each keep their own ``VecgrepService`` instance, but they use
    the same Settings object. Sharing the inner executor through that identity
    keeps simultaneous requests from multiplying into one thread pool per
    request (and per transport). Query vectors are singleflighted only while
    work is in progress; completed queries are left to the persistent embed
    cache, so raw query text never becomes process-global retained state.
    """

    def __init__(self, workers: int) -> None:
        self.workers = max(1, workers)
        self.executor = ThreadPoolExecutor(
            max_workers=self.workers,
            thread_name_prefix="vecgrep-corpus",
        )
        self._embedding_lock = threading.Lock()
        self._embedding_futures: dict[
            tuple[str, int, bytes], Future[list[float]]
        ] = {}

    def query_vector(
        self,
        service: "VecgrepService",
        corpus: Corpus,
        query: str,
    ) -> list[float]:
        # A digest is sufficient to coordinate identical in-flight work and
        # avoids retaining private query text in a long-lived runtime object.
        key = (
            service._cache_key_for(corpus),
            corpus.dim,
            hashlib.sha256(query.encode("utf-8")).digest(),
        )
        with self._embedding_lock:
            future = self._embedding_futures.get(key)
            owns_work = future is None
            if future is None:
                future = Future()
                self._embedding_futures[key] = future

        if owns_work:
            try:
                vector = service._embed_query_with_failover(corpus, query)
            except BaseException as exc:
                # Resolve before removal so every waiter already holding this
                # Future observes the same failure. A later request may retry.
                future.set_exception(exc)
            else:
                future.set_result(vector)
            finally:
                with self._embedding_lock:
                    if self._embedding_futures.get(key) is future:
                        self._embedding_futures.pop(key, None)
        return future.result()

    def shutdown(self) -> None:
        # Settings cannot become unreachable while submitted bound methods are
        # still running, so pending work is allowed to finish without blocking
        # the config-reload or garbage-collection thread.
        self.executor.shutdown(wait=False, cancel_futures=False)


_SEARCH_RUNTIMES_LOCK = threading.Lock()
_SEARCH_RUNTIMES: dict[int, _SearchRuntime] = {}
_SEARCH_RUNTIME_FINALIZERS: dict[int, weakref.finalize] = {}


def _release_search_runtime(key: int, runtime: _SearchRuntime) -> None:
    with _SEARCH_RUNTIMES_LOCK:
        if _SEARCH_RUNTIMES.get(key) is not runtime:
            return
        _SEARCH_RUNTIMES.pop(key, None)
        _SEARCH_RUNTIME_FINALIZERS.pop(key, None)
    runtime.shutdown()


def _search_runtime_for(settings: Settings) -> _SearchRuntime:
    """Return the bounded runtime shared by services using ``settings``."""
    key = id(settings)
    with _SEARCH_RUNTIMES_LOCK:
        runtime = _SEARCH_RUNTIMES.get(key)
        if runtime is None:
            workers = max(
                1,
                int(getattr(settings, "search_fanout_workers", 8) or 1),
            )
            runtime = _SearchRuntime(workers)
            _SEARCH_RUNTIMES[key] = runtime
            finalizer = weakref.finalize(
                settings, _release_search_runtime, key, runtime
            )
            # ThreadPoolExecutor has its own interpreter-exit handling. Avoid
            # running a registry callback after module globals are torn down.
            finalizer.atexit = False
            _SEARCH_RUNTIME_FINALIZERS[key] = finalizer
        return runtime


class _QueryVectorMemo:
    """One query embedding per backend/model/dimension within a search.

    Cross-corpus fan-out runs one thread per corpus. Most installations pin
    several corpora to the same embedding model, so without request-local
    coordination every thread can miss the persistent cache at the same time
    and send the identical query to Ollama. A Future gives the first thread
    ownership of that work while peers wait for its result. Different models
    retain their independent fan-out and failures still surface per corpus.
    """

    def __init__(self, service: "VecgrepService", query: str) -> None:
        self._service = service
        self._query = query
        self._lock = threading.Lock()
        self._futures: dict[tuple[str, int], Future[list[float]]] = {}

    def get(self, corpus: Corpus) -> list[float]:
        key = (self._service._cache_key_for(corpus), corpus.dim)
        with self._lock:
            future = self._futures.get(key)
            owns_work = future is None
            if future is None:
                future = Future()
                self._futures[key] = future

        if owns_work:
            try:
                vector = self._service._embed_query_singleflight(
                    corpus, self._query
                )
            except BaseException as exc:
                # Always resolve the Future: a waiting corpus must observe the
                # same backend failure rather than hang behind a dead owner.
                future.set_exception(exc)
            else:
                future.set_result(vector)
        return future.result()


def _bm25_store(home: Path | None, backend: str) -> BM25Store | BM25SqliteStore:
    """Construct the configured lexical store without silent downgrades."""
    backend = backend.strip().lower()
    if backend in {"", "pickle"}:
        return BM25Store(
            home,
            max_cached_corpora=_env_int("VECGREP_BM25_CACHE_CORPORA", None),
            max_cached_bytes=_env_int("VECGREP_BM25_CACHE_BYTES", None),
        )
    if backend != "sqlite":
        raise ValueError(
            f"unsupported VECGREP_BM25_BACKEND={backend!r}; use 'pickle' or 'sqlite'"
        )

    if home is not None:
        stale = []
        for path in home.glob("*.pkl"):
            database = path.with_suffix(".db")
            if (
                not database.exists()
                or database.stat().st_mtime_ns < path.stat().st_mtime_ns
            ):
                stale.append(path.stem)
        if stale:
            names = ", ".join(sorted(stale))
            raise RuntimeError(
                f"SQLite BM25 backend has unmigrated or stale pickle indexes: {names}; "
                "migrate them before switching backends"
            )
    return BM25SqliteStore(home)

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

# Lowest multiplier recency decay may apply (1.0 disables decay's effect,
# 0.0 restores the pre-2026-08 unfloored curve). 0.5 measured best on the
# eval harness: transcript hit@3 82 -> 84, hit@5 86 -> 89, nothing lost
# elsewhere. See _recency_factor and docs/STORAGE_RETRIEVAL_2026-08.md.
DECAY_FLOOR = float(os.environ.get("VECGREP_DECAY_FLOOR", "0.5"))


def _bm25_fusion_weight(corpus: Corpus | None) -> float:
    """The corpus's pinned BM25 fusion weight, else the install default."""
    w = getattr(corpus, "bm25_weight", None) if corpus is not None else None
    return float(w) if w is not None else BM25_WEIGHT


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
    # DECAY_FLOOR caps how far age alone can push a hit down. Unfloored, a
    # 100-day-old chunk on a 45-day half-life scores at 0.21x, so an EXACT
    # match on an old fact loses to a vague match on a recent one — measured
    # 2026-08-17: a Chinese-language query whose only correct transcript was
    # 100 days old ranked #1 on vector-only and dropped out of the top 10 the
    # moment decay was applied. Recency should break ties, not veto relevance.
    return max(DECAY_FLOOR, 0.5 ** (age_days / half_life_days))


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
    # The source document's own date (epoch seconds), when ingestion could
    # extract one. Drives the stub tier's timestamp column, date filters, and
    # timeline ordering. None for undated sources.
    doc_timestamp: float | None = None
    # Precise source anchors: 1-based inclusive line range of the chunk in
    # its source document, so a caller can cite/re-open the exact region.
    # None when the source text wasn't available at conversion time.
    line_start: int | None = None
    line_end: int | None = None
    # Per-retriever score breakdown — populated when --explain is on.
    # Empty dict otherwise. Keys: vector_cosine, vector_rank, bm25_score,
    # bm25_rank, rrf, rerank_score (when reranked).
    explain: dict = None  # type: ignore[assignment]
    # Deployment-defined origin stamp (source_labels.json glob map) — e.g.
    # which agent/host owns the channel a chat transcript came from. The
    # inline display names frozen into archived text drift; this doesn't.
    source_label: str | None = None

    def __post_init__(self) -> None:
        if self.explain is None:
            self.explain = {}

    @property
    def anchor(self) -> str:
        """Compact citation: 'source_id#L<start>-L<end>' (or just #L<start>
        for one-line chunks; bare source_id when lines are unknown)."""
        if self.line_start is None:
            return self.source_id
        if self.line_end and self.line_end != self.line_start:
            return f"{self.source_id}#L{self.line_start}-L{self.line_end}"
        return f"{self.source_id}#L{self.line_start}"

    @property
    def relevance_pct(self) -> float:
        """The going-forward name for the calibrated display percent.
        `similarity_pct` remains as a compatibility alias of the same value."""
        return self.similarity_pct

    @property
    def relevance_label(self) -> str:
        """Qualitative bucket so callers don't have to interpret percentages:
        exact >= 95, strong >= 75, related >= 40, else weak."""
        pct = self.similarity_pct
        if pct >= 95.0:
            return "exact"
        if pct >= 75.0:
            return "strong"
        if pct >= 40.0:
            return "related"
        return "weak"


class VecgrepService:
    def __init__(
        self,
        settings: Settings | None = None,
        ephemeral: bool = False,
        embed_cache_read_only: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.ephemeral = ephemeral
        self.locks = CorpusLocks(
            None if ephemeral else self.settings.home / "locks"
        )
        self.registry = CorpusRegistry(
            self.settings.corpora_file,
            locks=self.locks,
            in_memory=ephemeral,
        )
        self.store = QdrantStore(
            None if ephemeral else self.settings.qdrant_path,
            url=None if ephemeral else self.settings.qdrant_url,
        )
        # SQLite keeps large lexical indexes on disk; pickle remains the
        # default for compatibility and uses the resident-byte budget.
        self.bm25 = _bm25_store(
            None if ephemeral else self.settings.home / "bm25",
            self.settings.bm25_backend,
        )
        self.explorer_store = ExplorerStore(
            None if ephemeral else self.settings.home / "explorer.db"
        )
        self.mutations = MutationJournal(
            None if ephemeral else self.settings.home / "mutations"
        )
        self._backend_cache: dict[str, EmbedBackend] = {}
        self._backend_cache_lock = threading.RLock()
        self._search_runtime_instance: _SearchRuntime | None = None
        # Source-level explorer catalogs live in a compact, disposable SQLite
        # sidecar. Cache their rendered tree by corpus generation; canonical
        # search data remains Qdrant + BM25.
        self._explorer_cache: dict[
            str, tuple[tuple[float, int, int], ExplorerCatalog]
        ] = {}
        # Embedding cache lives on disk except in ephemeral mode. Wrapping
        # is opt-in per backend in _backend_for() so tests / mocks can
        # bypass it by sticking a backend directly into _backend_cache.
        self._embed_cache: EmbedCache | None = (
            None if ephemeral else EmbedCache(
                self.settings.home / "embed_cache.db",
                read_only=embed_cache_read_only,
            )
        )
        self.recover_pending_mutations()

    # ----- mutation recovery --------------------------------------------------
    def recover_pending_mutations(self) -> list[str]:
        """Repair incomplete corpus commits left by a dead writer.

        Admission is reacquired per corpus. If the journal belongs to a live
        writer this waits; once that writer commits the record disappears and
        there is nothing to recover.

        Also rebuilds any BM25 sidecar left `.dirty` by a bulk index that
        died before its single deferred persist (see BM25Store.bulk): qdrant
        is canonical, so the sidecar is regenerated from it.
        """
        recovered: list[str] = []
        for corpus_name in self.mutations.pending_corpora():
            with self.locks.write(corpus_name):
                if self._recover_corpus_locked(corpus_name):
                    recovered.append(corpus_name)
        for corpus_name in self.bm25.dirty_corpora():
            with self.locks.write(corpus_name):
                if not self.bm25.dirty_corpora().count(corpus_name):
                    continue    # the live writer finished while we waited
                if self.registry.has(corpus_name):
                    self._rebuild_bm25_from_store(self.registry.get(corpus_name))
                else:
                    self.bm25.drop(corpus_name)
                self.bm25.clear_dirty(corpus_name)
                recovered.append(corpus_name)
        return recovered

    def rebuild_bm25(self, corpus_name: str) -> int:
        """Regenerate one corpus's BM25 sidecar from qdrant under the corpus
        write lock. Needed after a tokenizer change (the sidecar stores
        pre-tokenized documents) or whenever the sidecar is suspect. Returns
        the point count rebuilt."""
        with self.locks.write(corpus_name):
            self._recover_corpus_locked(corpus_name)
            corpus = self.registry.get(corpus_name)
            self._rebuild_bm25_from_store(corpus)
            return self.bm25.count(corpus_name)

    def _rebuild_bm25_from_store(self, corpus: Corpus) -> None:
        """Regenerate one corpus's BM25 sidecar from qdrant, the canonical
        payload store. Raises if the point count read back differs."""
        collection = _collection_for(corpus.name)
        expected = self.store.count(collection)

        def records():
            for point_id, payload in self.store.iter_payloads(
                collection,
                exclude_payload_fields={"source_text"},
            ):
                text = payload.get("text")
                if not isinstance(text, str):
                    raise RuntimeError(
                        f"cannot rebuild BM25 for {corpus.name}: {point_id} has no chunk text"
                    )
                # Qdrant is the canonical source payload store. The scroll
                # omits source_text, which would otherwise repeat the same
                # large transcript once per chunk in both RAM and the pickle.
                yield point_id, text, payload

        rebuilt = self.bm25.replace(corpus.name, records())
        if rebuilt != expected:
            self.bm25.drop(corpus.name)
            raise RuntimeError(
                f"BM25 rebuild for {corpus.name} read {rebuilt} points; expected {expected}"
            )

    def _recover_if_pending(self, corpus_name: str) -> None:
        if self.mutations.read(corpus_name) is None:
            return
        with self.locks.write(corpus_name):
            self._recover_corpus_locked(corpus_name)

    def _recover_corpus_locked(self, corpus_name: str) -> bool:
        record = self.mutations.read(corpus_name)
        if record is None:
            return False
        operation = record.get("operation")
        phase = record.get("phase")
        collection = _collection_for(corpus_name)

        if operation == "delete_corpus":
            self.store.drop_collection(collection)
            self.bm25.drop(corpus_name)
            self._drop_explorer_catalog(corpus_name)
            if self.registry.has(corpus_name):
                self.registry.delete(corpus_name)
            self.mutations.finish(corpus_name)
            return True

        if operation not in {"index_source", "delete_source"}:
            raise CorpusError(
                f"unknown pending mutation {operation!r} for {corpus_name}"
            )

        before_data = record.get("corpus_before")
        target_data = record.get("corpus_target")
        source_id = str(record.get("source_id") or "")

        if operation == "index_source" and phase == "prepared":
            # Qdrant may have accepted zero, some, or all batches before the
            # writer died. Restore the exact old point set, then rebuild both
            # derivatives from that canonical state.
            metadata = before_data
            old_points = list(record.get("old_points") or [])
            if old_points and metadata:
                self.store.ensure_collection(
                    collection, int(metadata["dim"]),
                    datatype=str(metadata.get("datatype") or "float32"),
                )
            self.store.restore_source(collection, source_id, old_points)
        elif operation == "delete_source":
            # A persisted delete intent is always completed. The source file
            # is outside vecgrep's authority and may already be gone.
            self.store.delete_by_source(collection, source_id)
            metadata = target_data or before_data
        else:
            # Qdrant finished before the phase marker landed. Complete BM25 +
            # registry from Qdrant; both operations are deterministic.
            metadata = target_data or before_data

        live_count = self.store.count(collection)
        if metadata is None and live_count == 0:
            self.store.drop_collection(collection)
            self.bm25.drop(corpus_name)
            self._drop_explorer_catalog(corpus_name)
            if self.registry.has(corpus_name):
                self.registry.delete(corpus_name)
            self.mutations.finish(corpus_name)
            return True
        if metadata is None:
            raise CorpusError(
                f"cannot recover {corpus_name}: live points have no corpus metadata"
            )

        corpus = Corpus(**metadata)
        sources: list[str] = []
        seen_sources: set[str] = set()
        source_hashes: dict[str, str] = {}
        bm25_records: list[tuple[str, str, dict]] = []
        for point_id, payload in self.store.iter_payloads(collection):
            text = payload.get("text")
            if not isinstance(text, str):
                raise CorpusError(
                    f"cannot recover {corpus_name}: {point_id} has no chunk text"
                )
            compact_payload = dict(payload)
            compact_payload.pop("source_text", None)
            bm25_records.append((point_id, text, compact_payload))
            sid = payload.get("source_id")
            if isinstance(sid, str):
                # Dedup the source list on its own. Gating it on the hash
                # (as this used to) appends a source once per chunk whenever
                # the hash can't be taken — which is now every non-carrier
                # chunk, since only chunk 0 holds the document.
                if sid not in seen_sources:
                    seen_sources.add(sid)
                    sources.append(sid)
                source_text = payload.get("source_text")
                if isinstance(source_text, str) and source_text and sid not in source_hashes:
                    source_hashes[sid] = hashlib.sha256(
                        source_text.encode("utf-8")
                    ).hexdigest()

        self.bm25.replace(corpus_name, bm25_records)
        # The catalog is derived. Invalidate it here and let the next Browse
        # request backfill from the freshly recovered BM25 sidecar.
        self._drop_explorer_catalog(corpus_name)
        corpus.sources = sources
        corpus.source_hashes = source_hashes
        corpus.doc_count = len(sources)
        corpus.chunk_count = len(bm25_records)
        corpus.updated_at = time.time()
        self.registry.upsert(corpus)
        self.mutations.finish(corpus_name)
        return True

    # ----- backend resolution ---------------------------------------------------
    def _backend_for(self, corpus: Corpus | None) -> EmbedBackend:
        prefer = corpus.embed_backend if corpus else None
        # Resolve to the corpus's OWN pinned model, not the global env model.
        # This is what lets one running service query corpora embedded with
        # different models — each gets its own backend, keyed by (backend,model).
        model = corpus.embed_model if corpus else None
        cache_key = f"{prefer}:{model}" if (prefer and model) else (prefer or "auto")
        with self._backend_cache_lock:
            if cache_key not in self._backend_cache:
                # Reuse an already-resolved 'auto' backend if it happens to
                # match the corpus's pinned backend AND model — avoids a
                # redundant live resolve (and lets tests inject one mock).
                auto = self._backend_cache.get("auto")
                if (
                    auto is not None
                    and auto.name == prefer
                    and (model is None or auto.model == model)
                ):
                    self._backend_cache[cache_key] = auto
                else:
                    raw = get_embed_backend(self.settings, prefer=prefer, model=model)
                    self._backend_cache[cache_key] = (
                        CachedBackend(raw, self._embed_cache)
                        if self._embed_cache
                        else raw
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

    def _search_runtime(self) -> _SearchRuntime:
        runtime = self._search_runtime_instance
        if runtime is None:
            runtime = _search_runtime_for(self.settings)
            self._search_runtime_instance = runtime
        return runtime

    def _embed_query_singleflight(self, corpus: Corpus, query: str) -> list[float]:
        return self._search_runtime().query_vector(self, corpus, query)

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
            with self._backend_cache_lock:
                # Do not evict a healthy replacement another request resolved
                # after this request captured its now-stale backend reference.
                if self._backend_cache.get(key) is backend:
                    self._backend_cache.pop(key, None)
                if self._backend_cache.get("auto") is backend:
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
        **kwargs,
    ) -> tuple[int, int, int]:
        """Serialize one corpus mutation across every process and surface."""
        with self.locks.write(corpus_name):
            self._recover_corpus_locked(corpus_name)
            return self._index_locked(
                source,
                corpus_name,
                chunker_name,
                force,
                include,
                **kwargs,
            )

    def _index_locked(
        self,
        source: str,
        corpus_name: str,
        chunker_name: str = "sentence_window",
        force: bool = False,
        include: str | None = None,
        *,
        bypass_embed_cache: bool | None = None,
        update_bm25: bool = True,
        update_registry: bool = True,
        replace_existing: bool = True,
        resume_source_counts: dict[str, int] | None = None,
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
            self.store.ensure_collection(
                _collection_for(corpus_name), corpus.dim,
                datatype=corpus.datatype,
            )

        # Incremental updates may only advance a catalog that was complete for
        # the prior corpus generation. On an upgrade, explorer.db starts empty
        # while BM25 already has years of sources; stamping the first changed
        # document complete would hide every untouched source.
        catalog_sync_ok = (
            corpus.doc_count == 0
            and corpus.chunk_count == 0
            and not corpus.sources
        )
        if not catalog_sync_ok:
            try:
                catalog_sync_ok = (
                    self.explorer_store.generation(corpus.name)
                    == self._explorer_generation(corpus)
                )
            except Exception as exc:
                logger.warning(
                    "explorer catalog state read failed for %s: %s",
                    corpus_name,
                    exc,
                )

        backend = self._backend_for(corpus)
        chunker = self.chunker(chunker_name)
        collection = _collection_for(corpus_name)

        # --force should also bust the embedding cache for these texts;
        # otherwise we just re-write identical vectors. The wrapped backend
        # has a `bypass` flag we toggle for the duration of this call.
        prev_bypass = getattr(backend, "bypass", None)
        use_bypass = force if bypass_embed_cache is None else bypass_embed_cache
        if use_bypass and hasattr(backend, "bypass"):
            backend.bypass = True

        total_docs = 0
        total_chunks = 0
        skipped = 0
        docs = _expand(source, adapter, include=include)
        # A multi-source index (a directory) defers the BM25 sidecar write to
        # the end of the batch: persisting per source re-pickled the whole
        # growing index N times (O(N^2) bytes -- see BM25Store.bulk).
        _bulk = contextlib.ExitStack()
        # SQLite must commit before each source intent is retired. Its bulk
        # transaction would otherwise roll back earlier sources after their
        # Qdrant writes and journal removals have already committed.
        if update_bm25 and len(docs) > 1 and not isinstance(self.bm25, BM25SqliteStore):
            _bulk.enter_context(self.bm25.bulk(corpus_name))
        with _bulk:
            for doc in docs:
                # doc-aware chunkers (code_symbol) see the source path for
                # language detection; text-only chunkers keep the old contract
                chunks = (chunker.chunk_doc(doc)
                          if hasattr(chunker, "chunk_doc") else chunker.chunk(doc.text))
                if not chunks:
                    continue

                doc_hash = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()
                matches_recorded_source = corpus.source_hashes.get(doc.source_id) == doc_hash
                if (
                    resume_source_counts is not None
                    and matches_recorded_source
                    and resume_source_counts.get(doc.source_id) == len(chunks)
                ):
                    # This source already has every deterministic point for the
                    # exact content recorded in the registry. Leave it alone.
                    skipped += 1
                    continue
                if not force and corpus.source_hashes.get(doc.source_id) == doc_hash:
                    # Already indexed at this exact content — skip embed call.
                    skipped += 1
                    continue

                # Normal re-indexing removes a prior version so a source that
                # shrank cannot leave old tail chunks behind. A partial Qdrant
                # recovery can skip that scan: point IDs are deterministic for the
                # same corpus/source/chunk index, so upsert overwrites completed
                # sources and fills only the missing points.
                if replace_existing and not (
                    resume_source_counts is not None and matches_recorded_source
                ):
                    old_records = self.store.source_records(collection, doc.source_id)
                else:
                    old_records = []

                journaled = update_bm25 and update_registry
                if journaled:
                    target = Corpus(**asdict(corpus))
                    if doc.source_id not in target.sources:
                        target.sources.append(doc.source_id)
                    target.source_hashes[doc.source_id] = doc_hash
                    self.mutations.write({
                        "version": 1,
                        "corpus": corpus_name,
                        "operation": "index_source",
                        "phase": "prepared",
                        "source_id": doc.source_id,
                        "corpus_before": asdict(corpus) if self.registry.has(corpus_name) else None,
                        "corpus_target": asdict(target),
                        "old_points": old_records,
                    })

                if replace_existing and not (
                    resume_source_counts is not None and matches_recorded_source
                ):
                    self.store.delete_by_source(collection, doc.source_id)
                if update_bm25:
                    self.bm25.delete_by_source(corpus_name, doc.source_id)

                ids = [_chunk_id(corpus_name, doc.source_id, c.index) for c in chunks]
                vectors = backend.embed([c.text for c in chunks])
                payloads = [
                    {
                        "corpus": corpus_name,
                        "source_id": doc.source_id,
                        # The full document rides on chunk 0 only — "the carrier".
                        # Duplicating it onto every chunk cost ~8.3 GB across a
                        # 186k-point install and grows as S^2/chunk_size in
                        # document size. Whole-document readers fetch the carrier
                        # in O(1) via _source_text_for; per-chunk context comes
                        # from the precomputed fields below.
                        **({"source_text": doc.text} if c.index == 0 else {}),
                        **_chunk_context_fields(doc.text, c.start, c.end),
                        "chunk_index": c.index,
                        "chunk_start": c.start,
                        "chunk_end": c.end,
                        "text": c.text,
                        # Doc metadata + per-chunk enrichment (speakers, bot flag,
                        # content shapes) — powers speaker:/bot:/has: filters.
                        "metadata": {**doc.metadata, **chunk_enrichment(c.text),
                                     **(getattr(c, "meta", None) or {})},
                        # Document's own date (epoch seconds) when discoverable.
                        # Powers optional recency decay at search time. None is fine.
                        "doc_timestamp": doc.timestamp,
                    }
                    for c in chunks
                ]
                self.store.ensure_collection(
                    collection, corpus.dim, datatype=corpus.datatype,
                )
                self.store.upsert(collection, ids, vectors, payloads)
                if journaled:
                    record = self.mutations.read(corpus_name) or {}
                    record["phase"] = "qdrant_done"
                    self.mutations.write(record)
                if update_bm25:
                    self.bm25.upsert(corpus_name, ids, [c.text for c in chunks], payloads)
                    try:
                        self.explorer_store.upsert(
                            corpus_name,
                            {
                                "source_id": doc.source_id,
                                "metadata": payloads[0].get("metadata") or {},
                                "doc_timestamp": doc.timestamp,
                                "chunk_count": len(chunks),
                            },
                        )
                    except Exception as exc:
                        # Browse metadata is disposable. Never fail canonical
                        # indexing because its derived cache needs a rebuild.
                        catalog_sync_ok = False
                        logger.warning(
                            "explorer catalog update failed for %s: %s",
                            corpus_name,
                            exc,
                        )
                    if journaled:
                        record = self.mutations.read(corpus_name) or {}
                        record["phase"] = "bm25_done"
                        self.mutations.write(record)

                total_docs += 1
                total_chunks += len(chunks)
                if doc.source_id not in corpus.sources:
                    corpus.sources.append(doc.source_id)
                corpus.source_hashes[doc.source_id] = doc_hash

                if update_registry:
                    corpus.doc_count = len(corpus.sources)
                    corpus.chunk_count = self.store.count(collection)
                    corpus.updated_at = time.time()
                    if self.ephemeral:
                        self.registry._corpora[corpus.name] = corpus
                    else:
                        self.registry.upsert(corpus)
                    if catalog_sync_ok:
                        try:
                            self.explorer_store.set_generation(
                                corpus.name, self._explorer_generation(corpus)
                            )
                        except Exception as exc:
                            catalog_sync_ok = False
                            logger.warning(
                                "explorer catalog commit failed for %s: %s",
                                corpus_name,
                                exc,
                            )
                    if journaled:
                        record = self.mutations.read(corpus_name) or {}
                        record["phase"] = "registry_done"
                        self.mutations.write(record)
                        self.mutations.finish(corpus_name)

        # During recovery, do not persist an intermediate point count as the
        # corpus's expected total. If interrupted, diagnose() must retain count
        # drift instead of accepting a partial collection as healthy.
        if not update_registry:
            if prev_bypass is not None and hasattr(backend, "bypass"):
                backend.bypass = prev_bypass
            return total_docs, total_chunks, skipped

        # Normal mutations commit the registry per source so every journal
        # record has a complete recovery boundary. Maintenance callers may
        # deliberately defer registry publication until a whole repair ends.
        if update_registry and total_docs == 0 and self.registry.has(corpus_name):
            corpus = self.registry.get(corpus_name)

        # Restore the bypass flag for subsequent calls (other corpora etc.)
        if prev_bypass is not None and hasattr(backend, "bypass"):
            backend.bypass = prev_bypass

        return total_docs, total_chunks, skipped

    # ----- search ---------------------------------------------------------------
    def _searchable_corpora(self) -> list[Corpus]:
        """Corpora an unscoped search fans out over.

        Skips names matching settings.cross_corpus_exclude (fnmatch). Never
        applied to an explicitly named corpus -- see search().
        """
        return [c for c in self.registry.list()
                if not self.is_hidden_corpus(c.name)]

    def is_hidden_corpus(self, name: str) -> bool:
        """Is this corpus a build artifact rather than one someone curates?

        Reads settings.cross_corpus_exclude -- the same patterns that keep an
        eval-* copy out of an unscoped fan-out. Deliberately one setting, so a
        corpus cannot end up hidden from search but still offered in a picker.

        Callers that present a LIST use this; list_corpora() itself stays
        unfiltered, because backup and the eval harness read it and a backup
        that silently skips corpora is a much worse bug than an untidy list.
        """
        patterns = list(getattr(self.settings, "cross_corpus_exclude", None) or [])
        return any(fnmatch.fnmatch(name, pat) for pat in patterns)

    def search_with_diagnostics(
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
        expand_aliases: bool = True,
        corpus_names: list[str] | None = None,
    ) -> SearchOutcome:
        started = time.monotonic()
        top_k = top_k or self.settings.default_top_k
        if expand_aliases:
            # Entity alias expansion (user-supplied map, outside the repo;
            # empty/missing map = exact no-op). A query naming one surface
            # form of an entity also matches evidence written under its
            # other forms.
            alias_map = load_alias_map_cached(aliases_path())
            if alias_map:
                query, _matched = expand_query(query, alias_map)
        if corpus_name is not None and corpus_names is not None:
            raise CorpusError("corpus and corpora are mutually exclusive")
        plural_scope = corpus_names is not None
        if corpus_names is not None:
            if not corpus_names:
                raise CorpusError("corpora cannot be empty")
            names = list(dict.fromkeys(corpus_names))
            # Resolve every requested name before beginning any search. This
            # avoids a typo returning a misleading partial result.
            corpora = [self.registry.get(name) for name in names]
        elif corpus_name:
            # An explicitly named corpus is always searched, exclusions or not
            # -- otherwise the eval harness could not query its own build.
            corpora = [self.registry.get(corpus_name)]
        else:
            corpora = self._searchable_corpora()
        if not corpora:
            if corpus_name is None:
                logger.info(
                    "cross-corpus search requested=[] successful=[] failed=[] "
                    "result_counts={} elapsed_ms=%d",
                    round((time.monotonic() - started) * 1000),
                )
            return SearchOutcome([], [])

        # Pull a wider pool than top_k whenever post-retrieval steps can shrink
        # the set — filtering, reranking, OR dedup — so we don't return fewer
        # than top_k results just because the top hits got filtered/collapsed.
        # Budgeted searches ask for top_k > CANDIDATE_POOL; widen so the pool
        # itself never silently clips the requested breadth.
        per_corpus_k = max(CANDIDATE_POOL, top_k)

        results: list[SearchResult] = []
        warnings: list[SearchWarning] = []
        successful: list[str] = []
        failures: list[tuple[Corpus, Exception]] = []
        query_vectors = (
            _QueryVectorMemo(self, query)
            if len(corpora) > 1 and mode in ("hybrid", "vector")
            else None
        )
        workers = min(
            len(corpora), max(1, int(getattr(self.settings, "search_fanout_workers", 8) or 1))
        )
        if workers > 1 and len(corpora) > 1:
            # Serial fan-out made unscoped latency the SUM of per-corpus cost.
            # Each _search_one is an independent read (its own qdrant query and
            # BM25 lookup), so they overlap cleanly. The executor is shared by
            # every request using this settings generation: concurrent bots
            # queue behind one bound instead of creating N request-local pools.
            # Results are still consumed in corpus order, not completion order,
            # so ranking and warnings remain deterministic.
            pool = self._search_runtime().executor
            pending = [
                (c, pool.submit(
                    self._search_one, c, query, per_corpus_k, mode,
                    explain=explain,
                    query_vectors=query_vectors,
                ))
                for c in corpora
            ]
            # Consume in request/registry order, not completion order, so
            # both results and warnings remain deterministic.
            for c, future in pending:
                try:
                    results.extend(future.result())
                    successful.append(c.name)
                except Exception as exc:
                    failures.append((c, exc))
        else:
            for c in corpora:
                try:
                    results.extend(self._search_one(
                        c, query, per_corpus_k, mode, explain=explain,
                        query_vectors=query_vectors,
                    ))
                    successful.append(c.name)
                except Exception as exc:
                    failures.append((c, exc))

        partial_allowed = corpus_name is None
        if failures and (not partial_allowed or not successful):
            if corpus_name is None or plural_scope:
                logger.info(
                    "cross-corpus search requested=%s successful=%s failed=%s "
                    "result_counts=%s elapsed_ms=%d",
                    [c.name for c in corpora],
                    successful,
                    [c.name for c, _ in failures],
                    {name: 0 for name in successful},
                    round((time.monotonic() - started) * 1000),
                )
            raise failures[0][1]
        warnings = [
            SearchWarning(
                corpus=c.name,
                code="search_failed",
                message=f"{type(exc).__name__}: corpus search failed",
            )
            for c, exc in failures
        ]

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

        # Collapse near-duplicate chunks from the same source before selecting,
        # so adjacent sentence-windows and repeated messages (bot alert spam,
        # quoted replies) don't each eat a top_k slot.
        results = dedup_near_duplicates(results)

        # A cold cross-encoder takes minutes to load and would hold a sync
        # threadpool slot for all of it (see rerank.RERANK_WAIT_S). Falling
        # through to the plain path costs accuracy on one search; blocking
        # costs the server.
        reranked = False
        if rerank and self._rerank_ready(rerank_model):
            try:
                results = self._apply_rerank(
                    query, results, top_k, rerank_model, explain=explain)
                reranked = True
            except Exception as exc:
                # Readiness only proves the model LOADED. predict() can still
                # blow up later — a wedged GPU is the live example: the host's
                # card started reporting a garbage VRAM figure and every
                # rerank raised `CUDA error: unknown error`, which 500'd the
                # whole search for every bot in the squad (2026-08-20).
                # Reranking is an enhancement over a result set we already
                # have; losing it must degrade the answer, never fail it.
                logger.warning(
                    "rerank failed (%s: %s) — returning fusion order",
                    type(exc).__name__, exc)
        if not reranked:
            # Diversity-aware top_k: MMR keeps relevance dominant but skips
            # near-clones of already-selected results, so the returned set is
            # distinct evidence, not five slices of one exchange. On corpora
            # with no near-dups this degrades to plain score order.
            results = mmr_select(results, top_k)
            # Display order follows similarity_pct scaled by the corpus rank
            # weight — identical to raw pct order when weights are neutral;
            # where they differ, a curated corpus deliberately edges out a
            # transcript hit of comparable %, which is the point of weighting.
            weights = {c.name: (getattr(c, "rank_weight", 1.0) or 1.0) for c in corpora}
            results.sort(key=lambda r: r.similarity_pct * weights.get(r.corpus, 1.0),
                         reverse=True)
        if corpus_name is None or plural_scope:
            counts = {name: 0 for name in successful}
            for result in results:
                if result.corpus in counts:
                    counts[result.corpus] += 1
            logger.info(
                "cross-corpus search requested=%s successful=%s failed=%s "
                "result_counts=%s elapsed_ms=%d",
                [c.name for c in corpora],
                successful,
                [c.name for c, _ in failures],
                counts,
                round((time.monotonic() - started) * 1000),
            )
        # Origin stamp (deployment-supplied glob map, outside the repo;
        # missing map = exact no-op). Applied here so every assembly built on
        # this — search, budget, timeline, incident — inherits the stamp.
        apply_labels(results, load_source_labels_cached(source_labels_path()))
        return SearchOutcome(results, warnings)

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
        expand_aliases: bool = True,
        corpus_names: list[str] | None = None,
    ) -> list[SearchResult]:
        """Backward-compatible search result list without diagnostics."""
        return self.search_with_diagnostics(
            query,
            corpus_name,
            top_k,
            mode=mode,
            rerank=rerank,
            rerank_model=rerank_model,
            filters=filters,
            explain=explain,
            include_superseded=include_superseded,
            expand_aliases=expand_aliases,
            corpus_names=corpus_names,
        ).results

    def search_budgeted(
        self,
        query: str,
        corpus_name: str | None = None,
        full_k: int = DEFAULT_FULL_K,
        max_total: int = DEFAULT_MAX_TOTAL,
        token_ceiling: int = DEFAULT_STUB_TOKEN_CEILING,
        **kwargs,
    ) -> tuple[list[SearchResult], list[ResultStub]]:
        """Breadth search: (full head, stub tail).

        Runs a normal search() with top_k=max_total (dedup + MMR apply, so
        the widened budget is spent on distinct evidence), then splits: the
        top full_k results keep context windows, the rest degrade to
        one-line stubs until `token_ceiling` estimated tokens. Stubs carry
        chunk_id — expand any of them via get_chunk_window / /api/chunk /
        MCP get_chunk. Extra kwargs pass through to search() (mode, rerank,
        filters, ...).
        """
        results = self.search(query, corpus_name, top_k=max_total, **kwargs)
        return split_full_and_stubs(
            results, full_k=full_k, max_total=max_total, token_ceiling=token_ceiling
        )

    def search_budgeted_with_diagnostics(
        self,
        query: str,
        corpus_name: str | None = None,
        full_k: int = DEFAULT_FULL_K,
        max_total: int = DEFAULT_MAX_TOTAL,
        token_ceiling: int = DEFAULT_STUB_TOKEN_CEILING,
        **kwargs,
    ) -> tuple[list[SearchResult], list[ResultStub], list[SearchWarning]]:
        outcome = self.search_with_diagnostics(
            query, corpus_name, top_k=max_total, **kwargs
        )
        full, stubs = split_full_and_stubs(
            outcome.results,
            full_k=full_k,
            max_total=max_total,
            token_ceiling=token_ceiling,
        )
        return full, stubs, outcome.warnings

    def timeline(
        self,
        query: str,
        corpus_name: str | None = None,
        top_k: int = ANCHOR_TOP_K,
        max_groups: int = MAX_GROUPS,
        padding: int = SLICE_PADDING,
        **kwargs,
    ) -> list[dict]:
        """'What happened?' mode: contiguous chronological slices, not chunks.

        Runs a normal search (dedup/MMR/filters all apply) to find anchor
        chunks, keeps the best `max_groups` source files, cuts ONE contiguous
        slice per file spanning its anchors (+`padding` chars each side), and
        parses transcript slices into (speaker, time, text) events — in
        document order, which for transcripts IS chronological order. Groups
        come back oldest → newest. Non-transcript sources return the raw
        slice with no events. Extra kwargs pass to search() (mode, filters…).
        """
        anchors = self.search(query, corpus_name, top_k=top_k, **kwargs)
        return build_timeline(
            anchors,
            self._payload_for_result,
            max_groups=max_groups,
            padding=padding,
            source_text_for=self._source_text_for,
        )

    def incident(
        self,
        query: str,
        corpus_name: str | None = None,
        **kwargs,
    ) -> dict | None:
        """One structured answer for "what happened?" — pure assembly over
        the existing primitives (search anchors → timeline groups), no new
        retrieval.

        Returns None when nothing matches; otherwise:
          title             — the caller's question, verbatim (no invention)
          confidence        — best anchor's relevance label
          sources           — every file contributing evidence
          participants      — distinct speakers across all events
          time_range        — {"start","end"}: "<YYYY-MM-DD> <HH:MM>" bounds
          primary_source    — the file holding the strongest evidence
          primary_timeline  — that file's chronological events
          related           — the other timeline groups, kept separate so
                              context never blends into the primary narrative
        """
        anchors = self.search(query, corpus_name, top_k=ANCHOR_TOP_K, **kwargs)
        if not anchors:
            return None
        groups = build_timeline(
            anchors,
            lambda r: self._payload_for_result(r),
            source_text_for=self._source_text_for,
        )
        if not groups:
            return None

        primary_sid = max(anchors, key=lambda a: a.score).source_id
        primary = next(
            (g for g in groups if g["source_id"] == primary_sid), groups[0]
        )
        related = [g for g in groups if g is not primary]

        def _day(g: dict) -> str:
            if not g["doc_timestamp"]:
                return ""
            return datetime.fromtimestamp(
                g["doc_timestamp"], tz=timezone.utc
            ).strftime("%Y-%m-%d")

        stamps = [
            f"{_day(g)} {e['time']}".strip()
            for g in groups
            for e in g["events"]
        ]
        participants = sorted(
            {e["speaker"] for g in groups for e in g["events"]}
        )
        return {
            "title": query,
            "confidence": anchors[0].relevance_label,
            "sources": [g["source_id"] for g in groups],
            "participants": participants,
            "time_range": {
                "start": min(stamps) if stamps else "",
                "end": max(stamps) if stamps else "",
            },
            "primary_source": primary["source_id"],
            "primary_timeline": primary["events"],
            "related": related,
        }

    def browse(
        self,
        corpus_name: str,
        channel: str | None = None,
        date: str | None = None,
        source_path: str | None = None,
        since: str | None = None,
        until: str | None = None,
        tail: int | None = None,
    ) -> list[dict]:
        """Location-first reading: no query, no ranking — the full event
        sequence of the sources matching channel / date / path glob.

        Search answers "where did we say X?"; browse answers "show me
        channel Y on day Z". At least one selector is required — refusing a
        bare corpus dump keeps a fat corpus from flooding the caller.
        Groups come back oldest → newest, events chronological (document
        order). Non-transcript sources return raw text, no events.
        """
        if not (channel or date or source_path or since or until):
            raise ValueError(
                "browse needs at least one selector: channel, date, "
                "since/until, or source_path"
            )
        corpus = self.registry.get(corpus_name)
        day_start = _parse_filter_time(date) if date else None
        if date and day_start is None:
            raise ValueError(f"unparseable date: {date!r} (want YYYY-MM-DD)")
        since_ts = _parse_filter_time(since) if since else None
        if since and since_ts is None:
            raise ValueError(f"unparseable since: {since!r} (want YYYY-MM-DD)")
        until_ts = _parse_filter_time(until) if until else None
        if until and until_ts is None:
            raise ValueError(f"unparseable until: {until!r} (want YYYY-MM-DD)")

        groups: list[dict] = []
        for source_id, payload, _chunk_count in self.bm25.iter_sources(corpus.name):
            meta = payload.get("metadata") or {}
            if channel is not None:
                have = str(meta.get("channel", "")).strip().strip("\"'")
                if have != channel:
                    continue
            ts = payload.get("doc_timestamp")
            if day_start is not None:
                if ts is None or not (
                    day_start <= float(ts) < day_start + _SECONDS_IN_DAY
                ):
                    continue
            if source_path is not None and not fnmatch.fnmatch(
                source_id, source_path
            ):
                continue
            # since/until: inclusive day range (until covers its whole day).
            if since_ts is not None and (ts is None or float(ts) < since_ts):
                continue
            if until_ts is not None and (
                ts is None or float(ts) >= until_ts + _SECONDS_IN_DAY
            ):
                continue
            # browse reads whole documents ("channel Y on day Z", "the last
            # N messages"), so this must be the full text, not the hit chunk.
            source_text = self._source_text_for(corpus.name, source_id)
            events = parse_events(source_text)
            groups.append(
                {
                    "corpus": corpus.name,
                    "source_id": source_id,
                    "doc_timestamp": float(ts) if ts is not None else None,
                    "events": [
                        {"speaker": e.speaker, "time": e.time, "text": e.text}
                        for e in events
                    ],
                    "slice_text": "" if events else source_text,
                }
            )
        groups.sort(
            key=lambda g: (g["doc_timestamp"] is None, g["doc_timestamp"] or 0.0)
        )
        # tail=N: keep only the NEWEST N events across the matched sources —
        # "the last N messages", not the last N days. Walk newest-first,
        # trimming the oldest events of the boundary group; groups that fall
        # entirely outside the window are dropped. A non-transcript source
        # (no parsed events) counts as one slot.
        if tail is not None and int(tail) > 0:
            remaining = int(tail)
            kept: list[dict] = []
            for g in reversed(groups):
                evs = g["events"]
                if evs:
                    if len(evs) > remaining:
                        g["events"] = evs[-remaining:]
                    remaining -= len(g["events"])
                else:
                    remaining -= 1
                kept.append(g)
                if remaining <= 0:
                    break
            kept.reverse()
            groups = kept
        return groups

    def get_source(self, corpus_name: str, source_id: str) -> dict | None:
        """Whole source document by source_id: raw text + parsed events (for
        transcripts) + timestamp/metadata. The by-source complement of
        get_chunk_window — a stub or corpus listing hands you the source_id;
        this hands back the file. None when the source isn't indexed."""
        corpus = self.registry.get(corpus_name)
        first = self.bm25.first_chunk_for_source(corpus.name, source_id)
        if first is None:
            return None
        first_cid, payload = first
        # A recovery-built BM25 sidecar intentionally omits the duplicated
        # full source document. Qdrant is the canonical payload store, so use
        # it when available and retain BM25 only as the offline fallback.
        qdrant_payload = self.store.get_by_id(
            _collection_for(corpus.name), first_cid
        )
        if qdrant_payload is not None:
            payload = qdrant_payload
        # Whole document, not the hit's chunk — resolved from the carrier.
        source_text = self._source_text_for(corpus.name, source_id)
        ts = payload.get("doc_timestamp")
        events = parse_events(source_text)
        return {
            "corpus": corpus.name,
            "source_id": source_id,
            "doc_timestamp": float(ts) if ts is not None else None,
            "metadata": payload.get("metadata", {}) or {},
            "text": source_text,
            "events": [
                {"speaker": e.speaker, "time": e.time, "text": e.text}
                for e in events
            ],
        }

    def _explorer_catalog_for(self, corpus_name: str) -> ExplorerCatalog:
        corpus = self.registry.get(corpus_name)
        generation = self._explorer_generation(corpus)
        cached = self._explorer_cache.get(corpus.name)
        if cached is not None and cached[0] == generation:
            return cached[1]

        try:
            catalog_is_current = (
                self.explorer_store.generation(corpus.name) == generation
            )
            records = (
                self.explorer_store.records(corpus.name)
                if catalog_is_current
                else None
            )
        except Exception as exc:
            logger.warning(
                "explorer catalog read failed for %s: %s", corpus.name, exc
            )
            records = None
        if records is None:
            # One-time path for existing installations (and self-healing after
            # catalog loss): collapse the canonical chunk sidecar, then persist
            # the compact source rows so future process starts stay cheap.
            records = []
            for source_id, payload, chunk_count in self.bm25.iter_sources(corpus.name):
                records.append(
                    {
                        "source_id": source_id,
                        "metadata": payload.get("metadata") or {},
                        "doc_timestamp": payload.get("doc_timestamp"),
                        "chunk_count": chunk_count,
                    }
                )
            try:
                self.explorer_store.replace(corpus.name, records, generation)
            except Exception as exc:
                logger.warning(
                    "explorer catalog backfill persist failed for %s: %s",
                    corpus.name,
                    exc,
                )
        catalog = build_catalog(records)
        self._explorer_cache[corpus.name] = (generation, catalog)
        return catalog

    @staticmethod
    def _explorer_generation(corpus: Corpus) -> CatalogGeneration:
        return corpus.updated_at, corpus.doc_count, corpus.chunk_count

    def _drop_explorer_catalog(self, corpus_name: str) -> None:
        try:
            self.explorer_store.drop(corpus_name)
        except Exception as exc:
            logger.warning(
                "explorer catalog drop failed for %s: %s", corpus_name, exc
            )

    def explore(
        self,
        corpus_name: str,
        *,
        path: list[str] | None = None,
        query: str = "",
        sort: str = "name",
        offset: int = 0,
        limit: int = 50,
    ) -> dict:
        """Paginated source-level navigation for one corpus."""
        with self.locks.read(corpus_name):
            return list_catalog(
                self._explorer_catalog_for(corpus_name),
                corpus=corpus_name,
                path=path,
                query=query,
                sort=sort,
                offset=offset,
                limit=limit,
            )

    def explorer_source(
        self,
        corpus_name: str,
        source_id: str,
        *,
        max_chars: int = 100_000,
    ) -> dict | None:
        """Resolve a source's explorer home and return a bounded text preview."""
        if max_chars < 1:
            raise ValueError("max_chars must be at least 1")
        with self.locks.read(corpus_name):
            catalog = self._explorer_catalog_for(corpus_name)
            entry = catalog.find(source_id)
            if entry is None:
                return None
            source = self.get_source(corpus_name, source_id)
            if source is None:
                return None
            text = str(source.get("text") or "")
            return {
                **entry.summary(),
                "corpus": corpus_name,
                "metadata": source.get("metadata") or {},
                "text": text[:max_chars],
                "source_length": len(text),
                "truncated": len(text) > max_chars,
            }

    # ----- v1.2 tools: related / compare / stats / summarize ----------------

    def related(
        self, chunk_id: str, corpus_name: str, top_k: int = 8
    ) -> list[SearchResult]:
        """Nearest neighbours of an EXISTING chunk by its stored vector —
        query-by-example without re-embedding anything. Excludes the seed
        chunk and collapses near-duplicate slices of the same exchange with
        the same dedup the normal search path uses."""
        corpus = self.registry.get(corpus_name)
        collection = _collection_for(corpus.name)
        vector = self.store.get_vector(collection, chunk_id)
        if vector is None:
            raise CorpusError(
                f"chunk '{chunk_id}' not found in corpus '{corpus_name}'"
            )
        hits = self.store.search(collection, vector, top_k * 3 + 1)
        results: list[SearchResult] = []
        for h in hits:
            r = _hit_to_result(h, ["vector"], model=corpus.embed_model)
            if r.chunk_id == chunk_id:
                continue
            results.append(r)
        return dedup_near_duplicates(results)[:top_k]

    def compare(
        self,
        query: str,
        corpus_name: str,
        a_after: str | None = None,
        a_before: str | None = None,
        b_after: str | None = None,
        b_before: str | None = None,
        top_k: int = 8,
        mode: str = "hybrid",
    ) -> dict:
        """Temporal diff: the same query run in two time windows, side by
        side, plus the source-level delta. Answers "how did we talk about X
        then vs now" — the institutional-memory question a flat search can't.
        Window values accept everything after:/before: accept (ISO or
        relative like '30d'). The delta is source-set based; the caller (an
        LLM, usually) narrates the content change from the two result sets."""

        def _side(after: str | None, before: str | None) -> dict:
            filters = []
            if after:
                filters.append(f"after:{after}")
            if before:
                filters.append(f"before:{before}")
            res = self.search(
                query, corpus_name, top_k=top_k, mode=mode, filters=filters
            )
            return {
                "results": res,
                "sources": sorted({r.source_id for r in res}),
            }

        a = _side(a_after, a_before)
        b = _side(b_after, b_before)
        a_set, b_set = set(a["sources"]), set(b["sources"])
        return {
            "windows": {
                "a": {"after": a_after, "before": a_before},
                "b": {"after": b_after, "before": b_before},
            },
            "a": a,
            "b": b,
            "only_in_a": sorted(a_set - b_set),
            "only_in_b": sorted(b_set - a_set),
            "in_both": sorted(a_set & b_set),
        }

    def corpus_stats(self, corpus_name: str) -> dict:
        """Ops-facing health snapshot: counts, date coverage, gaps (days with
        zero chunks inside the covered span — a broken archiver shows up as a
        growing gap), and per-source chunk sizes."""
        corpus = self.registry.get(corpus_name)
        payloads = list(self.bm25.iter_payloads(corpus.name))
        by_source: dict[str, int] = {}
        for p in payloads:
            sid = p.get("source_id", "") or ""
            by_source[sid] = by_source.get(sid, 0) + 1
        ts = [float(p["doc_timestamp"]) for p in payloads
              if p.get("doc_timestamp")]
        span: dict = {"first": None, "last": None}
        gap_days = 0
        dated_days: set[int] = set()
        if ts:
            first, last = min(ts), max(ts)
            span = {
                "first": datetime.fromtimestamp(first, tz=timezone.utc)
                .strftime("%Y-%m-%d"),
                "last": datetime.fromtimestamp(last, tz=timezone.utc)
                .strftime("%Y-%m-%d"),
            }
            dated_days = {int(t // 86400) for t in ts}
            total_days = int(last // 86400) - int(first // 86400) + 1
            gap_days = total_days - len(dated_days)
        top_sources = dict(
            sorted(by_source.items(), key=lambda kv: -kv[1])[:25]
        )
        return {
            "corpus": corpus.name,
            "chunks": len(payloads),
            "docs": len(by_source),
            "date_span": span,
            "days_covered": len(dated_days),
            "gap_days": gap_days,
            "sources": top_sources,
            # No silent caps: say when the source table was cut.
            "sources_truncated": len(by_source) > len(top_sources),
        }

    def summarize_corpus(
        self,
        corpus_name: str,
        after: str | None = None,
        before: str | None = None,
        sample: int = 40,
    ) -> dict:
        """Boot-context rollup of a corpus (optionally windowed): speaker
        tally (from chunk enrichment), date span, top sources, and an evenly
        spaced deterministic sample of chunks for an LLM to theme. Sampling
        is explicit in the output — never a silent truncation."""
        corpus = self.registry.get(corpus_name)
        payloads = list(self.bm25.iter_payloads(corpus.name))
        if after or before:
            filters = []
            if after:
                filters.append(f"after:{after}")
            if before:
                filters.append(f"before:{before}")
            payloads = [
                p for p in payloads
                if _passes_filters(_payload_to_result(p, 0.0, 0.0, []), filters)
            ]
        speakers: dict[str, int] = {}
        by_source: dict[str, int] = {}
        ts: list[float] = []
        for p in payloads:
            md = p.get("metadata", {}) or {}
            for sp in md.get("speakers") or []:
                speakers[str(sp)] = speakers.get(str(sp), 0) + 1
            sid = p.get("source_id", "") or ""
            by_source[sid] = by_source.get(sid, 0) + 1
            if p.get("doc_timestamp"):
                ts.append(float(p["doc_timestamp"]))
        span: dict = {"first": None, "last": None}
        if ts:
            span = {
                "first": datetime.fromtimestamp(min(ts), tz=timezone.utc)
                .strftime("%Y-%m-%d"),
                "last": datetime.fromtimestamp(max(ts), tz=timezone.utc)
                .strftime("%Y-%m-%d"),
            }
        n = len(payloads)
        take = min(max(0, sample), n)
        step = max(1, n // take) if take else 1
        sample_payloads = payloads[::step][:take]
        sample_chunks = [
            {
                "chunk_id": _chunk_id(
                    corpus.name, p.get("source_id", ""), int(p.get("chunk_index", 0))
                ),
                "source_id": p.get("source_id", ""),
                "text": (p.get("text", "") or "")[:240],
            }
            for p in sample_payloads
        ]
        return {
            "corpus": corpus.name,
            "chunks": n,
            "date_span": span,
            "top_speakers": sorted(speakers.items(), key=lambda kv: -kv[1])[:15],
            "top_sources": sorted(by_source.items(), key=lambda kv: -kv[1])[:15],
            "sampled": take < n,
            "sample_chunks": sample_chunks,
        }

    def _source_text_for(self, corpus_name: str, source_id: str) -> str:
        """Full document text for one source, from its carrier point.

        O(1): the carrier is chunk_index 0 and _chunk_id is a pure function
        of (corpus, source_id, index), so this is a direct retrieve. A
        filtered lookup would be a full scan — the collections carry no
        payload indexes.

        Falls back to scanning the source's points when the carrier is
        missing, which also covers stores written before v1.1 (where every
        chunk carried the text) and sources whose chunk 0 was pruned.
        """
        collection = _collection_for(corpus_name)
        carrier_id = _chunk_id(corpus_name, source_id, 0)
        payload = self.store.get_by_id(collection, carrier_id)
        if payload is None:
            payload = self.bm25.get_by_id(corpus_name, carrier_id)
        text = (payload or {}).get("source_text", "") or ""
        if text:
            return text
        for record in self.store.source_records(collection, source_id):
            text = (record.get("payload") or {}).get("source_text", "") or ""
            if text:
                return text
        return ""

    def _source_texts_for(
        self, corpus_name: str, source_ids: list[str]
    ) -> dict[str, str]:
        """_source_text_for over many sources in one round trip."""
        if not source_ids:
            return {}
        collection = _collection_for(corpus_name)
        wanted = list(dict.fromkeys(source_ids))
        by_carrier = {_chunk_id(corpus_name, s, 0): s for s in wanted}
        found = self.store.get_many_by_id(collection, list(by_carrier))
        out: dict[str, str] = {}
        for pid, payload in found.items():
            text = (payload or {}).get("source_text", "") or ""
            if text:
                out[by_carrier[pid]] = text
        for sid in wanted:
            if sid not in out:
                out[sid] = self._source_text_for(corpus_name, sid)
        return out

    def _payload_for_result(self, r: SearchResult) -> dict | None:
        """Stored payload for a result's chunk. Carries the precomputed
        context fields; source_text only when the chunk is its carrier."""
        corpus = self.registry.get(r.corpus)
        payload = self.store.get_by_id(_collection_for(corpus.name), r.chunk_id)
        if payload is None:
            payload = self.bm25.get_by_id(corpus.name, r.chunk_id)
        return payload

    def _copy_points(self, src: str, dst: str, expect: int) -> None:
        """Bulk-copy every point from one collection to another.

        `wait=False` on the upserts: waiting for each 256-point batch to be
        flushed measured ~1,900 points/min on a 123k-point corpus, which is
        90 minutes of downtime for a copy qdrant can do in a fraction of that.
        Correctness does not depend on per-batch acknowledgement — it depends
        on the count check below, which the caller then re-verifies before
        anything is dropped.
        """
        from qdrant_client.http import models as qm

        offset: object = None
        while True:
            points, offset = self.store.client.scroll(
                collection_name=src,
                with_payload=True,
                with_vectors=True,
                limit=2048,
                offset=offset,
            )
            if points:
                self.store.client.upsert(
                    collection_name=dst,
                    points=[
                        qm.PointStruct(id=p.id, vector=p.vector, payload=p.payload or {})
                        for p in points
                    ],
                    wait=False,
                )
            if offset is None:
                break

        # Unacknowledged writes are still in flight; settle before counting.
        deadline = time.time() + 900
        while time.time() < deadline:
            if self.store.client.count(dst, exact=True).count >= expect:
                return
            time.sleep(2)
        raise CorpusError(
            f"copy {src!r} -> {dst!r} settled at "
            f"{self.store.client.count(dst, exact=True).count} of {expect} points"
        )

    def migrate_datatype(self, name: str, datatype: str) -> "Corpus":
        """Rebuild a corpus's qdrant collection with a different on-disk vector
        datatype, in place, WITHOUT re-embedding anything.

        Qdrant pins the vector datatype at collection creation, so changing it
        means a new collection. The vectors themselves are already correct --
        float16 is a storage width, not a different embedding -- so this
        scrolls the existing points across rather than re-running the embedder.

        Two passes, because qdrant cannot rename: live -> temp, then back into
        a freshly-created live at the new datatype. The original is only
        dropped after the temp copy is verified point-for-point, so an
        interrupted run loses nothing.

        BM25 is untouched: it holds no vectors.

        Takes the corpus WRITE lock for the whole run. A scroll+copy is not
        atomic, so a concurrent indexer makes the copy come up short and the
        migration abort — which is exactly what happened on the first live
        attempt (2026-08-18): `vecgrep-indexer-chats` runs `vecgrep watch` as
        its own daemon, independent of vecgrep-serve, and added 19 points
        mid-copy. Stopping the server is not enough; the lock is what actually
        excludes the writer.
        """
        if datatype not in ("float32", "float16"):
            raise ValueError(f"unsupported vector datatype {datatype!r}")
        corpus = self.registry.get(name)
        if corpus.datatype == datatype:
            return corpus
        with self.locks.write(name):
            return self._migrate_datatype_locked(name, datatype, corpus)

    def _migrate_datatype_locked(self, name: str, datatype: str, corpus) -> "Corpus":
        live = _collection_for(name)
        temp = _collection_for(f"{name}__dt")
        src_n = self.store.client.count(live, exact=True).count

        # A leftover temp from an interrupted run must not be appended to.
        self.store.drop_collection(temp)
        self.store.ensure_collection(temp, corpus.dim, datatype=datatype)
        self._copy_points(live, temp, src_n)

        # Verify BEFORE dropping the original: it is the only copy.
        dst_n = self.store.client.count(temp, exact=True).count
        if dst_n != src_n:
            self.store.drop_collection(temp)
            raise CorpusError(
                f"datatype migration for {name!r} copied {dst_n} of {src_n} "
                f"points; original left untouched"
            )

        self.store.drop_collection(live)
        self.store.ensure_collection(live, corpus.dim, datatype=datatype)
        self._copy_points(temp, live, src_n)
        final_n = self.store.client.count(live, exact=True).count
        if final_n != src_n:
            raise CorpusError(
                f"datatype migration for {name!r} left {final_n} of {src_n} "
                f"points on the live collection; temp {temp!r} still holds a "
                f"full copy"
            )
        self.store.drop_collection(temp)

        corpus.datatype = datatype
        self.registry.upsert(corpus)
        return corpus

    def _rerank_ready(self, model_name: str | None) -> bool:
        """Is the cross-encoder loaded (or loadable within the wait budget)?

        Kicks off the background warm on the way past, so the first search
        after a cold start pays nothing and the second one usually reranks."""
        from .rerank import DEFAULT_RERANKER, wait_ready

        name = model_name or DEFAULT_RERANKER
        try:
            ready = wait_ready(name)
        except Exception as exc:   # never let warming break a search
            logger.warning("reranker readiness check failed: %s", exc)
            return False
        if not ready:
            logger.info(
                "reranker %s not ready — returning fusion order for this search",
                name,
            )
        return ready

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
        for score, original in scored:
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
        # order -- so `out` already reflects the reranked ordering. MMR selects
        # top_k from it using the rerank score as relevance, skipping
        # near-clones; selection order starts from the best hit, so the
        # reranked ordering survives for everything selected. Do NOT re-sort
        # by similarity_pct here (that would undo the rerank).
        return mmr_select(
            out, top_k, key=lambda r: r.explain.get("rerank_score", 0.0)
        )

    def _search_one(
        self,
        corpus: Corpus,
        query: str,
        top_k: int,
        mode: SearchMode,
        explain: bool = False,
        query_vectors: _QueryVectorMemo | None = None,
    ) -> list[SearchResult]:
        self._recover_if_pending(corpus.name)
        with self.locks.read(corpus.name):
            # A migration/delete may have completed while query assembly was
            # selecting corpora. Re-resolve metadata inside admission.
            corpus = self.registry.get(corpus.name)
            return self._search_one_locked(
                corpus, query, top_k, mode, explain, query_vectors
            )

    def _search_one_locked(
        self,
        corpus: Corpus,
        query: str,
        top_k: int,
        mode: SearchMode,
        explain: bool = False,
        query_vectors: _QueryVectorMemo | None = None,
    ) -> list[SearchResult]:
        collection = _collection_for(corpus.name)

        if mode in ("hybrid", "bm25") and isinstance(self.bm25, BM25SqliteStore):
            expected = self.store.count(collection)
            if expected and (
                not self.bm25.exists(corpus.name)
                or self.bm25.count(corpus.name) != expected
            ):
                raise CorpusError(
                    f"BM25 SQLite index missing or inconsistent for {corpus.name}; "
                    f"run vecgrep bm25 rebuild {corpus.name} before searching"
                )

        vector_hits: list[StoredHit] = []
        bm25_hits: list[tuple[str, float, dict]] = []

        if mode in ("hybrid", "vector"):
            # Use the failover-aware embed so a backend that died mid-session
            # (primary Ollama down) re-resolves to the fallback instead of
            # raising forever on a stale cached backend.
            qv = (
                query_vectors.get(corpus)
                if query_vectors is not None
                else self._embed_query_singleflight(corpus, query)
            )
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
            selected = bm25_hits[:top_k]
            missing = [
                cid for cid, _, payload in selected
                if "context_before" not in payload
            ]
            hydrated = self.store.get_many_by_id(collection, missing)
            for rank, (cid, score, payload) in enumerate(selected):
                if "context_before" not in payload:
                    payload = hydrated.get(cid) or payload
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
            rrf[cid] = rrf.get(cid, 0.0) + _bm25_fusion_weight(corpus) / (RRF_K + rank + 1)
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
        # Cross-corpus rank weight rides the same multiplier: neutral (1.0)
        # within a single corpus, decisive when corpora compete in a merged
        # search (curated shelves above the transcript firehose).
        weight = getattr(corpus, "rank_weight", 1.0) or 1.0
        now = time.time()
        decay_by_id: dict[str, float] = {}
        decayed: dict[str, float] = {}
        for cid, raw in rrf.items():
            ts = payloads_by_id.get(cid, {}).get("doc_timestamp")
            factor = _recency_factor(ts, half_life, now)
            decay_by_id[cid] = factor
            decayed[cid] = raw * factor * weight

        fused = sorted(decayed.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        missing = [
            cid for cid, _ in fused
            if "context_before" not in payloads_by_id[cid]
        ]
        hydrated = self.store.get_many_by_id(collection, missing)
        # For BM25-only display: rescale per-query so the top BM25 hit reads
        # at BM25_DISPLAY_TOP (~90%) and weaker BM25 hits taper toward
        # BM25_DISPLAY_FLOOR. The raw fused RRF score is unchanged for ranking.
        max_bm25 = max(bm25_score_by_id.values()) if bm25_score_by_id else 0.0

        out = []
        for cid, fused_score in fused:
            payload = payloads_by_id[cid]
            # A recovery-built BM25 sidecar omits the derived context fields;
            # Qdrant is canonical, so refetch when they are absent.
            if "context_before" not in payload:
                payload = hydrated.get(cid) or payload
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

    # ----- embed cache housekeeping ------------------------------------------------
    def cache_keep_set(self) -> dict[str, set[str]]:
        """{embed identity: shas of every chunk text a REGISTERED corpus holds}.

        Derived from qdrant, not from the cache, so it is exactly the set a
        re-index of any live corpus would ask the cache for. Identity is the
        backend's `name:model`, which is what CachedBackend keys rows under."""
        keep: dict[str, set[str]] = {}
        for c in self.registry.list():
            ident = f"{c.embed_backend}:{c.embed_model}"
            shas = keep.setdefault(ident, set())
            for _pid, payload in self.store.iter_payloads(
                _collection_for(c.name), include_payload_fields={"text"}
            ):
                text = payload.get("text")
                if text:
                    shas.add(EmbedCache._sha(text))
        return keep

    def cache_sweep(
        self, *, dry_run: bool = False, identities: list[str] | None = None
    ) -> dict:
        """Delete cached vectors no registered corpus references.

        The cache exists to make a re-index cheap; rows for chunks that no
        longer exist in any corpus (deleted sources, retired corpora, a
        chunker change) buy nothing and were making it effectively
        append-only. Returns {kept: {identity: n}, deleted: {identity: n},
        dry_run: bool}."""
        if self._embed_cache is None:
            return {"kept": {}, "deleted": {}, "dry_run": dry_run}
        keep = self.cache_keep_set()
        deleted = self._embed_cache.sweep(keep, identities=identities, dry_run=dry_run)
        return {
            "kept": {k: len(v) for k, v in keep.items()},
            "deleted": deleted,
            "dry_run": dry_run,
        }

    # ----- corpus management ----------------------------------------------------
    def delete_corpus(self, name: str) -> None:
        with self.locks.write(name):
            self._recover_corpus_locked(name)
            corpus = self.registry.get(name)
            self.mutations.write({
                "version": 1,
                "corpus": name,
                "operation": "delete_corpus",
                "phase": "prepared",
                "corpus_before": asdict(corpus),
            })
            self.store.drop_collection(_collection_for(corpus.name))
            record = self.mutations.read(name) or {}
            record["phase"] = "qdrant_done"
            self.mutations.write(record)
            self.bm25.drop(corpus.name)
            self._drop_explorer_catalog(corpus.name)
            record["phase"] = "bm25_done"
            self.mutations.write(record)
            self.registry.delete(name)
            record["phase"] = "registry_done"
            self.mutations.write(record)
            self.mutations.finish(name)

    def set_decay(self, name: str, half_life_days: float | None) -> Corpus:
        """Set (or clear, with None) a corpus's recency-decay half-life in days.

        No re-index needed — decay is applied at search time from the
        per-chunk doc_timestamp already in each payload.
        """
        with self.locks.write(name):
            self._recover_corpus_locked(name)
            corpus = self.registry.get(name)
            if half_life_days is not None and half_life_days <= 0:
                raise CorpusError("half-life must be positive (or omit to disable decay)")
            corpus.decay_half_life_days = half_life_days
            self.registry.upsert(corpus)
            return corpus

    def set_bm25_weight(self, name: str, weight: float | None) -> Corpus:
        """Pin (or with None, unpin) a corpus's BM25 fusion weight."""
        with self.locks.write(name):
            self._recover_corpus_locked(name)
            corpus = self.registry.get(name)
            if weight is not None and weight < 0:
                raise CorpusError("bm25 weight must be >= 0 (or omit to use the default)")
            corpus.bm25_weight = None if weight is None else float(weight)
            self.registry.upsert(corpus)
            return corpus

    def set_rank_weight(self, name: str, weight: float | None) -> Corpus:
        """Set (or reset, with None) a corpus's cross-corpus rank weight."""
        with self.locks.write(name):
            self._recover_corpus_locked(name)
            corpus = self.registry.get(name)
            if weight is not None and weight <= 0:
                raise CorpusError("rank weight must be positive (or omit to reset to 1.0)")
            corpus.rank_weight = 1.0 if weight is None else weight
            self.registry.upsert(corpus)
            return corpus

    def set_corpus_context(
        self,
        name: str,
        description: str = "",
        use_for: list[str] | None = None,
        avoid_for: list[str] | None = None,
    ) -> Corpus:
        """Replace operator-authored routing metadata without re-indexing."""
        description = description.strip()
        use_for = [hint.strip() for hint in (use_for or [])]
        avoid_for = [hint.strip() for hint in (avoid_for or [])]
        if len(description) > 500:
            raise CorpusError("description must be at most 500 characters")
        for label, hints in (("use_for", use_for), ("avoid_for", avoid_for)):
            if len(hints) > 8:
                raise CorpusError(f"{label} accepts at most 8 hints")
            if any(not hint for hint in hints):
                raise CorpusError(f"{label} hints cannot be empty")
            if any(len(hint) > 240 for hint in hints):
                raise CorpusError(f"each {label} hint must be at most 240 characters")
        with self.locks.write(name):
            self._recover_corpus_locked(name)
            corpus = self.registry.get(name)
            corpus.description = description
            corpus.use_for = use_for
            corpus.avoid_for = avoid_for
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
        chunk_start = int(payload.get("chunk_start", 0))
        chunk_end = int(payload.get("chunk_end", 0))
        # An expanded window is explicitly asking to see past the chunk, so
        # resolve the whole document from the carrier rather than clamping to
        # the CONTEXT_CHARS stored alongside this chunk.
        source_text = payload.get("source_text", "") or ""
        if not source_text:
            source_text = self._source_text_for(
                corpus.name, payload.get("source_id", "") or ""
            )
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
        with self.locks.write(corpus_name):
            self._recover_corpus_locked(corpus_name)
            corpus = self.registry.get(corpus_name)
            collection = _collection_for(corpus.name)
            target = Corpus(**asdict(corpus))
            if source_id in target.sources:
                target.sources.remove(source_id)
            target.source_hashes.pop(source_id, None)
            self.mutations.write({
                "version": 1,
                "corpus": corpus_name,
                "operation": "delete_source",
                "phase": "prepared",
                "source_id": source_id,
                "corpus_before": asdict(corpus),
                "corpus_target": asdict(target),
            })
            self.store.delete_by_source(collection, source_id)
            record = self.mutations.read(corpus_name) or {}
            record["phase"] = "qdrant_done"
            self.mutations.write(record)
            self.bm25.delete_by_source(corpus.name, source_id)
            catalog_sync_ok = True
            try:
                self.explorer_store.delete_source(corpus.name, source_id)
            except Exception as exc:
                catalog_sync_ok = False
                logger.warning(
                    "explorer catalog delete failed for %s: %s",
                    corpus_name,
                    exc,
                )
            record["phase"] = "bm25_done"
            self.mutations.write(record)
            target.doc_count = len(target.sources)
            target.chunk_count = self.store.count(collection)
            target.updated_at = time.time()
            if self.ephemeral:
                self.registry._corpora[target.name] = target
            else:
                self.registry.upsert(target)
            if catalog_sync_ok:
                try:
                    self.explorer_store.set_generation(
                        target.name, self._explorer_generation(target)
                    )
                except Exception as exc:
                    logger.warning(
                        "explorer catalog commit failed for %s: %s",
                        corpus_name,
                        exc,
                    )
            record["phase"] = "registry_done"
            self.mutations.write(record)
            self.mutations.finish(corpus_name)

    def list_corpora(self) -> list[Corpus]:
        return self.registry.list()

    def _record_recovery_progress(
        self,
        corpus: Corpus,
        *,
        source_done: int,
        sources_total: int,
        chunks_done: int,
        active: bool,
    ) -> None:
        """Atomically publish coarse recovery progress for an off-box monitor."""
        path = self.settings.home / "recovery-progress.json"
        payload = {
            "corpus": corpus.name,
            "source_done": source_done,
            "sources_total": sources_total,
            "chunks_done": chunks_done,
            "last_progress_at": time.time(),
            "active": active,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, path)

    def diagnose(self, *, corpora: set[str] | None = None) -> list[dict]:
        """Reconcile the registry against the vector store — the check that
        catches a corpus a Qdrant flap wiped, a count that drifted, or an
        orphan collection with no registry entry. One dict per issue found:

          {corpus, kind, detail, fixable}

        kind is one of:
          - "missing_collection": registry has the corpus but Qdrant has no
            points (0 or a 404'd collection). The classic post-flap data loss.
            Fixable iff every recorded source still exists on disk.
          - "count_drift": registry chunk_count != live Qdrant point count.
            Always fixable (recount, or re-index if sources exist).
          - "orphan_collection": a vecgrep__* collection with points but no
            registry entry — searchable via Qdrant but invisible to the app.
          - "orphaned_source": a registered source whose file no longer exists.
            Its chunks are still indexed, so a deleted document keeps being
            returned as a live answer. Always fixable — the fix is to purge it
            from both backends. Carries an extra "source_id" key.
          - "embed_model_split": corpora on the same backend disagree about
            which embedding model to use. Not corruption — everything still
            answers — which is exactly why it hides. Ollama treats each model
            name as a separate model and only keeps OLLAMA_MAX_LOADED_MODELS
            resident, so any search crossing the split evicts and reloads.
            Measured on a live install 2026-08-18: 900 loads in 6 hours and
            embeds at 13-28s against a 0.13s warm baseline, presenting as
            "retrieval feels slow" for who knows how long. NOT auto-fixable:
            repointing a corpus is only safe when the two models emit the same
            vectors, which is a human call.

        Read-only. `reconcile()` acts on what this reports."""
        issues: list[dict] = []
        selected = set(corpora or ())
        registered = {
            c.name: c
            for c in self.registry.list()
            if not selected or c.name in selected
        }

        # Majority is taken over EVERY corpus, not just the selected ones, so
        # `--corpora x` still compares x against the install rather than
        # against itself.
        by_backend: dict[str, list] = {}
        for c in self.registry.list():
            by_backend.setdefault(c.embed_backend, []).append(c)
        majority: dict[str, str] = {}
        for backend, group in by_backend.items():
            models = [c.embed_model for c in group]
            if len(set(models)) > 1:
                majority[backend] = max(set(models), key=models.count)

        for name, c in registered.items():
            want = majority.get(c.embed_backend)
            if want is not None and c.embed_model != want:
                issues.append({
                    "corpus": name,
                    "kind": "embed_model_split",
                    "detail": f"embed_model {c.embed_model!r} differs from the "
                              f"{c.embed_backend!r} majority {want!r}; every "
                              f"search crossing the two reloads the model",
                    "fixable": False,
                })
            live = self.store.count(_collection_for(name))
            if live == 0 and c.chunk_count > 0:
                srcs = list(c.sources or [])
                have = [s for s in srcs if _source_exists(s)]
                issues.append({
                    "corpus": name,
                    "kind": "missing_collection",
                    "detail": f"registry has {c.chunk_count} chunks, vector store has 0"
                              + (f"; {len(have)}/{len(srcs)} sources on disk" if srcs else "; no source paths recorded"),
                    "fixable": bool(srcs) and len(have) == len(srcs),
                })
            elif live != c.chunk_count:
                issues.append({
                    "corpus": name,
                    "kind": "count_drift",
                    "detail": f"registry {c.chunk_count} vs vector store {live}",
                    "fixable": True,
                })
            # A source the indexer recorded, whose file has since been
            # deleted. Nothing in the write path notices this: an ingest
            # pipeline unlinks the file and vecgrep is never told, so the
            # chunks outlive the document they came from. Observed in a live
            # deployment: a deleted document was still the top hit at 94.4%
            # for a question it should no longer have answered, and 198 such
            # points across 24 dead files had to be removed by hand.
            # Reported per source so a purge names exactly what it removes.
            for src_id in (c.sources or []):
                if not _source_exists(src_id):
                    issues.append({
                        "corpus": name,
                        "kind": "orphaned_source",
                        "source_id": src_id,
                        "detail": f"source no longer exists: {src_id}",
                        "fixable": True,
                    })

            if c.chunk_count > 0:
                if not self.bm25.exists(name):
                    issues.append({
                        "corpus": name,
                        "kind": "missing_bm25",
                        "detail": f"registry has {c.chunk_count} chunks but the BM25 index is absent",
                        "fixable": live > 0,
                    })
                elif live > 0:
                    bm25_count = self.bm25.count(name)
                    if bm25_count != live:
                        issues.append({
                            "corpus": name,
                            "kind": "bm25_count_drift",
                            "detail": f"vector store {live} vs BM25 {bm25_count}",
                            "fixable": True,
                        })

        # Orphan collections: in Qdrant under our prefix, absent from registry.
        for coll in self.store.list_collections():
            cname = _corpus_from_collection(coll)
            if (
                cname
                and (not selected or cname in selected)
                and cname not in registered
                and self.store.count(coll) > 0
            ):
                issues.append({
                    "corpus": cname,
                    "kind": "orphan_collection",
                    "detail": f"{self.store.count(coll)} points in vector store, no registry entry",
                    "fixable": False,  # needs the source to rebuild the registry row
                })
        return issues

    def reconcile(
        self,
        *,
        reindex: bool = False,
        corpora: set[str] | None = None,
    ) -> list[dict]:
        """Repair what diagnose() finds. Returns the actions taken (one dict per
        issue: {corpus, kind, action}).

        - count_drift → recount chunk_count from Qdrant (cheap, always safe).
        - missing_collection → only re-index from recorded sources when
          `reindex=True` AND every source still exists (embeds — not free).
        Otherwise reported as "needs_reindex" so a human decides.
        - orphan_collection → left alone (rebuilding the registry row needs the
          original source; reported for a human to `vecgrep index` it back)."""
        def rebuild_qdrant_only(corpus: Corpus) -> None:
            resume_source_counts = self.store.source_counts(_collection_for(corpus.name))
            sources = list(corpus.sources or [])
            self._record_recovery_progress(
                corpus,
                source_done=0,
                sources_total=len(sources),
                chunks_done=self.store.count(_collection_for(corpus.name)),
                active=True,
            )
            for source_done, src in enumerate(sources, start=1):
                self.index(
                    src,
                    corpus.name,
                    force=True,
                    bypass_embed_cache=False,
                    update_bm25=False,
                    update_registry=False,
                    resume_source_counts=resume_source_counts,
                )
                # Every source is a durable resume point. Avoid thousands of
                # point-count calls during a normal run while still updating a
                # monitor much more often than its ten-minute stale threshold.
                if source_done % 16 == 0 or source_done == len(sources):
                    self._record_recovery_progress(
                        corpus,
                        source_done=source_done,
                        sources_total=len(sources),
                        chunks_done=self.store.count(_collection_for(corpus.name)),
                        active=True,
                    )
            # Commit registry health only after every source has completed.
            corpus.chunk_count = self.store.count(_collection_for(corpus.name))
            corpus.doc_count = len(corpus.sources)
            corpus.updated_at = time.time()
            self.registry.upsert(corpus)
            self._record_recovery_progress(
                corpus,
                source_done=len(sources),
                sources_total=len(sources),
                chunks_done=corpus.chunk_count,
                active=False,
            )

        rebuild_bm25_only = self._rebuild_bm25_from_store

        actions: list[dict] = []
        for issue in self.diagnose(corpora=corpora):
            name, kind = issue["corpus"], issue["kind"]
            if kind == "count_drift":
                c = self.registry.get(name)
                sources = list(c.sources or [])
                all_sources_exist = bool(sources) and all(_source_exists(src) for src in sources)
                live = self.store.count(_collection_for(name))
                if reindex and live < c.chunk_count and all_sources_exist:
                    rebuild_qdrant_only(c)
                    action = "reindexed"
                else:
                    c.chunk_count = live
                    c.updated_at = time.time()
                    self.registry.upsert(c)
                    action = "recounted"
                actions.append({"corpus": name, "kind": kind, "action": action})
            elif kind == "missing_collection":
                if reindex and issue["fixable"]:
                    c = self.registry.get(name)
                    rebuild_qdrant_only(c)
                    actions.append({"corpus": name, "kind": kind, "action": "reindexed"})
                else:
                    actions.append({"corpus": name, "kind": kind, "action": "needs_reindex"})
            elif kind in {"missing_bm25", "bm25_count_drift"}:
                if kind == "bm25_count_drift":
                    live = self.store.count(_collection_for(name))
                    if self.bm25.count(name) == live:
                        # An earlier count-drift repair in this same pass may
                        # already have restored Qdrant to the sidecar's count.
                        continue
                if issue["fixable"]:
                    rebuild_bm25_only(self.registry.get(name))
                    actions.append({"corpus": name, "kind": kind, "action": "rebuilt_bm25"})
                else:
                    actions.append({"corpus": name, "kind": kind, "action": "needs_reindex"})
            elif kind == "orphaned_source":
                # delete_source drops Qdrant points AND BM25 entries under one
                # write lock with a mutation-journal record, so an interrupted
                # purge resumes instead of leaving the half-purged state —
                # which is the worst one: vectors gone, BM25 still serving the
                # dead document at a plausible mid rank that reads as real.
                self.delete_source(name, issue["source_id"])
                actions.append({
                    "corpus": name,
                    "kind": kind,
                    "source_id": issue["source_id"],
                    "action": "purged",
                })
            else:  # orphan_collection
                actions.append({"corpus": name, "kind": kind, "action": "needs_manual_index"})
        return actions

    def calibration(self, corpus_name: str | None) -> dict:
        """The display calibration the UI must mirror to reproduce a search's
        similarity_pct for THIS corpus's embed model.

        The web UI re-derives the displayed % client-side (so a tuning slider
        updates without a round-trip). It used to seed those sliders from a
        HARDCODED calibration (nomic's 0.66/12) regardless of the corpus's actual
        model, so on a bge-m3 corpus (the default) the panel's % drifted from the
        server's until the user matched the sliders by hand. Emitting the real
        per-model calibration here lets the client seed the panel correctly for
        whatever model the corpus is pinned to.
        """
        model = None
        if corpus_name:
            try:
                model = self.registry.get(corpus_name).embed_model
            except CorpusError:
                model = None
        center, slope = _calibration_for(model)
        return {
            "cosine_center": center,
            "cosine_slope": slope,
            "bm25_top": BM25_DISPLAY_TOP,
            "bm25_floor": BM25_DISPLAY_FLOOR,
        }

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
        meta_values: dict[str, set] = {}
        has_timestamp = False
        # One pass: the old form scanned the payload list twice, which is free
        # when it is already in memory and is not when it is a query.
        for payload in self.bm25.iter_payloads(corpus.name):
            if payload.get("doc_timestamp") is not None:
                has_timestamp = True
            meta = payload.get("metadata") or {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    meta_values.setdefault(k, set())
                    if len(meta_values[k]) < max_values:
                        meta_values[k].add(v)
        out: dict = {
            "corpus": corpus.name,
            "filters": {
                "source": {"form": "source:GLOB", "description": "fnmatch on source_id"},
                "source_path": {
                    "form": "source_path:GLOB",
                    "description": "alias of source: (fnmatch on source_id)",
                },
                "corpus": {"form": "corpus:NAME", "description": "exact corpus match"},
                "channel": {
                    "form": "channel:NAME",
                    "description": "metadata channel match (quote-tolerant)",
                },
                "meta": {
                    "form": "meta.KEY=VALUE",
                    "keys": {
                        k: sorted(str(x) for x in vals) for k, vals in sorted(meta_values.items())
                    },
                },
                "speaker": {
                    "form": "speaker:NAME (alias author:NAME)",
                    "description": "chunk contains a transcript line by NAME "
                                   "(case-insensitive, ' [bot]' suffix optional; "
                                   "chunk-level, not per-line attribution)",
                },
                "bot": {
                    "form": "bot:true|false",
                    "description": "chunk does/doesn't contain a bot speaker "
                                   "(archiver '[bot]' marker)",
                },
                "has": {
                    "form": "has:code|table|link",
                    "description": "chunk contains a fenced code block / "
                                   "markdown table / URL",
                },
                "negation": {
                    "form": "-<any filter>",
                    "description": "leading '-' inverts: -corpus:scratch, "
                                   "-channel:cl-6, -speaker:NAME …",
                },
            },
            "has_doc_timestamp": has_timestamp,
        }
        if has_timestamp:
            # Time filters only make sense on corpora whose chunks carry dates.
            out["filters"]["date"] = {
                "form": "date:YYYY-MM-DD | date:today | date:yesterday",
                "description": "doc_timestamp inside that UTC day (hard constraint)",
            }
            out["filters"]["after"] = {
                "form": "after:ISO | after:7d|24h|30m|2w",
                "description": "doc_timestamp >= ISO date/datetime, or a "
                               "relative window back from now",
            }
            out["filters"]["before"] = {
                "form": "before:ISO | before:today|7d…",
                "description": "doc_timestamp < ISO date/datetime or relative",
            }
        return out

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
        self.store.ensure_collection(
            new_collection, new_corpus.dim, datatype=new_corpus.datatype,
        )
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

        # Rebuild from the final canonical payloads through the selected store.
        # Renaming a .pkl stranded SQLite corpora after embedding migration.
        self._rebuild_bm25_from_store(new_corpus)
        self.bm25.drop(temp_name)
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
            old_collection = _collection_for(meta["name"])
            new_collection = _collection_for(target_name)
            if src_qdrant.is_dir():
                src_collection_dir = src_qdrant / old_collection
                if src_collection_dir.is_dir():
                    dest_collection_dir = (
                        self.settings.qdrant_path / "collection" / new_collection
                    )
                    dest_collection_dir.parent.mkdir(parents=True, exist_ok=True)
                    _copytree(src_collection_dir, dest_collection_dir)

            # Register the restored collection in Qdrant's meta.json. Embedded
            # QdrantLocal builds get_collections() PURELY from meta.json at open
            # time — so copying the collection DIRECTORY isn't enough; without its
            # meta.json entry the collection is invisible and every search against
            # the restored corpus returns nothing (silent data-loss on round-trip;
            # Jeff 2026-07-06). The export ships the source meta.json, so we read
            # the old collection's config out of it, re-key it to the (possibly
            # renamed) target, and merge it into the live meta.json.
            src_meta = staging / "qdrant" / "meta.json"
            dst_meta = self.settings.qdrant_path / "meta.json"
            if src_meta.is_file():
                src_collections = json.loads(src_meta.read_text()).get("collections", {})
                entry = src_collections.get(old_collection)
                if entry is not None:
                    if dst_meta.is_file():
                        dst = json.loads(dst_meta.read_text())
                    else:
                        dst = {"collections": {}, "aliases": {}}
                    dst.setdefault("collections", {})[new_collection] = entry
                    dst_meta.parent.mkdir(parents=True, exist_ok=True)
                    dst_meta.write_text(json.dumps(dst))

            # SECURITY: do NOT import the tarball's bm25.pkl — loading an
            # attacker-supplied pickle is arbitrary code execution. The BM25
            # index is rebuilt from the imported documents on the next index
            # (bm25_store load is a no-op when the .pkl is absent), so skipping
            # the untrusted pickle is safe and closes the RCE vector.
            pass

            # Corpus dataclass defaults make old archives compatible; using
            # the complete metadata mapping means new routing/weight/storage
            # fields survive export/import without one-off copy code.
            meta["name"] = target_name
            meta["updated_at"] = time.time()
            corpus = Corpus(**meta)
            self.registry.upsert(corpus)

        # Re-open store so the new collection is visible.
        self.store = QdrantStore(
            None if self.ephemeral else self.settings.qdrant_path,
            url=None if self.ephemeral else self.settings.qdrant_url,
        )

        # Rebuild the BM25 index from the restored points. We deliberately did NOT
        # import the tarball's bm25.pkl (loading an attacker-supplied pickle is
        # RCE), and the "rebuilt on next index" the old comment promised never
        # happens on a plain round-trip — so without this, a restored corpus has
        # NO keyword index, hybrid/bm25 search returns nothing, and a vector-only
        # hit below the model's cosine floor is silently dropped (the round-trip
        # data-loss bug; Jeff 2026-07-06). Scroll the collection's payloads and
        # re-upsert them into BM25 under the target name — text + payload only, no
        # pickle, no re-embedding.
        try:
            self.bm25.drop(target_name)
            offset = None
            ids: list[str] = []
            texts: list[str] = []
            payloads: list[dict] = []
            while True:
                points, offset = self.store.client.scroll(
                    collection_name=new_collection, limit=256,
                    offset=offset, with_payload=True, with_vectors=False,
                )
                for pt in points:
                    pl = pt.payload or {}
                    ids.append(str(pt.id))
                    texts.append(pl.get("text", ""))
                    payloads.append(pl)
                if offset is None:
                    break
            if ids:
                self.bm25.upsert(target_name, ids, texts, payloads)
        except Exception:
            # BM25 rebuild is best-effort — a restored corpus still works on the
            # vector half; the next index of any doc rebuilds BM25 anyway.
            pass

        return corpus


# ----- helpers ------------------------------------------------------------------
_COLLECTION_PREFIX = "vecgrep__"


def _collection_for(corpus_name: str) -> str:
    return f"{_COLLECTION_PREFIX}{corpus_name}"


def _corpus_from_collection(collection: str) -> str | None:
    """Inverse of _collection_for. None for a collection we don't own."""
    if collection.startswith(_COLLECTION_PREFIX):
        return collection[len(_COLLECTION_PREFIX):]
    return None


def _source_exists(source_id: str) -> bool:
    """A recorded source is on disk (so a wiped corpus can be rebuilt from it).
    URLs are treated as present — we can't cheaply verify them, and re-indexing
    a URL just re-fetches it."""
    if source_id.startswith(("http://", "https://")):
        return True
    try:
        return Path(source_id).exists()
    except OSError:
        return False


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
        "source_length": hit.source_length,
        "chunk_index": hit.chunk_index,
        "chunk_start": hit.chunk_start,
        "chunk_end": hit.chunk_end,
        "text": hit.chunk_text,
        "context_before": hit.context_before,
        "context_after": hit.context_after,
        "line_start": hit.line_start,
        "line_end": hit.line_end,
        "metadata": hit.metadata,
        "doc_timestamp": hit.doc_timestamp,
    }


def _corpus_to_dict(c: Corpus) -> dict:
    return asdict(c)


def _copytree(src: Path, dst: Path) -> None:
    import shutil

    shutil.copytree(src, dst, dirs_exist_ok=True)


# Relative time sugar: after:7d, before:24h, after:2w, date:today/yesterday.
# Resolved against an injectable `now` so tests stay deterministic.
_RELATIVE_TIME_RE = re.compile(r"^(\d+)([mhdw])$")
_RELATIVE_UNIT_S = {"m": 60.0, "h": 3600.0, "d": 86400.0, "w": 7 * 86400.0}


def _parse_filter_time(value: str, now: float | None = None) -> float | None:
    """ISO date/datetime OR a relative form → epoch seconds. None = unparseable.

    Relative forms: `<N>[m|h|d|w]` (that long ago, from `now`), `today` (start
    of the current UTC day), `yesterday` (start of the previous UTC day). ISO
    input behaves exactly as before (UTC when naive)."""
    value = value.strip()
    low = value.lower()
    if low in ("today", "yesterday"):
        ref = time.time() if now is None else now
        day_start = (
            datetime.fromtimestamp(ref, tz=timezone.utc)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .timestamp()
        )
        return day_start if low == "today" else day_start - _SECONDS_IN_DAY
    rel = _RELATIVE_TIME_RE.match(low)
    if rel:
        ref = time.time() if now is None else now
        return ref - int(rel.group(1)) * _RELATIVE_UNIT_S[rel.group(2)]
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


_SECONDS_IN_DAY = 86400.0


# Sentinel: a RECOGNIZED filter form whose value is unparseable (typo'd date).
# Fails closed in EITHER polarity — `-after:notadate` must not invert into
# match-everything.
_INVALID = object()


def _filter_matches(
    result: SearchResult, f: str, now: float | None = None
):
    """One filter's verdict for one result. Returns True/False (matched / not),
    None (unknown form — ignored for back-compat), or _INVALID (recognized
    form, unparseable value — the caller fails closed)."""
    if f.startswith("source:") or f.startswith("source_path:"):
        pat = f.split(":", 1)[1]
        return fnmatch.fnmatch(result.source_id, pat)
    if f.startswith("corpus:"):
        return result.corpus == f[len("corpus:"):]
    if f.startswith("date:"):
        day_start = _parse_filter_time(f[len("date:"):], now=now)
        if day_start is None:
            return _INVALID
        ts = result.doc_timestamp
        return ts is not None and day_start <= ts < day_start + _SECONDS_IN_DAY
    if f.startswith("after:"):
        cut = _parse_filter_time(f[len("after:"):], now=now)
        if cut is None:
            return _INVALID
        ts = result.doc_timestamp
        return ts is not None and ts >= cut
    if f.startswith("before:"):
        cut = _parse_filter_time(f[len("before:"):], now=now)
        if cut is None:
            return _INVALID
        ts = result.doc_timestamp
        return ts is not None and ts < cut
    if f.startswith("channel:"):
        want = f[len("channel:"):].strip()
        have = str(result.metadata.get("channel", "")).strip().strip("\"'")
        return have == want
    if f.startswith("speaker:") or f.startswith("author:"):
        # Chunk-level membership: a chunk containing ANY line by NAME passes.
        # Case-insensitive; the archiver's " [bot]" suffix is stripped on both
        # sides so `speaker:analysis_bot` matches "analysis_bot [bot]".
        # Chunks without enrichment (pre-reindex) fail closed — hard filter.
        want = f.split(":", 1)[1].strip().lower().removesuffix(" [bot]").strip()
        if not want:
            return _INVALID
        speakers = result.metadata.get("speakers")
        if not isinstance(speakers, list):
            return False
        return any(
            str(s).strip().lower().removesuffix(" [bot]").strip() == want
            for s in speakers
        )
    if f.startswith("bot:"):
        val = f[len("bot:"):].strip().lower()
        if val not in ("true", "false", "1", "0", "yes", "no"):
            return _INVALID
        want_bot = val in ("true", "1", "yes")
        return bool(result.metadata.get("has_bot_speaker")) is want_bot
    if f.startswith("has:"):
        shape = f[len("has:"):].strip().lower()
        if shape not in ("code", "table", "link"):
            return _INVALID
        return bool(result.metadata.get(f"has_{shape}"))
    if f.startswith("meta."):
        key_value = f[len("meta."):]
        if "=" not in key_value:
            return None
        key, value = key_value.split("=", 1)
        return str(result.metadata.get(key, "")) == value
    return None


def _passes_filters(
    result: SearchResult, filters: list[str], default_active: bool = False,
    now: float | None = None,
) -> bool:
    """Apply --filter expressions. Supported forms:

        source:GLOB       — fnmatch against result.source_id
        source_path:GLOB  — alias of source: (the assistant-facing name)
        corpus:NAME       — exact corpus name match
        meta.KEY=VALUE    — exact metadata field match (string compare)
        date:YYYY-MM-DD   — doc_timestamp inside that UTC day
        after:<t>         — doc_timestamp >= <t>
        before:<t>        — doc_timestamp <  <t>
        channel:NAME      — metadata 'channel' match (quote-tolerant, since
                            archiver frontmatter is `channel: "name"`)

    Time values <t> accept ISO dates/datetimes AND relative forms: `7d`,
    `24h`, `30m`, `2w`, `today`, `yesterday` (resolved against `now`, which
    is injectable for tests and defaults to the wall clock).

    NEGATION: a leading `-` on any recognized filter inverts it —
    `-corpus:scratch` excludes the scratch corpus, `-channel:cl-6` excludes a
    channel, `-source:GLOB` excludes matching sources.

    All filters AND together. Unknown filter forms are silently ignored
    (back-compat), but recognized forms with unparseable values are HARD
    fail-closed in either polarity: a typo'd date is visible as zero results
    instead of silently leaking out-of-window evidence back in. A chunk with
    no doc_timestamp fails every positive time filter.

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
        negated = f.startswith("-")
        form = f[1:] if negated else f
        verdict = _filter_matches(result, form, now=now)
        if verdict is None:
            continue  # unknown form — ignored (back-compat)
        if verdict is _INVALID:
            return False  # typo'd value: zero results, either polarity
        if negated:
            verdict = not verdict
        if not verdict:
            return False
    return True


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
# genuine-but-weak hits ~0.57, strong hits 0.66-0.75. Center 0.57, slope 18 maps:
#   0.51 → ~25%   (noise)
#   0.57 → 50%    (uncertain boundary)
#   0.66 → ~84%   (strong)
#   0.75 → ~96%   (very strong)
# Slope 18 (was 35): the steeper sigmoid saturated everything strong into a
# 99.x cluster, so 99.6-vs-99.2 carried no information. The flatter curve
# keeps the anchors (noise low, boundary 50%) while spreading the strong
# range across ~84-96 — differences are visible again. Ranking is unaffected
# (monotonic remap; order comes from the rerank score itself).
RERANK_CALIBRATION_CENTER = 0.57
RERANK_CALIBRATION_SLOPE = 18.0


def _rerank_to_pct(prob: float) -> float:
    """Sigmoid-calibrated cross-encoder score (0..1) → display percentage."""
    x = RERANK_CALIBRATION_SLOPE * (prob - RERANK_CALIBRATION_CENTER)
    if x > 60:
        return 100.0
    if x < -60:
        return 0.0
    return 100.0 / (1.0 + math.exp(-x))


# Near-duplicate collapse + MMR selection live in assembly.py (Phase 1 of the
# memory-v1 release). `_dedup_overlapping` remains as the historical name:
# same span-overlap semantics as v0.7.0, now also collapsing same-source text
# clones at distant spans (repeated messages / bot alert spam).
_dedup_overlapping = dedup_near_duplicates


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


# How much text on each side of a chunk is stored with the chunk itself.
# Sized to what the search path renders per hit; anything wider resolves
# against the carrier point instead (see VecgrepService._source_text_for).
CONTEXT_CHARS = 400


def _line_anchors(text: str, start: int, end: int) -> tuple[int | None, int | None]:
    """1-based inclusive line span of text[start:end].

    chunk_end is exclusive, so the end line is counted at the last char
    INSIDE the span (end-1) — otherwise a chunk ending exactly at a newline
    would claim the following line too.
    """
    if not text or end <= start:
        return None, None
    return (
        text.count("\n", 0, start) + 1,
        text.count("\n", 0, max(start, end - 1)) + 1,
    )


def _chunk_context_fields(source_text: str, start: int, end: int) -> dict:
    """Per-chunk payload fields derived from the full document at write time.

    Storing these (~800 bytes) beats storing the whole document on every
    chunk (~97 KB on the chats corpus) and keeps the search path free of
    per-hit store lookups.
    """
    line_start, line_end = _line_anchors(source_text, start, end)
    return {
        "source_length": len(source_text),
        "context_before": source_text[max(0, start - CONTEXT_CHARS):start],
        "context_after": source_text[end:end + CONTEXT_CHARS],
        "line_start": line_start,
        "line_end": line_end,
    }


def _payload_to_result(
    payload: dict,
    score: float,
    pct: float,
    matched_by: list[str],
) -> SearchResult:
    chunk_start = int(payload.get("chunk_start", 0))
    chunk_end = int(payload.get("chunk_end", 0))
    corpus_name = payload.get("corpus", "") or ""
    source_id = payload.get("source_id", "") or ""
    chunk_index = int(payload.get("chunk_index", 0))
    cid = _chunk_id(corpus_name, source_id, chunk_index) if corpus_name and source_id else ""
    doc_ts = payload.get("doc_timestamp")
    # Context and line anchors are precomputed at write time (see
    # _chunk_context_fields) so the hot search path never needs the full
    # document. Points written before v1.1 still carry source_text on every
    # chunk; derive from it so a pre-migration store keeps rendering.
    before = payload.get("context_before")
    after = payload.get("context_after")
    line_start = payload.get("line_start")
    line_end = payload.get("line_end")
    if before is None or after is None or line_start is None:
        legacy = payload.get("source_text", "") or ""
        if before is None:
            before = legacy[max(0, chunk_start - CONTEXT_CHARS):chunk_start]
        if after is None:
            after = legacy[chunk_end:chunk_end + CONTEXT_CHARS]
        if line_start is None and legacy:
            line_start, line_end = _line_anchors(legacy, chunk_start, chunk_end)
    before = before or ""
    after = after or ""
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
        doc_timestamp=float(doc_ts) if doc_ts is not None else None,
        line_start=line_start,
        line_end=line_end,
    )
