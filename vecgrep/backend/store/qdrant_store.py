"""Qdrant wrapper.

Embedded mode (path=...) — no Docker, no server. Each named corpus maps
to its own Qdrant collection. Payload schema:

    {
      "corpus": str,
      "source_id": str,
      "source_text": str,        # the FULL source document — stored only on
                                  #   the chunk_index==0 point of each source
                                  #   ("the carrier"). Absent on every other
                                  #   chunk. Whole-document readers reach it
                                  #   in O(1) by recomputing the carrier's
                                  #   deterministic uuid5 id; see
                                  #   service._source_text_for.
      "source_length": int,       # len(source_text), on EVERY chunk, so a
                                  #   reader can bound a window without
                                  #   fetching the carrier.
      "chunk_index": int,
      "chunk_start": int,
      "chunk_end": int,
      "text": str,                # the chunk itself
      "context_before": str,      # <=CONTEXT_CHARS before chunk_start, and
      "context_after": str,       #   <=CONTEXT_CHARS after chunk_end. The
                                  #   search path renders these on every hit;
                                  #   precomputing keeps it lookup-free.
      "line_start": int | None,   # 1-based inclusive line span of the chunk,
      "line_end": int | None,     #   computed once at write time instead of
                                  #   rescanning the document per hit.
      "metadata": {...},
    }

Until v1.1 source_text was duplicated byte-identically onto every chunk so
any hit could rebuild context without a second lookup. That cost ~8.3 GB
across a 186k-point install (91%+ redundant, growing as S^2/chunk_size in
document size). The carrier layout keeps every reader exact for ~40x less.
"""
from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


class SearchBackendError(RuntimeError):
    """Search storage is unavailable; callers must not report a zero-hit success."""


_PAYLOAD_INDEX_FIELDS: tuple[tuple[str, qm.PayloadSchemaType], ...] = (
    ("source_id", qm.PayloadSchemaType.KEYWORD),
    ("doc_timestamp", qm.PayloadSchemaType.FLOAT),
    ("metadata.channel", qm.PayloadSchemaType.KEYWORD),
    ("metadata.status", qm.PayloadSchemaType.KEYWORD),
    ("metadata.speakers", qm.PayloadSchemaType.KEYWORD),
    ("metadata.has_bot_speaker", qm.PayloadSchemaType.BOOL),
    ("metadata.has_code", qm.PayloadSchemaType.BOOL),
    ("metadata.has_table", qm.PayloadSchemaType.BOOL),
    ("metadata.has_link", qm.PayloadSchemaType.BOOL),
)


@dataclass
class StoredHit:
    score: float
    chunk_text: str
    source_id: str
    # Empty on every chunk but the carrier (chunk_index 0). Rendering context
    # comes from context_before/context_after, not from this.
    source_text: str
    chunk_start: int
    chunk_end: int
    chunk_index: int
    metadata: dict
    corpus: str
    # None means "this point predates the derived fields", which is NOT the
    # same as "" -- an empty context_before is correct for a document's
    # first chunk. Readers key their pre-v1.1 fallback on None, so coercing
    # absence to "" silently strips context from every legacy hit.
    context_before: str | None = None
    context_after: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    source_length: int | None = None
    # Document's own date (epoch seconds) for recency decay; None if undated
    # or indexed before doc_timestamp existed.
    doc_timestamp: float | None = None


class QdrantStore:
    def __init__(self, path: Path | None, url: str | None = None) -> None:
        # url=str   -> Qdrant server mode (multi-client safe)
        # path=None -> in-memory (ephemeral)
        # path=Path -> on-disk embedded (single-process lock)
        if url:
            # Generous timeout: wait=True upserts during a large reindex can
            # block while Qdrant flushes/indexes a batch. The client default is
            # too short and surfaced as ResponseHandlingException('timed out')
            # partway through indexing big corpora (e.g. chat transcripts).
            #
            # limits: qdrant-client ships max_keepalive_connections=0, i.e.
            # connection pooling OFF — every request opens a fresh TCP
            # connection and leaves it in TIME_WAIT for 60s. At the request
            # rates indexing and migration produce that exhausts the host's
            # ephemeral port range and the next connect() is refused with
            # ECONNRESET. Re-enabling keepalive takes 300 requests from ~400
            # sockets to 2.
            self.client = QdrantClient(
                url=url,
                timeout=120,
                limits=httpx.Limits(
                    max_connections=32,
                    max_keepalive_connections=16,
                    keepalive_expiry=120.0,
                ),
            )
        elif path is None:
            self.client = QdrantClient(":memory:")
        else:
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
        self._meta_lock = threading.RLock()
        self._known_collections: set[str] | None = None
        self._indexed_collections: set[str] = set()

    def _lock_for_meta(self) -> threading.RLock:
        lock = getattr(self, "_meta_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._meta_lock = lock
        return lock

    def _has_collection(self, name: str) -> bool:
        with self._lock_for_meta():
            known = getattr(self, "_known_collections", None)
            if known is not None and name in known:
                return True
            refreshed = {c.name for c in self.client.get_collections().collections}
            self._known_collections = refreshed
            return name in refreshed

    def _ensure_payload_indexes(self, name: str) -> None:
        if not getattr(self, "_url", None):
            return
        with self._lock_for_meta():
            indexed = getattr(self, "_indexed_collections", None)
            if indexed is None:
                indexed = set()
                self._indexed_collections = indexed
            if name in indexed:
                return
            indexed.add(name)
        for field_name, field_schema in _PAYLOAD_INDEX_FIELDS:
            try:
                self.client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=field_schema,
                    wait=False,
                )
            except Exception:  # noqa: BLE001, S110 -- The storage adapter retains its existing missing-item and optional-index error contract.
                # Embedded local mode or older server builds may ignore or
                # reject index creation on specific field types; search still
                # works with unindexed evaluation.
                pass

    def ensure_collection(
        self,
        name: str,
        dim: int,
        *,
        datatype: str = "float32",
        quantization: str | None = None,
    ) -> None:
        """Create the collection if missing.

        `datatype` is the ON-DISK vector storage type: "float32" (default) or
        "float16", which halves the dominant term of a large collection's
        footprint. `quantization` ("int8" | "binary") adds a compressed copy
        used for the search pass with rescoring against the originals -- it
        buys RAM and latency, NOT disk (qdrant keeps the originals), so it is
        not the lever for storage. Both are pinned at creation; measure any
        change on the eval harness before adopting it for a live corpus.
        """
        if self._has_collection(name):
            self._ensure_payload_indexes(name)
            return
        dt = {"float32": qm.Datatype.FLOAT32, "float16": qm.Datatype.FLOAT16}.get(datatype)
        if dt is None:
            raise ValueError(f"unsupported vector datatype {datatype!r}")
        quant = None
        if quantization == "int8":
            quant = qm.ScalarQuantization(scalar=qm.ScalarQuantizationConfig(
                type=qm.ScalarType.INT8, quantile=0.99, always_ram=False))
        elif quantization == "binary":
            quant = qm.BinaryQuantization(binary=qm.BinaryQuantizationConfig(always_ram=False))
        elif quantization:
            raise ValueError(f"unsupported quantization {quantization!r}")
        self.client.create_collection(
            collection_name=name,
            # Vecgrep corpora can grow well beyond available WSL RAM. Keep
            # both vectors and the HNSW graph disk-backed so qdrant recovery
            # does not fault an entire large collection into memory.
            vectors_config=qm.VectorParams(
                size=dim,
                distance=qm.Distance.COSINE,
                on_disk=True,
                datatype=dt,
            ),
            hnsw_config=qm.HnswConfigDiff(on_disk=True),
            quantization_config=quant,
        )
        with self._lock_for_meta():
            if getattr(self, "_known_collections", None) is None:
                self._known_collections = {name}
            else:
                self._known_collections.add(name)
        self._ensure_payload_indexes(name)

    def drop_collection(self, name: str) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        with self._lock_for_meta():
            self._known_collections = existing - {name}
            if getattr(self, "_indexed_collections", None) is not None:
                self._indexed_collections.discard(name)
        if name in existing:
            self.client.delete_collection(name)

    def list_collections(self) -> list[str]:
        """Every collection name the store holds — used to find orphans (a
        collection with no registry entry)."""
        names = [c.name for c in self.client.get_collections().collections]
        with self._lock_for_meta():
            self._known_collections = set(names)
        return names

    def count(self, name: str) -> int:
        """Live point count for a collection — the ground truth for chunk_count.

        Returns 0 for a missing collection so callers can recount unconditionally
        without a separate existence check. Used to re-derive corpus.chunk_count
        from Qdrant after an index instead of accumulating a delta, which drifts
        the moment the BM25-derived `chunks_freed` diverges from reality (e.g. a
        collection wiped out-of-band leaves a stale accumulator forever wrong)."""
        if not self._has_collection(name):
            return 0
        try:
            return self.client.count(collection_name=name, exact=True).count
        except Exception:  # noqa: BLE001 -- The storage adapter retains its existing missing-item and optional-index error contract.
            with self._lock_for_meta():
                if getattr(self, "_known_collections", None) is not None:
                    self._known_collections.discard(name)
            return 0

    def source_counts(self, name: str) -> dict[str, int]:
        """Return the live point count for each source in one collection scan.

        Recovery uses deterministic point IDs, so this one cheap payload-only
        pass distinguishes complete source files from files that need an
        upsert.  Counting each source with an unindexed filter would turn a
        large recovery into thousands of full collection scans.
        """
        if not self._has_collection(name):
            return {}
        counts: Counter[str] = Counter()
        offset = None
        selector = qm.PayloadSelectorInclude(include=["source_id"])
        while True:
            points, offset = self.client.scroll(
                collection_name=name,
                offset=offset,
                limit=1000,
                with_payload=selector,
                with_vectors=False,
            )
            for point in points:
                source_id = (point.payload or {}).get("source_id")
                if isinstance(source_id, str):
                    counts[source_id] += 1
            if offset is None:
                return dict(counts)

    def iter_payloads(
        self,
        name: str,
        *,
        exclude_payload_fields: set[str] | None = None,
        include_payload_fields: set[str] | None = None,
    ) -> Iterator[tuple[str, dict]]:
        """Stream every point ID and payload from one collection.

        Used to reconstruct a derived local index from the vector store without
        transferring vectors or making a huge all-points request. Callers can
        exclude fields not needed by the derivative; this matters when a large
        source document is deliberately duplicated on every point.
        """
        if not self._has_collection(name):
            return
        offset = None
        if include_payload_fields:
            selector = qm.PayloadSelectorInclude(include=sorted(include_payload_fields))
        elif exclude_payload_fields:
            selector = qm.PayloadSelectorExclude(exclude=sorted(exclude_payload_fields))
        else:
            selector = True
        while True:
            points, offset = self.client.scroll(
                collection_name=name,
                offset=offset,
                limit=1000,
                with_payload=selector,
                with_vectors=False,
            )
            for point in points:
                yield str(point.id), dict(point.payload or {})
            if offset is None:
                return

    def source_records(self, name: str, source_id: str) -> list[dict]:
        """Materialize one source's points for mutation rollback.

        A source update can span several Qdrant batches.  Persisting the old
        records before the first batch lets recovery restore the complete old
        version if the writer dies before Qdrant reports the stage complete.
        """
        if not self._has_collection(name):
            return []
        records: list[dict] = []
        offset = None
        query_filter = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="source_id", match=qm.MatchValue(value=source_id)
                )
            ]
        )
        while True:
            points, offset = self.client.scroll(
                collection_name=name,
                scroll_filter=query_filter,
                offset=offset,
                limit=256,
                with_payload=True,
                with_vectors=True,
            )
            for point in points:
                vector = point.vector
                if isinstance(vector, dict):
                    raise RuntimeError("named-vector collections are unsupported")  # noqa: TRY004 -- Unsupported backend snapshot shape retains the RuntimeError contract.
                records.append(
                    {
                        "id": str(point.id),
                        "vector": list(vector or []),
                        "payload": dict(point.payload or {}),
                    }
                )
            if offset is None:
                return records

    def restore_source(self, collection: str, source_id: str, records: list[dict]) -> None:
        """Replace one source with a previously captured point snapshot."""
        self.delete_by_source(collection, source_id)
        if not records:
            return
        self.upsert(
            collection,
            [str(r["id"]) for r in records],
            [list(r["vector"]) for r in records],
            [dict(r["payload"]) for r in records],
        )

    # Max points per upsert request. Only the carrier chunk holds the full
    # source_text now, so batches are far smaller than they were — but one
    # oversized document still lands its whole body in a single point, and
    # Qdrant's default request-payload ceiling is 256MB. Batching keeps every
    # request well clear of it.
    _UPSERT_BATCH = 64

    def upsert(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> int:
        if not vectors:
            return 0
        total = 0
        for start in range(0, len(vectors), self._UPSERT_BATCH):
            end = start + self._UPSERT_BATCH
            points = [
                qm.PointStruct(id=cid, vector=v, payload=p)
                for cid, v, p in zip(ids[start:end], vectors[start:end], payloads[start:end])
            ]
            # Wait only on the final batch so multi-batch upserts do not pay a
            # synchronous WAL flush per 64-point slice while still returning
            # only after all batches are durable.
            self.client.upsert(
                collection_name=collection,
                points=points,
                wait=(end >= len(vectors)),
            )
            total += len(points)
        return total

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        *,
        query_filter: qm.Filter | None = None,
    ) -> list[StoredHit]:
        try:
            kwargs: dict = {
                "collection_name": collection,
                "query": query_vector,
                "limit": top_k,
                "with_payload": True,
            }
            if query_filter is not None:
                kwargs["query_filter"] = query_filter
            response = self.client.query_points(**kwargs)
        except Exception as exc:
            raise SearchBackendError("Search storage is unavailable; retry the query.") from exc
        results = response.points
        hits: list[StoredHit] = []
        for r in results:
            p = r.payload or {}
            hits.append(
                StoredHit(
                    score=float(r.score),
                    chunk_text=p.get("text", ""),
                    source_id=p.get("source_id", ""),
                    source_text=p.get("source_text", ""),
                    chunk_start=int(p.get("chunk_start", 0)),
                    chunk_end=int(p.get("chunk_end", 0)),
                    chunk_index=int(p.get("chunk_index", 0)),
                    metadata=p.get("metadata", {}) or {},
                    corpus=p.get("corpus", collection),
                    context_before=p.get("context_before"),
                    context_after=p.get("context_after"),
                    line_start=p.get("line_start"),
                    line_end=p.get("line_end"),
                    source_length=p.get("source_length"),
                    doc_timestamp=p.get("doc_timestamp"),
                )
            )
        return hits

    def get_vector(self, collection: str, point_id: str) -> list[float] | None:
        """Stored embedding vector for one point — powers query-by-example
        (`related`) without re-embedding. None if the point is missing."""
        if not self._has_collection(collection):
            return None
        try:
            points = self.client.retrieve(
                collection_name=collection,
                ids=[point_id],
                with_payload=False,
                with_vectors=True,
            )
        except Exception:  # noqa: BLE001 -- The storage adapter retains its existing missing-item and optional-index error contract.
            return None
        if not points or points[0].vector is None:
            return None
        v = points[0].vector
        # Named-vector collections return a dict; we use a single unnamed one.
        return list(v) if not isinstance(v, dict) else None

    def get_by_id(self, collection: str, point_id: str) -> dict | None:
        """Fetch a single point's payload by id. Returns None if missing."""
        if not self._has_collection(collection):
            return None
        try:
            points = self.client.retrieve(
                collection_name=collection,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
        except Exception:  # noqa: BLE001 -- The storage adapter retains its existing missing-item and optional-index error contract.
            return None
        if not points:
            return None
        return points[0].payload or {}

    def get_many_by_id(
        self, collection: str, point_ids: list[str]
    ) -> dict[str, dict]:
        """Fetch several payloads in one round trip, keyed by point id.

        Missing ids are simply absent from the result — callers treat that
        the same as get_by_id returning None.
        """
        if not point_ids:
            return {}
        try:
            points = self.client.retrieve(
                collection_name=collection,
                ids=point_ids,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:  # noqa: BLE001 -- The storage adapter retains its existing missing-item and optional-index error contract.
            return {}
        return {str(p.id): (p.payload or {}) for p in points}

    def set_payload(self, collection: str, point_id: str, patch: dict) -> None:
        """Merge keys into one point's payload. Vectors are untouched."""
        self.client.set_payload(
            collection_name=collection,
            payload=patch,
            points=[point_id],
            wait=True,
        )

    def clear_payload_keys(
        self, collection: str, point_id: str, keys: list[str]
    ) -> None:
        """Remove payload keys from one point. Vectors are untouched."""
        if not keys:
            return
        self.client.delete_payload(
            collection_name=collection,
            keys=list(keys),
            points=[point_id],
            wait=True,
        )

    def apply_payload_ops(
        self,
        collection: str,
        set_ops: list[tuple[str, dict]],
        delete_keys: list[tuple[str, list[str]]],
    ) -> None:
        """Apply many per-point payload edits in ONE request.

        Each point needs a different patch, so these can't be collapsed into
        a single set_payload call — but they can ride in one batch request.
        That matters beyond speed: a request-per-point run burns an
        ephemeral port per point, and a host with a narrow
        net.ipv4.ip_local_port_range will exhaust it and start refusing
        connections long before the walk finishes.
        """
        ops: list = []
        for point_id, patch in set_ops:
            ops.append(
                qm.SetPayloadOperation(
                    set_payload=qm.SetPayload(payload=patch, points=[point_id])
                )
            )
        for point_id, keys in delete_keys:
            ops.append(
                qm.DeletePayloadOperation(
                    delete_payload=qm.DeletePayload(keys=list(keys), points=[point_id])
                )
            )
        if not ops:
            return
        self.client.batch_update_points(
            collection_name=collection, update_operations=ops, wait=True
        )

    def clear_payload_keys_bulk(
        self, collection: str, point_ids: list[str], keys: list[str]
    ) -> None:
        """clear_payload_keys over many points in one request."""
        if not keys or not point_ids:
            return
        self.client.delete_payload(
            collection_name=collection,
            keys=list(keys),
            points=list(point_ids),
            wait=True,
        )

    def delete_points(self, collection: str, point_ids: list[str]) -> None:
        if not point_ids or not self._has_collection(collection):
            return
        self.client.delete(
            collection_name=collection,
            points_selector=qm.PointIdsList(points=list(point_ids)),
            wait=True,
        )

    def delete_by_source(self, collection: str, source_id: str) -> None:
        if not self._has_collection(collection):
            return
        self.client.delete(
            collection_name=collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="source_id", match=qm.MatchValue(value=source_id))]
                )
            ),
            wait=True,
        )
