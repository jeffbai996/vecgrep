"""Qdrant wrapper.

Embedded mode (path=...) — no Docker, no server. Each named corpus maps
to its own Qdrant collection. Payload schema:

    {
      "corpus": str,
      "source_id": str,
      "source_text": str,        # the full source text, duplicated onto
                                  #   every chunk so any single hit can
                                  #   reconstruct surrounding context
                                  #   without a second store lookup. Wastes
                                  #   space (one copy per chunk, not per
                                  #   source) — fine at MVP scale, slated
                                  #   for v1.0 cleanup once ingestion
                                  #   targets large corpora like full
                                  #   Discord/ChatGPT exports.
      "chunk_index": int,
      "chunk_start": int,
      "chunk_end": int,
      "text": str,                # the chunk itself
      "metadata": {...},
    }
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


@dataclass
class StoredHit:
    score: float
    chunk_text: str
    source_id: str
    source_text: str
    chunk_start: int
    chunk_end: int
    chunk_index: int
    metadata: dict
    corpus: str
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
            self.client = QdrantClient(url=url, timeout=120)
        elif path is None:
            self.client = QdrantClient(":memory:")
        else:
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))

    def ensure_collection(self, name: str, dim: int) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if name in existing:
            return
        self.client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

    def drop_collection(self, name: str) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if name in existing:
            self.client.delete_collection(name)

    # Max points per upsert request. We store the full source_text on every
    # chunk, so a large document's chunks can sum to >256MB — Qdrant's default
    # request-payload ceiling — and a single all-points upsert 400s. Batching
    # keeps each request well under the limit. 64 is conservative for sources
    # up to a few MB; smaller if individual chunks are unusually large.
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
            self.client.upsert(collection_name=collection, points=points, wait=True)
            total += len(points)
        return total

    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[StoredHit]:
        try:
            response = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
        except Exception:
            return []
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
                    doc_timestamp=p.get("doc_timestamp"),
                )
            )
        return hits

    def get_by_id(self, collection: str, point_id: str) -> dict | None:
        """Fetch a single point's payload by id. Returns None if missing."""
        existing = {c.name for c in self.client.get_collections().collections}
        if collection not in existing:
            return None
        try:
            points = self.client.retrieve(
                collection_name=collection,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return None
        if not points:
            return None
        return points[0].payload or {}

    def delete_by_source(self, collection: str, source_id: str) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if collection not in existing:
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

