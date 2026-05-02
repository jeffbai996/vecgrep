"""Qdrant wrapper.

Embedded mode (path=...) — no Docker, no server. Each named corpus maps
to its own Qdrant collection. Payload schema:

    {
      "corpus": str,
      "source_id": str,
      "source_text": str,        # the full source text, kept once on the
                                  #   first chunk of each source so context
                                  #   reconstruction needs no extra storage
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


class QdrantStore:
    def __init__(self, path: Path | None) -> None:
        # path=None -> in-memory (ephemeral). path=Path(...) -> on-disk.
        if path is None:
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

    def upsert(
        self,
        collection: str,
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> int:
        if not vectors:
            return 0
        points = [
            qm.PointStruct(id=str(uuid.uuid4()), vector=v, payload=p)
            for v, p in zip(vectors, payloads)
        ]
        self.client.upsert(collection_name=collection, points=points, wait=True)
        return len(points)

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
                )
            )
        return hits

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

    def list_sources(self, collection: str) -> list[str]:
        existing = {c.name for c in self.client.get_collections().collections}
        if collection not in existing:
            return []
        seen: set[str] = set()
        offset: int | str | None = None
        while True:
            batch, offset = self.client.scroll(
                collection_name=collection,
                with_payload=True,
                limit=512,
                offset=offset,
            )
            for point in batch:
                if point.payload:
                    sid = point.payload.get("source_id")
                    if sid:
                        seen.add(sid)
            if offset is None:
                break
        return sorted(seen)
