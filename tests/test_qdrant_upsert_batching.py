"""QdrantStore.upsert must batch large point sets.

vecgrep stores the full source_text on every chunk, so a big document's chunks
can sum past Qdrant's 256MB request-payload ceiling. A single all-points upsert
then 400s and aborts indexing. upsert() batches to keep each request small.
"""
from __future__ import annotations

from vecgrep.backend.store.qdrant_store import QdrantStore


class _FakeClient:
    """Captures upsert calls without a real Qdrant."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def upsert(self, collection_name, points, wait):  # noqa: D401
        self.batch_sizes.append(len(points))


def _store_with_fake() -> tuple[QdrantStore, _FakeClient]:
    # Bypass __init__ (which would spin up a real Qdrant); inject the fake.
    s = QdrantStore.__new__(QdrantStore)
    fake = _FakeClient()
    s.client = fake
    return s, fake


def _rows(n: int):
    ids = [f"id-{i}" for i in range(n)]
    vecs = [[0.1, 0.2, 0.3] for _ in range(n)]
    payloads = [{"text": f"chunk {i}"} for i in range(n)]
    return ids, vecs, payloads


def test_small_set_single_batch() -> None:
    s, fake = _store_with_fake()
    ids, vecs, payloads = _rows(10)
    n = s.upsert("c", ids, vecs, payloads)
    assert n == 10
    assert fake.batch_sizes == [10]  # one request


def test_large_set_is_split() -> None:
    s, fake = _store_with_fake()
    n_points = QdrantStore._UPSERT_BATCH * 3 + 5  # 197 with default 64
    ids, vecs, payloads = _rows(n_points)
    n = s.upsert("c", ids, vecs, payloads)
    assert n == n_points  # all points written
    # Batched: 64 + 64 + 64 + 5
    assert len(fake.batch_sizes) == 4
    assert fake.batch_sizes == [64, 64, 64, 5]
    assert sum(fake.batch_sizes) == n_points


def test_exact_multiple_of_batch() -> None:
    s, fake = _store_with_fake()
    n_points = QdrantStore._UPSERT_BATCH * 2
    ids, vecs, payloads = _rows(n_points)
    s.upsert("c", ids, vecs, payloads)
    assert fake.batch_sizes == [64, 64]


def test_empty_is_noop() -> None:
    s, fake = _store_with_fake()
    assert s.upsert("c", [], [], []) == 0
    assert fake.batch_sizes == []
