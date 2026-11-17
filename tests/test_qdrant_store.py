from __future__ import annotations

from typing import Any, cast

import pytest
from qdrant_client import QdrantClient

from vecgrep.backend.service import COSINE_FLOOR_MARGIN, _calibration_for, _cosine_floor
from vecgrep.backend.store.qdrant_store import QdrantStore, StoredHit

# ── test_cosine_floor ──

def test_floor_is_center_minus_margin():
    for model in ("bge-m3", "nomic-embed-text", "mxbai-embed-large", None, "unknown"):
        center, _ = _calibration_for(model)
        assert abs(_cosine_floor(model) - max(0.0, center - COSINE_FLOOR_MARGIN)) < 1e-9


def test_floor_never_negative():
    # A huge margin disables the floor (clamped to 0, no filtering).
    import vecgrep.backend.service as S
    saved = S.COSINE_FLOOR_MARGIN
    try:
        S.COSINE_FLOOR_MARGIN = 5.0
        assert S._cosine_floor("bge-m3") == 0.0
    finally:
        S.COSINE_FLOOR_MARGIN = saved


def _hit(score: float, text: str, idx: int) -> StoredHit:
    return StoredHit(
        score=score, chunk_text=text, source_id=f"s{idx}", source_text=text,
        chunk_start=0, chunk_end=len(text), chunk_index=idx, metadata={},
        corpus="test",
    )


def test_subnoise_vector_hits_dropped_before_results(svc, make_doc, monkeypatch):
    p = make_doc("doc.md", "alpha content here. beta content here.")
    svc.index(str(p), "test")
    corpus = svc.list_corpora()[0]
    floor = _cosine_floor(corpus.embed_model)

    strong = _hit(min(1.0, floor + 0.2), "strong real hit", 0)
    junk = _hit(max(0.0, floor - 0.2), "subnoise junk hit", 1)
    # Return both regardless of the query vector; the floor must drop `junk`.
    monkeypatch.setattr(svc.store, "search", lambda *a, **k: [strong, junk])

    hits = svc.search("anything", "test", mode="vector", top_k=10)
    texts = " ".join(h.chunk for h in hits)
    assert "strong real hit" in texts
    assert "subnoise junk hit" not in texts, "a sub-floor vector hit leaked through"

# ── test_qdrant_connection_pooling ──

@pytest.fixture
def no_compatibility_probe(monkeypatch):
    original_init = QdrantClient.__init__

    def offline_init(client, *args, **kwargs):
        # Inspect the real transport pool without probing a live daemon.
        kwargs["check_compatibility"] = False
        original_init(client, *args, **kwargs)

    monkeypatch.setattr(QdrantClient, "__init__", offline_init)


def test_server_mode_enables_connection_keepalive(no_compatibility_probe: None) -> None:
    store = QdrantStore(path=None, url="http://127.0.0.1:6333")
    raw_client: Any = store.client._client
    limits = raw_client._rest_args["limits"]
    assert limits.max_keepalive_connections >= 8, (
        "pooling disabled — every request would burn an ephemeral port"
    )
    assert limits.keepalive_expiry >= 30
    assert limits.max_connections and limits.max_connections >= limits.max_keepalive_connections


def test_server_mode_client_is_shared_across_api_groups(no_compatibility_probe: None) -> None:
    """One httpx client behind every api group, so the pool is actually
    shared rather than one pool per endpoint family."""
    store = QdrantStore(path=None, url="http://127.0.0.1:6333")
    raw_client: Any = store.client._client
    oc = raw_client.openapi_client
    clients = {
        id(getattr(oc, name).api_client._client)
        for name in dir(oc)
        if name.endswith("_api")
    }
    assert len(clients) == 1

# ── test_qdrant_upsert_batching ──

class _FakeClient:
    """Captures upsert calls without a real Qdrant."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def upsert(self, collection_name, points, wait):
        self.batch_sizes.append(len(points))


def _store_with_fake() -> tuple[QdrantStore, _FakeClient]:
    # Bypass __init__ (which would spin up a real Qdrant); inject the fake.
    s = QdrantStore.__new__(QdrantStore)
    fake = _FakeClient()
    s.client = cast(Any, fake)
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
