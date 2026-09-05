"""Server-mode QdrantStore must reuse TCP connections.

qdrant-client defaults to Limits(max_keepalive_connections=0) -- pooling
off. Every request then opens a fresh connection and parks it in TIME_WAIT
for 60s, so a sustained request rate exhausts the host's ephemeral port
range and subsequent connects fail with ECONNRESET mid-run.

This is a unit check on the constructed client's limits: it needs no live
Qdrant, so it stays hermetic like the rest of the suite.
"""
from __future__ import annotations

import pytest

from qdrant_client import QdrantClient
from vecgrep.backend.store.qdrant_store import QdrantStore


@pytest.fixture(autouse=True)
def no_compatibility_probe(monkeypatch):
    original_init = QdrantClient.__init__

    def offline_init(client, *args, **kwargs):
        # Inspect the real transport pool without probing a live daemon.
        kwargs["check_compatibility"] = False
        original_init(client, *args, **kwargs)

    monkeypatch.setattr(QdrantClient, "__init__", offline_init)


def test_server_mode_enables_connection_keepalive() -> None:
    store = QdrantStore(path=None, url="http://127.0.0.1:6333")
    limits = store.client._client._rest_args["limits"]
    assert limits.max_keepalive_connections >= 8, (
        "pooling disabled — every request would burn an ephemeral port"
    )
    assert limits.keepalive_expiry >= 30
    assert limits.max_connections and limits.max_connections >= limits.max_keepalive_connections


def test_server_mode_client_is_shared_across_api_groups() -> None:
    """One httpx client behind every api group, so the pool is actually
    shared rather than one pool per endpoint family."""
    store = QdrantStore(path=None, url="http://127.0.0.1:6333")
    oc = store.client._client.openapi_client
    clients = {
        id(getattr(oc, name).api_client._client)
        for name in dir(oc)
        if name.endswith("_api")
    }
    assert len(clients) == 1
