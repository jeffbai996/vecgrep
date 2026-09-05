import os

import pytest

from vecgrep.backend.service import VecgrepService
from vecgrep.backend.store import BM25SqliteStore, BM25Store


def _close(service: VecgrepService) -> None:
    service.store.client.close()
    close_all = getattr(service.bm25, "close_all", None)
    if close_all:
        close_all()


def test_service_uses_pickle_bm25_by_default(vg_home, monkeypatch):
    monkeypatch.delenv("VECGREP_BM25_BACKEND", raising=False)
    service = VecgrepService(ephemeral=False)
    try:
        assert isinstance(service.bm25, BM25Store)
    finally:
        _close(service)


def test_service_uses_sqlite_bm25_when_requested(vg_home, monkeypatch):
    monkeypatch.setenv("VECGREP_BM25_BACKEND", "sqlite")
    service = VecgrepService(ephemeral=False)
    try:
        assert isinstance(service.bm25, BM25SqliteStore)
    finally:
        _close(service)


def test_sqlite_backend_refuses_unmigrated_pickle_indexes(vg_home, monkeypatch):
    bm25 = vg_home / "bm25"
    bm25.mkdir(parents=True)
    (bm25 / "important.pkl").write_bytes(b"not migrated")
    monkeypatch.setenv("VECGREP_BM25_BACKEND", "sqlite")

    with pytest.raises(RuntimeError, match="important.*migrate"):
        VecgrepService(ephemeral=False)


def test_sqlite_backend_refuses_a_database_older_than_its_pickle(vg_home, monkeypatch):
    bm25 = vg_home / "bm25"
    bm25.mkdir(parents=True)
    database = bm25 / "important.db"
    pickle = bm25 / "important.pkl"
    database.write_bytes(b"old database")
    pickle.write_bytes(b"new pickle")
    os.utime(database, ns=(1, 1))
    os.utime(pickle, ns=(2, 2))
    monkeypatch.setenv("VECGREP_BM25_BACKEND", "sqlite")

    with pytest.raises(RuntimeError, match="important.*migrate"):
        VecgrepService(ephemeral=False)
