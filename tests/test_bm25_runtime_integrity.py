"""Per-corpus cutover cannot silently discard lexical retrieval."""
import sqlite3

import pytest

from vecgrep.backend.config import Settings, ConfigError, load_settings
from vecgrep.backend.service import VecgrepService, CorpusError
from vecgrep.backend.store.bm25_sqlite import BM25SqliteStore


@pytest.fixture
def sqlite_svc(vg_home, stub_backend):
    from vecgrep.backend.embed.cache import CachedBackend

    settings = Settings(home=vg_home, bm25_backend="sqlite")
    service = VecgrepService(settings=settings)
    service._backend_cache["auto"] = CachedBackend(stub_backend, service._embed_cache)
    yield service
    service.bm25.close_all()
    service.store.client.close()


@pytest.mark.parametrize("empty", [False, True])
def test_missing_or_empty_sidecar_rejects_search_and_can_be_repaired(
    sqlite_svc, make_doc, empty
):
    sqlite_svc.index(str(make_doc("sample.txt", "Unique error E1234 occurred.")), "notes")
    sqlite_svc.bm25.drop("notes")
    if empty:
        sqlite_svc.bm25.count("notes")
    with pytest.raises(CorpusError, match="BM25.*rebuild"):
        sqlite_svc.search("E1234", corpus_name="notes", mode="hybrid")
    assert sqlite_svc.rebuild_bm25("notes") > 0
    assert sqlite_svc.search("E1234", corpus_name="notes", mode="hybrid")


def test_sqlite_operational_failure_is_not_an_empty_result(tmp_path):
    store = BM25SqliteStore(tmp_path)
    store.upsert("notes", ["a"], ["E1234"], [{"source_id": "sample"}])
    store._conn("notes").execute("DROP TABLE chunks_fts")
    with pytest.raises(sqlite3.OperationalError):
        store.search("notes", "E1234", 5)
    store.close_all()


def test_source_intent_retires_only_after_sqlite_is_committed(sqlite_svc, tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Unique error E1234 occurred.")
    (docs / "b.txt").write_text("A second source says E5678.")
    observer = BM25SqliteStore(sqlite_svc.settings.home / "bm25")
    finish = sqlite_svc.mutations.finish
    observed = []

    def checked_finish(corpus):
        expected = sqlite_svc.registry.get(corpus).chunk_count
        assert observer.count(corpus) == expected
        observed.append(expected)
        return finish(corpus)

    monkeypatch.setattr(sqlite_svc.mutations, "finish", checked_finish)
    try:
        sqlite_svc.index(str(docs), "notes")
        assert len(observed) == 2
    finally:
        observer.close_all()
