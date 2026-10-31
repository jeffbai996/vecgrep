"""Audit reproduction tests for Area 2: Indexing."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vecgrep.backend.config import Settings
from vecgrep.backend.embed.base import EmbedBackend, EmbedBackendError
from vecgrep.backend.service import VecgrepService, _collection_for


class _FakeEmbedder(EmbedBackend):
    name = "ollama"
    model = "nomic-embed-text"
    dim = 16

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = [(b / 127.5) - 1.0 for b in h[: self.dim]]
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out


def _make_service(tmp_path: Path, *, bm25_sqlite: bool = False) -> VecgrepService:
    settings = Settings(
        home=tmp_path / "vg_home",
        qdrant_url=None,
        bm25_backend="sqlite" if bm25_sqlite else "pickle",
    )
    svc = VecgrepService(settings=settings, ephemeral=False)
    svc._backend_cache["auto"] = _FakeEmbedder()
    return svc


def test_f_index_01_incremental_reindex_duplicates_bm25_and_retains_deleted_text(tmp_path: Path):
    """F-INDEX-01a: Incremental update appends duplicate chunk IDs into BM25Store without deleting
    old entries for modified chunks, leaving deleted text searchable in BM25 and hybrid modes."""
    svc = _make_service(tmp_path, bm25_sqlite=False)
    doc = tmp_path / "doc.md"
    sec1 = "# Section One\n\nThis first section stays completely unchanged across edits so chunk zero is reused.\n"
    sec2_old = "# Section Two\n\nThe old secret codename is XYZZYPLUGH and should vanish after editing.\n"
    doc.write_text(f"{sec1}\n{sec2_old}", encoding="utf-8")

    svc.index(str(doc), "notes", chunker_name="markdown_section")
    assert svc.store.count(_collection_for("notes")) == 2
    assert svc.bm25.count("notes") == 2

    # Edit Section Two while keeping Section One unchanged -> triggers incremental_update = True
    sec2_new = "# Section Two\n\nThe new replacement codename is FROBOZZ and replaces the old one.\n"
    doc.write_text(f"{sec1}\n{sec2_new}", encoding="utf-8")
    svc.index(str(doc), "notes", chunker_name="markdown_section")

    # Qdrant still has 2 chunks, but BM25Store appended 2 more without deleting the existing ones!
    assert svc.store.count(_collection_for("notes")) == 2
    assert svc.bm25.count("notes") == 2  # BUG: actual is 4!

    # Worse: searching for the deleted token "XYZZYPLUGH" still returns the deleted chunk text!
    ghost_hits = svc.search("XYZZYPLUGH", "notes", mode="hybrid")
    assert ghost_hits == []


def test_f_index_01_incremental_reindex_breaks_bm25_sqlite_on_next_search(tmp_path: Path):
    """F-INDEX-01b: When bm25_sqlite=True, incremental update appends duplicate rows in SQLite
    BM25, causing _ensure_bm25_loaded() on the very next search to raise CorpusError."""
    svc = _make_service(tmp_path, bm25_sqlite=True)
    doc = tmp_path / "doc.md"
    sec1 = "# Section One\n\nThis first section stays completely unchanged across edits so chunk zero is reused.\n"
    sec2_old = "# Section Two\n\nInitial version of section two before incremental edit.\n"
    doc.write_text(f"{sec1}\n{sec2_old}", encoding="utf-8")

    svc.index(str(doc), "notes", chunker_name="markdown_section")
    sec2_new = "# Section Two\n\nUpdated version of section two after incremental edit.\n"
    doc.write_text(f"{sec1}\n{sec2_new}", encoding="utf-8")
    svc.index(str(doc), "notes", chunker_name="markdown_section")

    # BUG: Next search raises CorpusError("BM25 SQLite index for 'notes' is missing or inconsistent...")
    hits = svc.search("Updated version", "notes", mode="hybrid")
    assert len(hits) >= 1


def test_f_index_02_directory_index_and_moved_file_leave_duplicate_sources(tmp_path: Path):
    """F-INDEX-02: Indexing a moved/renamed file inside a directory leaves the old path's chunks
    active in the corpus, returning duplicate hits for the same document."""
    svc = _make_service(tmp_path)
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    p1 = docs_dir / "old_name.md"
    content = "# Architecture Decision\n\nWe selected SQLite WAL mode for local metadata persistence.\n"
    p1.write_text(content, encoding="utf-8")

    svc.index(str(docs_dir), "arch")
    assert svc.registry.get("arch").doc_count == 1

    # Rename file inside the watched/indexed directory and re-index the directory
    p2 = docs_dir / "new_name.md"
    p1.rename(p2)
    svc.index(str(docs_dir), "arch")

    # BUG: old_name.md is still in corpus.sources and its chunks remain in Qdrant & BM25
    corpus = svc.registry.get("arch")
    assert corpus.doc_count == 1
    assert str(p1.resolve()) not in corpus.sources


def test_f_index_03_bm25_bulk_unlinks_dirty_marker_on_exception(tmp_path: Path):
    """F-INDEX-03: BM25Store.bulk() deletes the .dirty crash marker in finally even when an
    exception aborts the block before _persist_now() runs."""
    svc = _make_service(tmp_path)
    with pytest.raises(RuntimeError, match="simulated crash mid-batch"), svc.bm25.bulk("notes"):
        svc.bm25.upsert(
            "notes",
            ["c1"],
            ["important indexed text that never reached disk"],
            [{"corpus": "notes", "source_id": "a.md", "chunk_index": 0, "text": "important"}],
        )
        raise RuntimeError("simulated crash mid-batch")

    # Because _persist_now("notes") was skipped on exception, the on-disk sidecar is missing/stale,
    # so "notes" MUST remain in dirty_corpora() so the next startup rebuilds BM25 from Qdrant.
    # BUG: finally unlinked the .dirty marker even though _persist_now never ran!
    assert "notes" in svc.bm25.dirty_corpora()


def test_f_index_04_migrate_corpus_destroys_live_corpus_on_mid_migration_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """F-INDEX-04: migrate_corpus deletes the live corpus before copying points from the temporary
    migration collection, without holding locks.write(name) or writing a crash journal. Any error
    during the post-delete copy permanently loses the corpus from the registry."""
    svc = _make_service(tmp_path)
    doc = tmp_path / "doc.md"
    doc.write_text("# Doc\n\nImportant content that must survive failed migration.\n", encoding="utf-8")
    svc.index(str(doc), "notes")
    assert svc.registry.has("notes")

    import vecgrep.backend.service as svc_mod

    replacement = _FakeEmbedder()
    replacement.model = "bge-m3"
    monkeypatch.setattr(svc_mod, "get_embed_backend", lambda *a, **kw: replacement)
    # Simulate an exception right after delete_corpus(name) when scrolling temp collection
    orig_client = svc.store.client
    orig_scroll = orig_client.scroll

    failed = False
    def _boom_scroll(*args, **kwargs):
        nonlocal failed
        if not failed and not svc.registry.has("notes") and kwargs.get("collection_name") == _collection_for("__migrate__notes__") and kwargs.get("with_vectors"):
            failed = True
            raise RuntimeError("simulated crash after delete_corpus during migration")
        return orig_scroll(*args, **kwargs)

    monkeypatch.setattr(orig_client, "scroll", _boom_scroll)
    with pytest.raises(RuntimeError, match="simulated crash"):
        svc.migrate_corpus("notes", to_backend="ollama", to_model="bge-m3")

    # Even after recovery, the original "notes" corpus was permanently deleted!
    svc.store.client.close()
    restored = _make_service(tmp_path)
    restored.recover_pending_mutations()
    assert restored.registry.has("notes")
    assert restored.store.count(_collection_for("notes")) == 1
    assert restored.registry.get("notes").embed_model == "bge-m3"
    assert restored.search("Important", "notes", mode="bm25")
    restored.store.client.close()



def test_f_index_05_failed_force_index_leaves_embed_cache_bypass_stuck_true(tmp_path: Path):
    """F-INDEX-05: _index_locked sets backend.bypass = True without try/finally; if indexing raises,
    the shared CachedBackend stays in bypass=True mode permanently."""
    calls = 0

    class _CountingEmbedder(_FakeEmbedder):
        should_fail = False

        def embed(self, texts: list[str]) -> list[list[float]]:
            nonlocal calls
            calls += 1
            if self.should_fail:
                raise EmbedBackendError("simulated transient ollama failure")
            return super().embed(texts)

    settings = Settings(home=tmp_path / "vg_home", qdrant_url=None)
    raw_backend = _CountingEmbedder()
    svc = VecgrepService(settings=settings, ephemeral=False)
    # Register CachedBackend wrapping raw_backend in svc._backend_cache
    from vecgrep.backend.embed.cache import CachedBackend

    assert svc._embed_cache is not None
    cached_backend = CachedBackend(raw_backend, svc._embed_cache)
    svc._backend_cache["ollama:nomic-embed-text"] = cached_backend
    svc._backend_cache["auto"] = cached_backend

    doc = tmp_path / "note.md"
    doc.write_text("# Note\n\nCached text for embedding cache verification test.\n", encoding="utf-8")

    # Initial index warms the embedding cache
    svc.index(str(doc), "notes")
    assert cached_backend.bypass is False

    # Now a force=True index fails mid-flight
    raw_backend.should_fail = True
    with pytest.raises(EmbedBackendError):
        svc.index(str(doc), "notes", force=True)

    # BUG: cached_backend.bypass was never restored to False!
    assert cached_backend.bypass is False
