from __future__ import annotations

import hashlib
import threading
import time

import pytest

from vecgrep.backend.service import _collection_for


def test_search_waits_for_same_corpus_commit(svc, make_doc, monkeypatch):
    source = make_doc("doc.md", "old marker sentence")
    svc.index(str(source), "notes")
    source.write_text("new marker sentence")

    entered = threading.Event()
    release = threading.Event()
    original = svc.store.upsert

    def paused(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(svc.store, "upsert", paused)
    writer = threading.Thread(
        target=lambda: svc.index(str(source), "notes", force=True), daemon=True
    )
    writer.start()
    assert entered.wait(5)

    finished = threading.Event()
    results = []

    def searcher():
        results.extend(svc.search("new marker", "notes"))
        finished.set()

    reader = threading.Thread(target=searcher, daemon=True)
    reader.start()
    time.sleep(0.15)
    assert not finished.is_set(), "search crossed a half-committed mutation"
    release.set()
    writer.join(5)
    reader.join(5)
    assert finished.is_set()
    assert results


def test_recovery_rolls_forward_after_qdrant_commit(svc, make_doc, monkeypatch):
    source = make_doc("doc.md", "old marker sentence")
    svc.index(str(source), "notes")
    source.write_text("new marker sentence")

    original = svc.bm25.upsert

    def fail_after_qdrant(*args, **kwargs):
        raise RuntimeError("fault after qdrant")

    monkeypatch.setattr(svc.bm25, "upsert", fail_after_qdrant)
    with pytest.raises(RuntimeError, match="fault after qdrant"):
        svc.index(str(source), "notes", force=True)
    record = svc.mutations.read("notes")
    assert record and record["phase"] == "qdrant_done"

    monkeypatch.setattr(svc.bm25, "upsert", original)
    assert svc.recover_pending_mutations() == ["notes"]
    assert svc.mutations.read("notes") is None
    assert svc.search("new marker", "notes")
    corpus = svc.registry.get("notes")
    assert corpus.source_hashes[str(source.resolve())] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert corpus.chunk_count == svc.store.count(_collection_for("notes"))


def test_recovery_rolls_back_partial_qdrant_batch(svc, make_doc, monkeypatch):
    old_text = "Old one. Old two. Old three."
    source = make_doc("doc.md", old_text)
    svc.index(str(source), "notes")
    source.write_text("New one. New two. New three. New four.")

    original = svc.store.upsert

    def partial(collection, ids, vectors, payloads):
        original(collection, ids[:1], vectors[:1], payloads[:1])
        raise RuntimeError("fault in qdrant batch")

    monkeypatch.setattr(svc.store, "upsert", partial)
    with pytest.raises(RuntimeError, match="fault in qdrant batch"):
        svc.index(str(source), "notes", force=True)
    record = svc.mutations.read("notes")
    assert record and record["phase"] == "prepared"

    monkeypatch.setattr(svc.store, "upsert", original)
    assert svc.recover_pending_mutations() == ["notes"]
    payloads = list(svc.store.iter_payloads(_collection_for("notes")))
    assert payloads
    assert {p[1]["source_text"] for p in payloads} == {old_text}
    assert svc.search("Old two", "notes")


def test_recovery_completes_interrupted_delete(svc, make_doc, monkeypatch):
    source = make_doc("doc.md", "delete me marker")
    svc.index(str(source), "notes")

    original = svc.bm25.delete_by_source

    def fail_after_qdrant(*args, **kwargs):
        raise RuntimeError("fault during delete")

    monkeypatch.setattr(svc.bm25, "delete_by_source", fail_after_qdrant)
    with pytest.raises(RuntimeError, match="fault during delete"):
        svc.delete_source("notes", str(source.resolve()))
    monkeypatch.setattr(svc.bm25, "delete_by_source", original)

    assert svc.recover_pending_mutations() == ["notes"]
    assert svc.store.count(_collection_for("notes")) == 0
    assert svc.registry.get("notes").chunk_count == 0
    assert not svc.bm25.search("notes", "delete", 10)
