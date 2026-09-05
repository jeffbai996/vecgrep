"""Dropping a corpus must leave none of its bytes on disk.

The sqlite backend removed the .db and its -wal/-shm sidecars and walked past
the pickle it had been converted FROM. Since convert() builds <corpus>.db out
of <corpus>.pkl, every converted corpus carries both, so a drop freed the
qdrant collection and silently kept the larger half of the sidecar.

Found by deleting three finished eval corpora (2026-08-30): qdrant gave back
639 MB, and 298 MB of orphaned pickle stayed behind owning corpora that no
longer existed. Nothing reads those files, so nothing failed — the bytes just
never came back.
"""
from __future__ import annotations

from vecgrep.backend.store.bm25_sqlite import BM25SqliteStore


def _seed(store: BM25SqliteStore, corpus: str, text: str) -> None:
    store.upsert(corpus, ["c1"], [text],
                 [{"source_id": f"/{corpus}", "chunk_index": 0, "metadata": {}}])


def test_dropping_a_corpus_leaves_none_of_its_bytes_behind(tmp_path):
    root = tmp_path / "bm25"
    store = BM25SqliteStore(root)
    _seed(store, "gone", "the quick brown fox jumps over the lazy dog")
    # A converted corpus keeps the pickle it was built from, plus the dirty
    # marker the pickle backend writes beside it.
    (root / "gone.pkl").write_bytes(b"legacy inverted index")
    (root / "gone.pkl.dirty").write_bytes(b"")

    store.drop("gone")

    survivors = sorted(p.name for p in root.iterdir() if p.name.startswith("gone"))
    assert survivors == [], f"drop left bytes behind: {survivors}"


def test_dropping_one_corpus_does_not_touch_its_neighbour(tmp_path):
    """The new deletion builds paths from the corpus name, so a prefix collision
    or a stray glob would take a live corpus with it."""
    root = tmp_path / "bm25"
    store = BM25SqliteStore(root)
    _seed(store, "gone", "the quick brown fox")
    _seed(store, "gone-too", "memory pressure on the host forced a reclaim")
    (root / "gone.pkl").write_bytes(b"legacy")
    (root / "gone-too.pkl").write_bytes(b"legacy")

    store.drop("gone")

    assert (root / "gone-too.db").exists()
    assert (root / "gone-too.pkl").exists(), "a name that merely starts the same"
    assert store.search("gone-too", "memory pressure", 3), "neighbour still ranks"


def test_dropping_a_corpus_that_was_never_converted_is_not_an_error(tmp_path):
    """Most corpora have no .pkl at all. missing_ok covers them."""
    root = tmp_path / "bm25"
    store = BM25SqliteStore(root)
    _seed(store, "fresh", "swap exhaustion and page cache reclaim")

    store.drop("fresh")
    store.drop("never-existed")

    assert sorted(p.name for p in root.iterdir()) == []
