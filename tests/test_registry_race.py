"""Registry write-race + mixed-model resolution.

Two coupled bugs this guards against:
1. A long-lived process (the server) clobbering another writer's (the CLI's)
   registry changes with a stale in-memory copy. The registry lock spans the
   complete reload-modify-replace cycle.
2. The engine being unable to serve corpora embedded with different models —
   query-embed backend now resolves per-corpus, not from one global env model.
"""
from __future__ import annotations

import multiprocessing
from pathlib import Path

from vecgrep.backend.store.corpora import Corpus, CorpusRegistry


def _corpus(name: str, model: str = "nomic-embed-text", dim: int = 768) -> Corpus:
    return Corpus(name=name, embed_backend="ollama", embed_model=model, dim=dim)


def _barrier_upsert(path: str, corpus: str, ready, start) -> None:
    registry = CorpusRegistry(path=Path(path))
    ready.put(corpus)
    start.wait(5)
    registry.upsert(_corpus(corpus))


def test_upsert_reloads_so_concurrent_writes_survive(tmp_path):
    """Writer A and writer B each hold their own registry on the same file.
    B adds a corpus; A (with stale in-memory state) then adds a different one.
    A's save must NOT erase B's corpus."""
    path = tmp_path / "corpora.json"
    reg_a = CorpusRegistry(path)
    reg_b = CorpusRegistry(path)

    reg_b.upsert(_corpus("from_b"))  # B writes first
    reg_a.upsert(_corpus("from_a"))  # A writes with stale in-memory (never saw from_b)

    # Fresh read from disk must contain BOTH.
    final = CorpusRegistry(path)
    names = {c.name for c in final.list()}
    assert names == {"from_a", "from_b"}, f"a writer clobbered the other: {names}"


def test_simultaneous_registry_writers_do_not_lose_an_update(tmp_path):
    path = tmp_path / "corpora.json"
    ctx = multiprocessing.get_context("spawn")
    ready, start = ctx.Queue(), ctx.Event()
    workers = [
        ctx.Process(target=_barrier_upsert, args=(str(path), name, ready, start))
        for name in ("alpha", "beta")
    ]
    for worker in workers:
        worker.start()
    assert {ready.get(timeout=5), ready.get(timeout=5)} == {"alpha", "beta"}
    start.set()
    for worker in workers:
        worker.join(5)
        assert worker.exitcode == 0

    assert {c.name for c in CorpusRegistry(path).list()} == {"alpha", "beta"}


def test_delete_reloads_and_preserves_others(tmp_path):
    path = tmp_path / "corpora.json"
    reg = CorpusRegistry(path)
    reg.upsert(_corpus("keep"))
    reg.upsert(_corpus("drop"))

    # A second handle deletes 'drop'; 'keep' must remain.
    reg2 = CorpusRegistry(path)
    reg2.delete("drop")

    final = CorpusRegistry(path)
    assert {c.name for c in final.list()} == {"keep"}


def test_mixed_model_corpora_coexist_in_registry(tmp_path):
    """Two corpora with different embed models persist side by side."""
    path = tmp_path / "corpora.json"
    reg = CorpusRegistry(path)
    reg.upsert(_corpus("legacy", model="nomic-embed-text", dim=768))
    reg.upsert(_corpus("upgraded", model="bge-m3", dim=1024))

    final = CorpusRegistry(path)
    by_name = {c.name: c for c in final.list()}
    assert by_name["legacy"].embed_model == "nomic-embed-text"
    assert by_name["upgraded"].embed_model == "bge-m3"
    assert by_name["upgraded"].dim == 1024


def test_corrupt_registry_raises_instead_of_wiping(tmp_path):
    """A torn/corrupt corpora.json must NOT load as empty — because the next
    upsert would then save that empty registry over the good data, silently
    dropping every corpus. The recurring "corpora vanished" bug. Loading a
    corrupt file raises so the mutation aborts and disk data survives."""
    import pytest
    from vecgrep.backend.store.corpora import CorpusError

    path = tmp_path / "corpora.json"
    reg = CorpusRegistry(path)
    reg.upsert(_corpus("keep_me"))
    assert path.exists()

    # Simulate a torn write: valid-JSON-prefix, truncated mid-object.
    path.write_text('{"keep_me": {"name": "keep_me", "embed_backend": "oll')

    # A fresh handle must refuse to load an empty registry over the file.
    with pytest.raises(CorpusError):
        CorpusRegistry(path)


def test_save_is_atomic_no_partial_file(tmp_path, monkeypatch):
    """_save writes via a temp file + os.replace, so an interrupted serialize
    never leaves a truncated corpora.json. If the replace step blows up, the
    original file is untouched (not half-written)."""
    import os
    path = tmp_path / "corpora.json"
    reg = CorpusRegistry(path)
    reg.upsert(_corpus("first"))
    good = path.read_text()

    # Force os.replace to fail during the next save; the original must survive.
    real_replace = os.replace
    def boom(src, dst):
        raise OSError("simulated crash during replace")
    monkeypatch.setattr(os, "replace", boom)
    try:
        reg.upsert(_corpus("second"))
    except OSError:
        pass
    monkeypatch.setattr(os, "replace", real_replace)

    # File is still the intact pre-crash version (never truncated), and no
    # stray .tmp litter remains.
    assert path.read_text() == good
    leftovers = [q.name for q in tmp_path.iterdir() if q.name.endswith(".tmp")]
    assert not leftovers, f"temp files left behind: {leftovers}"
