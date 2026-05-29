"""Registry write-race + mixed-model resolution (latent bug fix).

Two coupled bugs this guards against:
1. A long-lived process (the server) clobbering another writer's (the CLI's)
   registry changes with a stale in-memory copy. upsert/delete now reload from
   disk first, so concurrent writers' other corpora survive.
2. The engine being unable to serve corpora embedded with different models —
   query-embed backend now resolves per-corpus, not from one global env model.
"""
from __future__ import annotations

from vecgrep.backend.store.corpora import Corpus, CorpusRegistry


def _corpus(name: str, model: str = "nomic-embed-text", dim: int = 768) -> Corpus:
    return Corpus(name=name, embed_backend="ollama", embed_model=model, dim=dim)


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
