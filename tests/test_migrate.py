"""Migration regression tests.

The big one: we initially renamed the qdrant collection directory, which
corrupted Qdrant's in-memory registry. Fix: scroll points from temp
collection, upsert into final collection. This test ensures we never
regress to the rename approach.
"""
from __future__ import annotations

from vecgrep.backend.embed.base import EmbedBackend
from vecgrep.backend.embed.cache import CachedBackend


class _M1(EmbedBackend):
    name = "stub"
    model = "m1"
    dim = 8

    def embed(self, texts):
        return [[(len(t) % 7 + i) / 10 for i in range(self.dim)] for t in texts]


class _M2(EmbedBackend):
    name = "ollama"
    model = "m2"
    dim = 8

    def embed(self, texts):
        return [[(len(t) % 5 + i) / 10 for i in range(self.dim)] for t in texts]


def test_migrate_swaps_backend_and_keeps_data(svc, make_doc, monkeypatch):
    p = make_doc("doc.md", "Cats and dogs and fish. Three sentences here.")
    svc.index(str(p), "test")
    pre_hits = svc.search("cats", "test", top_k=5)
    assert pre_hits

    # Patch the factory so migrate() resolves to M2 regardless of env.
    import vecgrep.backend.service as s_mod

    monkeypatch.setattr(s_mod, "get_embed_backend", lambda settings, prefer=None: _M2())

    out = svc.migrate_corpus("test", to_backend="ollama", to_model="m2")
    assert out.embed_backend == "ollama"
    assert out.embed_model == "m2"
    assert out.dim == 8

    # Force the search backend to M2 (cache cleared because the old M1 entry
    # is now stale for this corpus).
    svc._backend_cache.clear()
    svc._backend_cache["auto"] = CachedBackend(_M2(), svc._embed_cache)
    svc._backend_cache["ollama"] = svc._backend_cache["auto"]

    post_hits = svc.search("cats", "test", top_k=5)
    assert post_hits, "search after migrate should still return hits"
    # Payload's `corpus` field must be the final name, not the temp.
    for h in post_hits:
        assert h.corpus == "test"


def test_migrate_no_op_raises(svc, make_doc):
    p = make_doc("doc.md", "noop test content here")
    svc.index(str(p), "test")
    import pytest
    from vecgrep.backend.store import CorpusError

    # Asking to migrate to the same backend+model the corpus is already on
    # should refuse rather than silently churn.
    corpus = svc.list_corpora()[0]
    with pytest.raises(CorpusError):
        svc.migrate_corpus(
            "test",
            to_backend=corpus.embed_backend,
            to_model=corpus.embed_model,
        )
