"""float16 is pinned when a qdrant collection is created, so every path that
can (re)create one has to read the corpus's datatype back.

Adopting it on a live corpus (docs/STORAGE_RETRIEVAL_2026-08: identical recall
at every k, qdrant -35%, vector p50 312ms -> 100ms) is only safe if it STICKS.
Before this, `datatype` was reachable from the eval harness and nowhere else,
so a migrated corpus would revert to float32 the next time it was reindexed,
restored from backup, or recovered after a crash -- silently, and months later
it just reads as "why is this corpus slow again".
"""
from vecgrep.backend.service import _collection_for


def _dt(svc, name: str) -> str:
    info = svc.store.client.get_collection(_collection_for(name))
    return str(info.config.params.vectors.datatype).lower()


def test_new_corpus_defaults_to_float32(svc, make_doc):
    svc.index(str(make_doc("a.md", "hello world")), "c1")
    assert svc.registry.get("c1").datatype == "float32"


def test_migrate_converts_the_collection_and_persists_the_choice(svc, make_doc):
    svc.index(str(make_doc("a.md", "alpha beta gamma")), "c1")
    before = svc.store.client.count(_collection_for("c1"), exact=True).count
    assert before > 0

    svc.migrate_datatype("c1", "float16")

    assert svc.registry.get("c1").datatype == "float16"
    assert "float16" in _dt(svc, "c1")
    assert svc.store.client.count(_collection_for("c1"), exact=True).count == before


def test_a_reindex_after_migrating_keeps_float16(svc, make_doc):
    """The whole point. An index that recreates the collection must not
    quietly hand the corpus back to float32."""
    svc.index(str(make_doc("a.md", "alpha beta gamma")), "c1")
    svc.migrate_datatype("c1", "float16")
    svc.index(str(make_doc("b.md", "delta epsilon")), "c1")
    assert svc.registry.get("c1").datatype == "float16"
    assert "float16" in _dt(svc, "c1")


def test_migrating_to_the_current_datatype_is_a_no_op(svc, make_doc):
    svc.index(str(make_doc("a.md", "alpha")), "c1")
    n = svc.store.client.count(_collection_for("c1"), exact=True).count
    svc.migrate_datatype("c1", "float32")
    assert svc.store.client.count(_collection_for("c1"), exact=True).count == n


def test_an_unknown_datatype_is_refused(svc, make_doc):
    import pytest

    svc.index(str(make_doc("a.md", "alpha")), "c1")
    with pytest.raises(ValueError):
        svc.migrate_datatype("c1", "bfloat16")
    assert svc.registry.get("c1").datatype == "float32"
