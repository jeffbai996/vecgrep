"""A deleted source must vanish from BOTH backends, or not at all.

An ingest pipeline that deletes a document and unlinks its file leaves
vecgrep holding chunks for a document that no longer exists: the vectors
and the BM25 entries survive, so the dead document keeps being returned as
a live answer and outranks the real one.

Half-purging is worse than either extreme. With the vectors dropped but the
keyword index intact, the same dead document still comes back at a plausible
mid rank, which reads as a real result rather than an obvious bug.
"""
from __future__ import annotations

from pathlib import Path

from vecgrep.backend.service import VecgrepService


def _index(svc: VecgrepService, path: Path, corpus: str) -> None:
    svc.index(str(path), corpus)


def test_diagnose_reports_a_source_whose_file_is_gone(svc, make_doc):
    live = make_doc("live.md", "the fund GPs are Alice and Bob")
    dead = make_doc("dead.md", "the fund GPs are Carol and Dave")
    _index(svc, live, "fixture")
    _index(svc, dead, "fixture")

    dead.unlink()

    issues = svc.diagnose(corpora={"fixture"})
    orphans = [i for i in issues if i["kind"] == "orphaned_source"]
    assert len(orphans) == 1, f"expected the deleted source to be reported, got {issues}"
    assert orphans[0]["source_id"] == str(dead)
    assert orphans[0]["fixable"] is True


def test_reconcile_purges_the_orphan_from_both_backends(svc, make_doc):
    live = make_doc("live.md", "the fund GPs are Alice and Bob")
    dead = make_doc("dead.md", "the fund GPs are Carol and Dave")
    _index(svc, live, "fixture")
    _index(svc, dead, "fixture")
    dead.unlink()

    svc.reconcile(corpora={"fixture"})

    # Vector side: no points left for the dead source.
    collection = f"vecgrep__fixture"
    assert svc.store.source_counts(collection).get(str(dead), 0) == 0

    # BM25 side: this is the half-purge trap. A keyword query that the dead
    # doc would answer must not return it.
    hits = svc.bm25.search("fixture", "Carol Dave fund GPs", 10)
    dead_hits = [h for h in hits if h[2].get("source_id") == str(dead)]
    assert not dead_hits, f"BM25 still serves the deleted source: {dead_hits}"

    # The living source is untouched — a purge must not be a corpus wipe.
    assert svc.store.source_counts(collection).get(str(live), 0) > 0


def test_reconcile_leaves_a_present_source_alone(svc, make_doc):
    keep = make_doc("keep.md", "the fund GPs are Alice and Bob")
    _index(svc, keep, "fixture")

    svc.reconcile(corpora={"fixture"})

    collection = f"vecgrep__fixture"
    assert svc.store.source_counts(collection).get(str(keep), 0) > 0
    corpus = svc.registry.get("fixture")
    assert str(keep) in corpus.sources


def test_url_sources_are_never_treated_as_orphans(svc, make_doc):
    """A URL can't be stat'd; treating an unreachable one as deleted would
    purge real data on a network blip."""
    doc = make_doc("page.md", "the fund GPs are Alice and Bob")
    _index(svc, doc, "fixture")
    corpus = svc.registry.get("fixture")
    corpus.sources.append("https://example.com/gone")
    svc.registry.upsert(corpus)

    issues = svc.diagnose(corpora={"fixture"})
    orphans = [i for i in issues if i["kind"] == "orphaned_source"]
    assert not orphans, f"a URL source must never be reported as orphaned: {orphans}"
