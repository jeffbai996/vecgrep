"""Opt-in-only docs: a doc tagged `opt-in-only` or `quarantine` stays out of
every read that does not name it.

Invariants under test:
  - unscoped and corpus-scoped search both drop it (naming the corpus is not
    naming the doc), as do related(), browse() and summarize_corpus()
  - a positive filter that names the doc brings it back: meta.title= or
    meta.id= with its exact value, or a source glob containing its file stem
  - a broad glob, a negated filter or an unrelated filter is not an opt-in
  - untagged docs are unaffected
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from vecgrep.backend.privacy import is_opt_in_only, opt_in_hidden


def _doc(path: Path, body: str, tags: str = "", title: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = "status: active\n"
    if title:
        fm += f"title: {title}\nid: {path.stem}\n"
    if tags:
        fm += f"tags: [{tags}]\n"
    path.write_text(f"---\n{fm}---\n\n{body}\n")
    return path


@pytest.fixture
def seeded(svc, tmp_path):
    d = tmp_path / "docs"
    svc.index(str(_doc(d / "private-doc.md", "walrus household boundaries plan",
                       tags="family, quarantine, opt-in-only",
                       title="private-walrus-plan")), "ext")
    svc.index(str(_doc(d / "public-doc.md", "walrus household grocery plan",
                       tags="family")), "ext")
    return svc


def _names(hits) -> set[str]:
    return {os.path.basename(h.source_id) for h in hits}


# ── the predicate ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tags", [["quarantine"], ["opt-in-only"],
                                  ["family", "Opt-In-Only"], "family, quarantine"])
def test_either_tag_marks_a_doc_opt_in_only(tags):
    assert is_opt_in_only({"tags": tags})


@pytest.mark.parametrize("meta", [{}, {"tags": []}, {"tags": ["family"]},
                                  {"tags": None}, {"tags": "family"}])
def test_docs_without_either_tag_are_not_opt_in_only(meta):
    assert not is_opt_in_only(meta)


def test_a_source_glob_must_mention_the_doc_stem_to_opt_in():
    meta = {"tags": ["opt-in-only"]}
    sid = "/w/external/external-abc123.md"
    assert opt_in_hidden(meta, sid, ["source:*"])
    assert opt_in_hidden(meta, sid, ["source:*/external/*"])
    assert not opt_in_hidden(meta, sid, ["source:*external-abc123*"])
    assert not opt_in_hidden(meta, sid, ["source_path:*external-abc123.md"])


def test_a_negated_filter_never_opts_in():
    meta = {"tags": ["opt-in-only"], "title": "t"}
    assert opt_in_hidden(meta, "/w/x.md", ["-meta.title=t"])
    assert opt_in_hidden(meta, "/w/x.md", ["-source:*x*"])


# ── search ───────────────────────────────────────────────────────────────────

def test_unscoped_search_drops_an_opt_in_only_doc(seeded):
    assert _names(seeded.search("walrus household plan", top_k=10)) == {"public-doc.md"}


def test_naming_the_corpus_is_not_naming_the_doc(seeded):
    hits = seeded.search("walrus household plan", corpus_name="ext", top_k=10)
    assert _names(hits) == {"public-doc.md"}


@pytest.mark.parametrize("flt", ["meta.title=private-walrus-plan",
                                 "meta.id=private-doc",
                                 "source:*private-doc*"])
def test_a_filter_naming_the_doc_returns_it(seeded, flt):
    hits = seeded.search("walrus household plan", filters=[flt], top_k=10)
    assert "private-doc.md" in _names(hits)


@pytest.mark.parametrize("flt", ["source:*", "meta.status=active", "corpus:ext",
                                 "meta.title=something-else"])
def test_a_filter_that_does_not_name_the_doc_keeps_it_hidden(seeded, flt):
    hits = seeded.search("walrus household plan", filters=[flt], top_k=10)
    assert "private-doc.md" not in _names(hits)


def test_timeline_and_compare_inherit_the_gate(seeded):
    groups = seeded.timeline("walrus household plan", top_k=10)
    assert all("private-doc" not in g["source_id"] for g in groups)
    diff = seeded.compare("walrus household plan", "ext")
    flat = repr(diff)
    assert "private-doc" not in flat


# ── the other readers ────────────────────────────────────────────────────────

def test_related_never_offers_an_opt_in_only_neighbour(seeded):
    seed = next(h for h in seeded.search("walrus grocery", corpus_name="ext", top_k=5)
                if "public-doc" in h.source_id)
    near = seeded.related(seed.chunk_id, "ext", top_k=10)
    assert all("private-doc" not in r.source_id for r in near)


def test_browse_by_date_skips_it_and_browse_by_its_path_reads_it(seeded):
    by_date = seeded.browse("ext", since="2000-01-01")
    assert all("private-doc" not in g["source_id"] for g in by_date)
    by_path = seeded.browse("ext", source_path="*private-doc*")
    assert [os.path.basename(g["source_id"]) for g in by_path] == ["private-doc.md"]


def test_corpus_summary_leaves_it_out(seeded):
    summary = seeded.summarize_corpus("ext", sample=50)
    assert all("private-doc" not in s["source_id"] for s in summary["sample_chunks"])
    assert all("private-doc" not in sid for sid, _ in summary["top_sources"])
