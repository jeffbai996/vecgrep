"""Directory indexing with an --include glob.

Real bug it fixes: transcripts live as both `.jsonl` (raw) and `.md` (clean
shadow) side by side; indexing the directory grabbed BOTH, double-indexing every
conversation and polluting results. `include="*.md"` restricts the walk so only
the clean shadows are indexed.
"""
from __future__ import annotations


def test_include_glob_restricts_directory_walk(svc, tmp_path):
    d = tmp_path / "transcripts"
    d.mkdir()
    # Same conversation, two formats side by side.
    (d / "2026-05-01.md").write_text("The committee held rates steady through Q2.")
    (d / "2026-05-01.jsonl").write_text(
        '{"author": "fed", "content": "The committee held rates steady through Q2."}'
    )
    (d / "2026-05-02.md").write_text("Labor data softened in the spring.")
    (d / "2026-05-02.jsonl").write_text(
        '{"author": "fed", "content": "Labor data softened in the spring."}'
    )

    docs, chunks, _ = svc.index(str(d), "md_only", include="*.md")
    assert docs == 2, f"expected only the 2 .md files, indexed {docs}"

    # Every indexed source must be a .md — no .jsonl leaked in.
    sources = svc.registry.get("md_only").sources
    assert all(s.endswith(".md") for s in sources), sources
    assert not any(s.endswith(".jsonl") for s in sources)


def test_no_include_indexes_everything(svc, tmp_path):
    d = tmp_path / "mixed"
    d.mkdir()
    (d / "a.md").write_text("markdown content here.")
    (d / "b.jsonl").write_text('{"author": "x", "content": "jsonl content here."}')

    docs, _, _ = svc.index(str(d), "all", include=None)
    assert docs == 2  # both formats indexed when no filter
