"""Source-level corpus explorer contracts.

The explorer navigates documents, not chunks.  Every indexed source has one
deterministic home and large corpora are paged at the document boundary.
"""
from __future__ import annotations

import os
from pathlib import Path


def _index(svc, path: Path, corpus: str = "library") -> None:
    svc.index(str(path), corpus)


def test_explorer_organizes_channel_corpora_by_channel_and_month(svc, tmp_path) -> None:
    first = tmp_path / "transcripts" / "general" / "2026-01-15.md"
    second = tmp_path / "transcripts" / "general" / "2026-02-03.md"
    first.parent.mkdir(parents=True)
    first.write_text(
        "---\nchannel: general\ndate: 2026-01-15\n---\n\nalice: first day",
        encoding="utf-8",
    )
    second.write_text(
        "---\nchannel: general\ndate: 2026-02-03\n---\n\nbob: second day",
        encoding="utf-8",
    )
    _index(svc, first)
    _index(svc, second)

    root = svc.explore("library")
    assert root["scheme"] == "channels"
    assert [(folder["name"], folder["document_count"]) for folder in root["folders"]] == [
        ("general", 2)
    ]

    year = svc.explore("library", path=["general"])
    assert [folder["name"] for folder in year["folders"]] == ["2026"]
    months = svc.explore("library", path=["general", "2026"])
    assert [folder["name"] for folder in months["folders"]] == [
        "01 January",
        "02 February",
    ]
    january = svc.explore(
        "library", path=["general", "2026", "01 January"]
    )
    assert january["total_documents"] == 1
    assert january["documents"][0]["name"] == "2026-01-15.md"


def test_explorer_groups_typed_notes_and_exposes_tag_facets(svc, tmp_path) -> None:
    note = tmp_path / "records" / "memory-001.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        "---\nsource_kind: memory\ntitle: Launch notes\ntags: [ops, docs]\n"
        "created_at: 2026-01-10T12:00:00+00:00\n---\n\nThe useful bit.",
        encoding="utf-8",
    )
    _index(svc, note)

    root = svc.explore("library")
    assert root["scheme"] == "kinds"
    assert root["folders"][0]["name"] == "Memories"
    assert root["facets"]["tags"] == {"docs": 1, "ops": 1}
    listing = svc.explore("library", path=["Memories"])
    assert listing["documents"][0]["name"] == "Launch notes"
    assert listing["documents"][0]["kind"] == "memory"


def test_explorer_organizes_flat_exported_records_by_filename_kind(svc, tmp_path) -> None:
    records = tmp_path / "export"
    records.mkdir()
    for name in (
        "memory-001.md",
        "journal-2026-01-10.md",
        "todo-004.md",
        "deep-architecture.md",
    ):
        path = records / name
        path.write_text(f"# {name}\n\nExported record.", encoding="utf-8")
        _index(svc, path)

    root = svc.explore("library")
    assert root["scheme"] == "records"
    assert [folder["name"] for folder in root["folders"]] == [
        "Journals",
        "Memories",
        "References",
        "To-dos",
    ]
    memories = svc.explore("library", path=["Memories"])
    assert memories["documents"][0]["name"] == "memory-001.md"
    assert memories["documents"][0]["kind"] == "memory"


def test_explorer_keeps_repository_trees_relative_to_the_shared_root(svc, tmp_path) -> None:
    source_root = tmp_path / "repo-snapshots"
    alpha = source_root / "alpha" / "src" / "main.py"
    beta = source_root / "beta" / "README.md"
    alpha.parent.mkdir(parents=True)
    beta.parent.mkdir(parents=True)
    alpha.write_text("def main():\n    return 'alpha'\n", encoding="utf-8")
    beta.write_text("# Beta\n\nRepository notes.", encoding="utf-8")
    _index(svc, alpha)
    _index(svc, beta)

    root = svc.explore("library")
    assert root["scheme"] == "files"
    assert [folder["name"] for folder in root["folders"]] == ["alpha", "beta"]
    alpha_root = svc.explore("library", path=["alpha"])
    assert alpha_root["folders"][0]["path"] == ["alpha", "src"]
    src = svc.explore("library", path=["alpha", "src"])
    assert src["documents"][0]["name"] == "main.py"


def test_explorer_pages_sorts_and_filters_documents_without_chunk_duplicates(
    svc, tmp_path
) -> None:
    root = tmp_path / "flat"
    root.mkdir()
    for index, name in enumerate(("alpha.md", "beta.md", "gamma.md"), start=1):
        path = root / name
        path.write_text(f"# {name}\n\nDocument {index}.", encoding="utf-8")
        os.utime(path, (index, index))
        _index(svc, path)

    first = svc.explore("library", limit=2, sort="name")
    assert first["total_documents"] == 3
    assert first["has_more"] is True
    assert [doc["name"] for doc in first["documents"]] == ["alpha.md", "beta.md"]
    second = svc.explore("library", offset=2, limit=2, sort="name")
    assert [doc["name"] for doc in second["documents"]] == ["gamma.md"]

    filtered = svc.explore("library", query="beta")
    assert filtered["total_documents"] == 1
    assert filtered["documents"][0]["name"] == "beta.md"


def test_explorer_source_reveals_home_and_returns_a_bounded_preview(svc, tmp_path) -> None:
    source = tmp_path / "docs" / "guide.md"
    source.parent.mkdir()
    source.write_text("# Guide\n\n" + "useful text " * 50, encoding="utf-8")
    _index(svc, source)

    listing = svc.explore("library")
    source_id = listing["documents"][0]["source_id"]
    preview = svc.explorer_source("library", source_id, max_chars=40)
    assert preview is not None
    assert preview["parent_path"] == []
    assert preview["name"] == "guide.md"
    assert preview["source_length"] > len(preview["text"])
    assert preview["truncated"] is True
    assert len(preview["text"]) == 40
