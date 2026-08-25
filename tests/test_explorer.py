"""Source-level corpus explorer contracts.

The explorer navigates documents, not chunks.  Every indexed source has one
deterministic home and large corpora are paged at the document boundary.
"""
from __future__ import annotations

import os
from pathlib import Path

from vecgrep.backend.explorer_store import ExplorerStore


def _index(svc, path: Path, corpus: str = "library") -> None:
    svc.index(str(path), corpus)


def test_source_catalog_store_survives_a_process_restart(tmp_path) -> None:
    path = tmp_path / "explorer.db"
    generation = (123.5, 1, 2)
    record = {
        "source_id": "/library/launch.md",
        "metadata": {"source_kind": "memory", "tags": ["ops"]},
        "doc_timestamp": 123.0,
        "chunk_count": 2,
    }

    first = ExplorerStore(path)
    first.replace("library", [record], generation)
    first.close()

    reopened = ExplorerStore(path)
    assert reopened.generation("library") == generation
    assert reopened.records("library") == [record]
    reopened.close()


def test_source_catalog_completeness_tracks_its_own_committed_rows(
    tmp_path
) -> None:
    store = ExplorerStore(tmp_path / "explorer.db")
    # Registry counts can drift from the chunk index on older corpora. The
    # explorer mirrors searchable sources, not stale registry bookkeeping.
    generation = (123.5, 5, 2)
    store.replace(
        "library",
        [
            {
                "source_id": "/library/visible.md",
                "metadata": {},
                "doc_timestamp": None,
                "chunk_count": 2,
            }
        ],
        generation,
    )
    assert store.generation("library") == generation
    store.close()


def test_normal_index_populates_explorer_without_reloading_bm25(
    svc, tmp_path, monkeypatch
) -> None:
    source = tmp_path / "guide.md"
    source.write_text("# Guide\n\nUseful text.", encoding="utf-8")
    _index(svc, source)
    svc._explorer_cache.clear()

    def fail_load(_corpus: str):
        raise AssertionError("explorer reloaded the chunk-level BM25 sidecar")

    monkeypatch.setattr(svc.bm25, "_load", fail_load)
    listing = svc.explore("library")
    assert listing["documents"][0]["name"] == "guide.md"


def test_legacy_catalog_backfills_once_then_uses_the_compact_store(
    svc, tmp_path, monkeypatch
) -> None:
    source = tmp_path / "legacy.md"
    source.write_text("# Legacy\n\nExisting indexed content.", encoding="utf-8")
    _index(svc, source)
    svc.explorer_store.drop("library")
    svc._explorer_cache.clear()

    original_load = svc.bm25._load
    calls = 0

    def counted_load(corpus: str):
        nonlocal calls
        calls += 1
        return original_load(corpus)

    monkeypatch.setattr(svc.bm25, "_load", counted_load)
    assert svc.explore("library")["documents"][0]["name"] == "legacy.md"
    assert calls == 1

    svc._explorer_cache.clear()

    def fail_load(_corpus: str):
        raise AssertionError("a completed legacy backfill ran twice")

    monkeypatch.setattr(svc.bm25, "_load", fail_load)
    assert svc.explore("library")["documents"][0]["name"] == "legacy.md"


def test_incremental_index_cannot_mark_a_partial_legacy_catalog_complete(
    svc, tmp_path
) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("# First\n\nExisting content.", encoding="utf-8")
    second.write_text("# Second\n\nNew content.", encoding="utf-8")
    _index(svc, first)

    # Simulate an upgrade where canonical search data predates explorer.db.
    svc.explorer_store.drop("library")
    _index(svc, second)
    assert svc.explorer_store.generation("library") is None

    svc._explorer_cache.clear()
    listing = svc.explore("library")
    assert [document["name"] for document in listing["documents"]] == [
        "first.md",
        "second.md",
    ]


def test_source_catalog_tracks_reindex_and_delete(svc, tmp_path) -> None:
    source = tmp_path / "record.md"
    source.write_text(
        "---\nsource_kind: memory\ntitle: First title\ntags: [ops]\n---\n\nShort.",
        encoding="utf-8",
    )
    svc.index(str(source), "library", chunker_name="fixed_token")
    before = svc.explore("library", path=["Memories"])["documents"][0]
    assert before["name"] == "First title"
    assert before["tags"] == ["ops"]

    source.write_text(
        "---\nsource_kind: memory\ntitle: Second title\ntags: [design]\n---\n\n"
        + "expanded content " * 2_000,
        encoding="utf-8",
    )
    svc.index(str(source), "library", chunker_name="fixed_token", force=True)
    svc._explorer_cache.clear()
    after = svc.explore("library", path=["Memories"])["documents"][0]
    assert after["name"] == "Second title"
    assert after["tags"] == ["design"]
    assert after["chunk_count"] > before["chunk_count"]

    svc.delete_source("library", str(source))
    svc._explorer_cache.clear()
    assert svc.explore("library")["recent_documents"] == []


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
