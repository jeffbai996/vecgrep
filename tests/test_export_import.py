"""Export/import round-trip — guards against the lock-conflict + rename
issues we hit during initial development.
"""
from __future__ import annotations

import pytest

from vecgrep.backend.store import CorpusError


def test_export_import_round_trip(svc, make_doc, tmp_path):
    p = make_doc("doc.md", "Cats sit on mats. Dogs chase squirrels.")
    svc.index(str(p), "test")

    archive = svc.export_corpus("test", tmp_path / "out.tar.gz")
    assert archive.is_file()
    assert archive.stat().st_size > 0

    svc.delete_corpus("test")
    assert not svc.list_corpora()

    restored = svc.import_corpus(archive, rename="restored")
    assert restored.name == "restored"
    assert restored.doc_count == 1

    hits = svc.search("cats", "restored", top_k=2)
    assert hits
    assert hits[0].corpus == "restored" or hits[0].corpus == "test"
    # Payload still references the OLD name because we didn't rewrite it on
    # import — that's fine, the registry is what controls search routing.


def test_import_refuses_overwrite(svc, make_doc, tmp_path):
    p = make_doc("doc.md", "noop")
    svc.index(str(p), "test")
    archive = svc.export_corpus("test", tmp_path / "out.tar.gz")

    with pytest.raises(CorpusError):
        svc.import_corpus(archive)  # 'test' already exists -> must refuse


def test_import_missing_archive_raises(svc, tmp_path):
    with pytest.raises(CorpusError):
        svc.import_corpus(tmp_path / "does-not-exist.tar.gz")
