"""Export/import round-trip — guards against the lock-conflict + rename
issues we hit during initial development.
"""
from __future__ import annotations

import json
import os
import pickle
import tarfile
from io import BytesIO
from pathlib import Path

import pytest

from vecgrep.backend.store import CorpusError


def _write_unpickle_marker(path: str) -> None:
    Path(path).write_text("pickle-executed")


class _WriteOnUnpickle:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __reduce__(self):
        return _write_unpickle_marker, (str(self.path),)


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


@pytest.mark.parametrize(
    ("unsafe_name", "symlink_name"),
    [
        ("../../../../../escaped", "vecgrep__.."),
        ("__migrate__../../../../../escaped", "vecgrep____migrate__.."),
    ],
)
def test_import_rejects_archive_name_before_copying_outside_home(
    svc, vg_home, tmp_path, unsafe_name, symlink_name
):
    source = tmp_path / "malicious-archive"
    collection_root = source / "qdrant" / "collection"
    (collection_root / "deep" / "a" / "b" / "c" / "d" / "e").mkdir(
        parents=True
    )
    escaped_source = collection_root / "deep" / "a" / "escaped"
    escaped_source.mkdir(parents=True)
    (escaped_source / "owned.txt").write_text("archive-controlled")
    try:
        os.symlink("deep/a/b/c/d/e", collection_root / symlink_name)
    except OSError as exc:  # pragma: no cover - platform capability
        pytest.skip(f"symlink creation unavailable: {exc}")

    (source / "corpus.json").write_text(json.dumps({
        "name": unsafe_name,
        "embed_backend": "stub",
        "embed_model": "stub-1",
        "dim": 16,
    }))
    archive = tmp_path / "malicious.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for child in source.iterdir():
            bundle.add(child, arcname=child.name)

    with pytest.raises(CorpusError, match="Invalid corpus name"):
        svc.import_corpus(archive)

    assert not (vg_home.parent / "escaped").exists()
    svc.store.client.get_collections()


@pytest.mark.parametrize(
    "unsafe_name",
    ["../../../../../escaped", "__migrate__../../../../../escaped"],
)
def test_import_rejects_unsafe_rename_before_copying(
    svc, make_doc, vg_home, tmp_path, unsafe_name
):
    svc.index(str(make_doc("doc.md", "alpha beta gamma")), "notes")
    archive = svc.export_corpus("notes", tmp_path / "notes.tar.gz")
    svc.delete_corpus("notes")

    with pytest.raises(CorpusError, match="Invalid corpus name"):
        svc.import_corpus(archive, rename=unsafe_name)

    assert not (vg_home.parent / "escaped").exists()
    svc.store.client.get_collections()


def test_import_rejects_tar_member_outside_staging(svc, tmp_path):
    archive = tmp_path / "traversal.tar.gz"
    payload = b"archive-controlled"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("../escaped.txt")
        member.size = len(payload)
        bundle.addfile(member, BytesIO(payload))

    with pytest.raises(tarfile.FilterError):
        svc.import_corpus(archive)

    assert not (tmp_path / "escaped.txt").exists()
    svc.store.client.get_collections()


def test_import_never_loads_archive_bm25_pickle(svc, tmp_path):
    marker = tmp_path / "pickle-marker"
    source = tmp_path / "pickle-archive"
    source.mkdir()
    (source / "corpus.json").write_text(json.dumps({
        "name": "restored",
        "embed_backend": "stub",
        "embed_model": "stub-1",
        "dim": 16,
    }))
    (source / "bm25.pkl").write_bytes(pickle.dumps(_WriteOnUnpickle(marker)))
    archive = tmp_path / "pickle.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for child in source.iterdir():
            bundle.add(child, arcname=child.name)

    restored = svc.import_corpus(archive)

    assert restored.name == "restored"
    assert not marker.exists()
