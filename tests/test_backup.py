from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

from vecgrep.backend.backup import BackupError, BackupManager, BackupScheduler, QdrantSnapshots
from vecgrep.backend.config import Settings


def test_backup_manifest_checksums_and_exclusions(vg_home: Path) -> None:
    vg_home.mkdir(parents=True)
    (vg_home / "config.json").write_text(json.dumps({"default_top_k": 7, "api_token": "nope"}))
    (vg_home / "corpora.json").write_text("{}")
    (vg_home / "aliases.json").write_text("{}")
    (vg_home / "embed_cache.db").write_bytes(b"large cache")
    (vg_home / "bm25").mkdir()
    (vg_home / "bm25" / "notes.pkl").write_bytes(b"unsafe pickle")
    (vg_home / "write").mkdir()
    (vg_home / "write" / "note.md").write_text("hello")

    archive = BackupManager(Settings(home=vg_home)).create(origin="manual")
    result = BackupManager(Settings(home=vg_home)).verify(archive)

    assert result["valid"] is True
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        manifest = json.loads(bundle.read("manifest.json"))
        stored_config = json.loads(bundle.read("config.json"))
    assert manifest["schema_version"] == 1
    assert manifest["origin"] == "manual"
    assert "embed_cache.db" not in names
    assert not any(name.startswith("bm25/") for name in names)
    assert "api_token" not in stored_config
    assert "write/note.md" in names


def test_verify_detects_tampering(vg_home: Path) -> None:
    archive = BackupManager(Settings(home=vg_home)).create()
    rewritten = vg_home / "tampered.vgbak"
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(rewritten, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name == "config.json":
                data += b"tampered"
            target.writestr(name, data)
    with pytest.raises(BackupError, match="checksum"):
        BackupManager(Settings(home=vg_home)).verify(rewritten)


def test_verify_rejects_path_traversal(vg_home: Path) -> None:
    archive = vg_home / "unsafe.vgbak"
    vg_home.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape", "bad")
        bundle.writestr("manifest.json", json.dumps({"schema_version": 1, "checksums": {}}))
    with pytest.raises(BackupError, match="unsafe"):
        BackupManager(Settings(home=vg_home)).verify(archive)


def test_restore_requires_exact_backup_id(vg_home: Path) -> None:
    manager = BackupManager(Settings(home=vg_home))
    archive = manager.create()
    backup_id = manager.read_manifest(archive)["backup_id"]
    with pytest.raises(BackupError, match="confirmation"):
        manager.restore(archive, confirm="wrong")
    assert backup_id != "wrong"


def test_retention_prunes_only_scheduled_backups(vg_home: Path, monkeypatch) -> None:
    manager = BackupManager(Settings(home=vg_home))
    manager.create(origin="manual", backup_id="manual-one")
    for value in ("scheduled-one", "scheduled-two", "scheduled-three"):
        manager.create(origin="scheduled", backup_id=value)
    removed = manager.prune_scheduled(retain=2)
    assert len(removed) == 1
    remaining = {item["backup_id"] for item in manager.list()}
    assert "manual-one" in remaining
    assert len(remaining & {"scheduled-one", "scheduled-two", "scheduled-three"}) == 2


def test_scheduler_disabled_and_deduplicates_same_slot(vg_home: Path) -> None:
    manager = BackupManager(Settings(home=vg_home))
    scheduler = BackupScheduler(manager, enabled=False, frequency="daily", local_time="03:00")
    now = datetime(2026, 7, 9, 3, 5)
    assert scheduler.run_if_due(now) is None

    scheduler.enabled = True
    first = scheduler.run_if_due(now)
    second = scheduler.run_if_due(now)
    assert first is not None
    assert second is None


def test_embedded_backup_restore_rebuilds_searchable_state(svc, make_doc, vg_home: Path) -> None:
    source = make_doc("notes.txt", "alpha recovery marker")
    svc.index(str(source), "notes")
    svc.set_corpus_context(
        "notes", "Durable notes", ["facts"], ["live state"]
    )
    svc.store.client.close()

    manager = BackupManager(Settings(home=vg_home))
    archive = manager.create()
    backup_id = manager.read_manifest(archive)["backup_id"]

    destructive = manager.service_factory()
    destructive.delete_corpus("notes")
    destructive.store.client.close()
    result = manager.restore(archive, confirm=backup_id)

    restored = manager.service_factory()
    assert restored.store.count("vecgrep__notes") > 0
    assert restored.bm25.search("notes", "alpha", top_k=5)
    metadata = restored.registry.get("notes")
    assert metadata.description == "Durable notes"
    assert metadata.use_for == ["facts"]
    assert metadata.avoid_for == ["live state"]
    assert result["issues"] == []
    restored.store.client.close()


def test_server_snapshot_transport_downloads_and_uploads(vg_home: Path, monkeypatch) -> None:
    class Description:
        name = "snapshot-1.snapshot"

    class Client:
        def create_snapshot(self, collection, wait=True):
            assert collection == "vecgrep__notes" and wait is True
            return Description()
        def delete_snapshot(self, collection, name, wait=True):
            assert (collection, name, wait) == ("vecgrep__notes", "snapshot-1.snapshot", True)

    class Store:
        client = Client()
        def drop_collection(self, collection):
            assert collection == "vecgrep__notes"

    class Service:
        store = Store()

    class Download:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def raise_for_status(self): return None
        def iter_bytes(self): yield b"native snapshot"

    class Upload:
        def raise_for_status(self): return None

    uploaded = []
    monkeypatch.setattr("vecgrep.backend.backup.httpx.stream", lambda *a, **k: Download())
    monkeypatch.setattr("vecgrep.backend.backup.httpx.post", lambda *a, **k: (uploaded.append((a, k)) or Upload()))
    settings = Settings(home=vg_home, qdrant_url="http://localhost:6333")
    snapshots = QdrantSnapshots(settings, Service())
    vg_home.mkdir(parents=True)

    snapshot_format, path = snapshots.create("notes", vg_home)
    snapshots.restore("notes", path, snapshot_format, 16)

    assert path.read_bytes() == b"native snapshot"
    assert snapshot_format == "qdrant-native"
    assert uploaded and uploaded[0][0][0].endswith("/vecgrep__notes/snapshots/upload")


def test_server_snapshot_cleanup_failure_keeps_download(
    vg_home: Path, monkeypatch, caplog
) -> None:
    class Description:
        name = "snapshot-1.snapshot"

    class Client:
        def create_snapshot(self, collection, wait=True):
            return Description()

        def delete_snapshot(self, collection, name, wait=True):
            raise RuntimeError("cleanup transport dropped")

    class Store:
        client = Client()

    class Service:
        store = Store()

    class Download:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def raise_for_status(self): return None
        def iter_bytes(self): yield b"verified native snapshot"

    monkeypatch.setattr("vecgrep.backend.backup.httpx.stream", lambda *a, **k: Download())
    settings = Settings(home=vg_home, qdrant_url="http://localhost:6333")
    snapshots = QdrantSnapshots(settings, Service())
    vg_home.mkdir(parents=True)

    snapshot_format, path = snapshots.create("notes", vg_home)

    assert snapshot_format == "qdrant-native"
    assert path.read_bytes() == b"verified native snapshot"
    assert "cleanup deferred" in caplog.text


def test_restore_failure_runs_automatic_rollback(vg_home: Path, monkeypatch) -> None:
    manager = BackupManager(Settings(home=vg_home))
    archive = manager.create()
    backup_id = manager.read_manifest(archive)["backup_id"]
    calls = []

    def restore_once(path, manifest):
        calls.append(manifest["origin"])
        if len(calls) == 1:
            raise RuntimeError("injected failure")
        return {"restored": manifest["backup_id"], "issues": []}

    monkeypatch.setattr(manager, "_restore_verified", restore_once)
    with pytest.raises(BackupError, match="rolled back"):
        manager.restore(archive, confirm=backup_id)
    assert calls == ["manual", "pre-restore"]
