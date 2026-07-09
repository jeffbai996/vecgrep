"""Verified whole-instance backups for vecgrep.

Backups contain authoritative state only: Qdrant collections, corpus registry,
non-secret configuration, aliases, and write-through documents. Derived BM25
and embedding caches are deliberately rebuilt or discarded.
"""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import hmac
import json
import os
import shutil
import stat
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator

import httpx

from .. import __version__
from .config import EDITABLE_FIELDS, SECRET_FIELDS, Settings, _atomic_write_json, _load_json
from .service import VecgrepService, _collection_for
from .store import CorpusError


class BackupError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_member(info: zipfile.ZipInfo) -> bool:
    path = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    return (
        bool(info.filename)
        and not path.is_absolute()
        and ".." not in path.parts
        and not stat.S_ISLNK(mode)
    )


class QdrantSnapshots:
    """Create/restore collection snapshots without leaking transport details."""

    def __init__(self, settings: Settings, service: VecgrepService) -> None:
        self.settings = settings
        self.service = service

    def create(self, corpus: str, destination: Path) -> tuple[str, Path]:
        collection = _collection_for(corpus)
        if self.settings.qdrant_url:
            description = self.service.store.client.create_snapshot(collection, wait=True)
            if description is None or not description.name:
                raise BackupError(f"Qdrant did not create a snapshot for {corpus}")
            path = destination / f"{corpus}.snapshot"
            url = (
                f"{self.settings.qdrant_url.rstrip('/')}/collections/"
                f"{collection}/snapshots/{description.name}"
            )
            try:
                with httpx.stream("GET", url, timeout=600.0) as response:
                    response.raise_for_status()
                    with path.open("wb") as output:
                        for chunk in response.iter_bytes():
                            output.write(chunk)
            finally:
                self.service.store.client.delete_snapshot(
                    collection, description.name, wait=True
                )
            return "qdrant-native", path

        path = destination / f"{corpus}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as output:
            offset = None
            while True:
                points, offset = self.service.store.client.scroll(
                    collection_name=collection,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
                for point in points:
                    output.write(json.dumps({
                        "id": str(point.id),
                        "vector": point.vector,
                        "payload": point.payload or {},
                    }, default=str) + "\n")
                if offset is None:
                    break
        return "portable-jsonl", path

    def restore(self, corpus: str, source: Path, snapshot_format: str, dim: int) -> None:
        collection = _collection_for(corpus)
        self.service.store.drop_collection(collection)
        if snapshot_format == "qdrant-native":
            if not self.settings.qdrant_url:
                raise BackupError("A server-mode Qdrant snapshot requires qdrant_url")
            url = (
                f"{self.settings.qdrant_url.rstrip('/')}/collections/"
                f"{collection}/snapshots/upload"
            )
            with source.open("rb") as handle:
                response = httpx.post(
                    url,
                    params={"wait": "true", "priority": "snapshot"},
                    files={"snapshot": (source.name, handle, "application/octet-stream")},
                    timeout=1200.0,
                )
            response.raise_for_status()
            return
        if snapshot_format != "portable-jsonl":
            raise BackupError(f"Unsupported snapshot format: {snapshot_format}")
        self.service.store.ensure_collection(collection, dim)
        ids: list[str] = []
        vectors: list[list[float]] = []
        payloads: list[dict] = []
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            for line in handle:
                point = json.loads(line)
                ids.append(point["id"])
                vectors.append(point["vector"])
                payloads.append(point["payload"])
                if len(ids) >= 64:
                    self.service.store.upsert(collection, ids, vectors, payloads)
                    ids, vectors, payloads = [], [], []
        if ids:
            self.service.store.upsert(collection, ids, vectors, payloads)


class BackupManager:
    SCHEMA_VERSION = 1

    def __init__(
        self,
        settings: Settings,
        *,
        service_factory: Callable[[], VecgrepService] | None = None,
    ) -> None:
        self.settings = settings
        self.backup_dir = (
            Path(settings.backup_destination).expanduser()
            if settings.backup_destination
            else settings.home / "backups"
        )
        self.service_factory = service_factory or (lambda: VecgrepService(settings=settings))

    @contextlib.contextmanager
    def mutation_lock(self) -> Iterator[None]:
        """A simple cross-process lock used by create and restore."""
        self.settings.home.mkdir(parents=True, exist_ok=True)
        path = self.settings.home / "backup.lock"
        deadline = time.monotonic() + 30
        fd = None
        while fd is None:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise BackupError("another backup or restore operation is running")
                time.sleep(0.1)
        try:
            os.write(fd, f"{os.getpid()}\n".encode())
            yield
        finally:
            os.close(fd)
            path.unlink(missing_ok=True)

    def create(
        self,
        destination: Path | None = None,
        *,
        origin: str = "manual",
        backup_id: str | None = None,
    ) -> Path:
        with self.mutation_lock():
            return self._create_unlocked(destination, origin=origin, backup_id=backup_id)

    def _create_unlocked(
        self,
        destination: Path | None,
        *,
        origin: str,
        backup_id: str | None = None,
    ) -> Path:
        if origin not in {"manual", "scheduled", "pre-restore"}:
            raise BackupError(f"Unknown backup origin: {origin}")
        now = datetime.now().astimezone()
        backup_id = backup_id or f"{now:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        destination = (destination or self.backup_dir / f"vecgrep-{backup_id}.vgbak").expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=self.settings.home, prefix=".backup-") as tmp:
            staging = Path(tmp)
            corpora_raw = _load_json(self.settings.corpora_file)
            _atomic_write_json(staging / "corpora.json", corpora_raw)
            raw_config = _load_json(self.settings.config_file)
            safe_config = {key: value for key, value in raw_config.items() if key in EDITABLE_FIELDS}
            _atomic_write_json(staging / "config.json", safe_config)
            if (self.settings.home / "aliases.json").is_file():
                shutil.copy2(self.settings.home / "aliases.json", staging / "aliases.json")
            if (self.settings.home / "write").is_dir():
                shutil.copytree(self.settings.home / "write", staging / "write")

            corpus_entries = []
            service = None
            if corpora_raw:
                service = self.service_factory()
                snapshotter = QdrantSnapshots(self.settings, service)
                snapshot_dir = staging / "snapshots"
                snapshot_dir.mkdir()
                for corpus in service.registry.list():
                    snapshot_format, snapshot_path = snapshotter.create(corpus.name, snapshot_dir)
                    corpus_entries.append({
                        "name": corpus.name,
                        "doc_count": corpus.doc_count,
                        "chunk_count": service.store.count(_collection_for(corpus.name)),
                        "embed_backend": corpus.embed_backend,
                        "embed_model": corpus.embed_model,
                        "dim": corpus.dim,
                        "snapshot": snapshot_path.relative_to(staging).as_posix(),
                        "snapshot_format": snapshot_format,
                    })
            try:
                files = sorted(path for path in staging.rglob("*") if path.is_file())
                checksums = {path.relative_to(staging).as_posix(): _sha256(path) for path in files}
                manifest = {
                    "schema_version": self.SCHEMA_VERSION,
                    "backup_id": backup_id,
                    "created_at": now.isoformat(),
                    "vecgrep_version": __version__,
                    "storage_mode": "server" if self.settings.qdrant_url else "embedded",
                    "origin": origin,
                    "corpora": corpus_entries,
                    "checksums": checksums,
                }
                _atomic_write_json(staging / "manifest.json", manifest)
                temp_archive = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
                with zipfile.ZipFile(temp_archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                    for path in sorted(staging.rglob("*")):
                        if path.is_file():
                            bundle.write(path, path.relative_to(staging).as_posix())
                os.chmod(temp_archive, 0o600)
                os.replace(temp_archive, destination)
            finally:
                if service is not None:
                    try:
                        service.store.client.close()
                    except Exception:
                        pass
        return destination

    def read_manifest(self, archive: Path) -> dict:
        try:
            with zipfile.ZipFile(archive) as bundle:
                if any(not _safe_member(info) for info in bundle.infolist()):
                    raise BackupError("archive contains an unsafe path or link")
                return json.loads(bundle.read("manifest.json"))
        except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
            raise BackupError(f"Invalid backup archive: {exc}") from exc

    def verify(self, archive: Path) -> dict:
        archive = archive.expanduser().resolve()
        manifest = self.read_manifest(archive)
        if manifest.get("schema_version") != self.SCHEMA_VERSION:
            raise BackupError(f"Unsupported backup schema: {manifest.get('schema_version')}")
        with zipfile.ZipFile(archive) as bundle:
            for name, expected in manifest.get("checksums", {}).items():
                try:
                    actual = hashlib.sha256(bundle.read(name)).hexdigest()
                except KeyError as exc:
                    raise BackupError(f"backup is missing checksummed file: {name}") from exc
                if not hmac.compare_digest(actual, expected):
                    raise BackupError(f"checksum mismatch: {name}")
        return {"valid": True, "path": str(archive), **manifest}

    def list(self) -> list[dict]:
        if not self.backup_dir.is_dir():
            return []
        items = []
        for path in self.backup_dir.glob("*.vgbak"):
            try:
                items.append({**self.read_manifest(path), "path": str(path), "size": path.stat().st_size})
            except BackupError:
                items.append({"path": str(path), "invalid": True, "created_at": ""})
        return sorted(items, key=lambda item: item.get("created_at", ""), reverse=True)

    def prune_scheduled(self, retain: int) -> list[Path]:
        if retain < 1:
            raise BackupError("retention count must be positive")
        scheduled = [item for item in self.list() if item.get("origin") == "scheduled"]
        removed = []
        for item in scheduled[retain:]:
            path = Path(item["path"])
            path.unlink()
            removed.append(path)
        return removed

    def restore(self, archive: Path, *, confirm: str) -> dict:
        archive = archive.expanduser().resolve()
        verified = self.verify(archive)
        if confirm != verified["backup_id"]:
            raise BackupError("restore confirmation must exactly match the backup ID")
        with self.mutation_lock():
            safety = self._create_unlocked(None, origin="pre-restore")
            try:
                result = self._restore_verified(archive, verified)
            except Exception as exc:
                rollback = self.verify(safety)
                try:
                    self._restore_verified(safety, rollback)
                except Exception as rollback_exc:
                    raise BackupError(
                        f"restore failed ({exc}); automatic rollback also failed ({rollback_exc})"
                    ) from exc
                raise BackupError(f"restore failed and was rolled back: {exc}") from exc
            return {**result, "safety_backup": str(safety)}

    def _restore_verified(self, archive: Path, manifest: dict) -> dict:
        with tempfile.TemporaryDirectory(dir=self.settings.home, prefix=".restore-") as tmp:
            staging = Path(tmp)
            with zipfile.ZipFile(archive) as bundle:
                if any(not _safe_member(info) for info in bundle.infolist()):
                    raise BackupError("archive contains an unsafe path or link")
                bundle.extractall(staging)

            service = self.service_factory()
            snapshotter = QdrantSnapshots(self.settings, service)
            for corpus in manifest.get("corpora", []):
                snapshotter.restore(
                    corpus["name"],
                    staging / corpus["snapshot"],
                    corpus["snapshot_format"],
                    int(corpus["dim"]),
                )

            try:
                service.store.client.close()
            except Exception:
                pass

            current_config = _load_json(self.settings.config_file)
            restored_config = _load_json(staging / "config.json")
            for secret in SECRET_FIELDS:
                if secret in current_config:
                    restored_config[secret] = current_config[secret]
            _atomic_write_json(self.settings.config_file, restored_config)
            _atomic_write_json(self.settings.corpora_file, _load_json(staging / "corpora.json"))
            aliases = staging / "aliases.json"
            if aliases.exists():
                os.replace(aliases, self.settings.home / "aliases.json")
            write_source = staging / "write"
            if write_source.is_dir():
                write_target = self.settings.home / "write"
                if write_target.exists():
                    shutil.rmtree(write_target)
                os.replace(write_source, write_target)

            # Re-open after replacing the registry so diagnostics and rebuilds
            # see restored metadata (and embedded Qdrant releases its lock).
            service = self.service_factory()
            for corpus in manifest.get("corpora", []):
                name = corpus["name"]
                collection = _collection_for(name)
                service.bm25.drop(name)
                offset = None
                while True:
                    points, offset = service.store.client.scroll(
                        collection_name=collection,
                        limit=256,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    if points:
                        service.bm25.upsert(
                            name,
                            [str(point.id) for point in points],
                            [(point.payload or {}).get("text", "") for point in points],
                            [point.payload or {} for point in points],
                        )
                    if offset is None:
                        break
            issues = service.diagnose()
            try:
                service.store.client.close()
            except Exception:
                pass
        return {"restored": manifest["backup_id"], "issues": issues}


class BackupScheduler:
    """Small persisted scheduler; the server owns the polling thread."""

    def __init__(
        self,
        manager: BackupManager,
        *,
        enabled: bool = False,
        frequency: str = "daily",
        local_time: str = "03:00",
        weekday: int = 0,
        retention: int = 7,
    ) -> None:
        self.manager = manager
        self.enabled = enabled
        self.frequency = frequency
        self.local_time = local_time
        self.weekday = weekday
        self.retention = retention
        self.state_file = manager.settings.home / "backup-scheduler.json"

    def _slot(self, now: datetime) -> str | None:
        try:
            hour, minute = (int(part) for part in self.local_time.split(":"))
        except (ValueError, TypeError):
            return None
        if now.hour < hour or (now.hour == hour and now.minute < minute):
            return None
        if self.frequency == "weekly" and now.weekday() != self.weekday:
            return None
        if self.frequency not in {"daily", "weekly"}:
            return None
        return now.strftime("%Y-%m-%d")

    def run_if_due(self, now: datetime | None = None) -> Path | None:
        if not self.enabled:
            return None
        now = now or datetime.now().astimezone()
        slot = self._slot(now)
        state = _load_json(self.state_file)
        if slot is None or state.get("last_slot") == slot:
            return None
        archive = self.manager.create(origin="scheduled")
        self.manager.prune_scheduled(self.retention)
        _atomic_write_json(self.state_file, {"last_slot": slot, "updated_at": now.isoformat()})
        return archive

    def run_forever(self, stop: threading.Event) -> None:
        while not stop.wait(30):
            try:
                self.run_if_due()
            except Exception:
                # Server logging owns surfacing; a failed run is retried next poll.
                continue
