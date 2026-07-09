from __future__ import annotations

import hmac
import ipaddress
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..config import (
    EDITABLE_FIELDS,
    SECRET_FIELDS,
    STRUCTURAL_FIELDS,
    ConfigError,
    config_provenance,
    get_settings,
    reset_settings,
    update_config,
)


def _is_loopback(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return value.lower() == "localhost"


def require_admin(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    peer = request.client.host if request.client else ""
    host = urlsplit(f"//{request.headers.get('host', '')}").hostname or ""
    if _is_loopback(peer) and _is_loopback(host):
        return

    expected = get_settings().admin_token
    if expected and authorization and authorization.startswith("Bearer "):
        provided = authorization[len("Bearer ") :].strip()
        if hmac.compare_digest(provided, expected.strip()):
            return
    raise HTTPException(status_code=401, detail="Admin access requires loopback or admin token")


router = APIRouter(prefix="/api/admin", dependencies=[Depends(require_admin)])


class ConfigPatch(BaseModel):
    values: dict = Field(default_factory=dict)
    confirm_qdrant: str | None = None


class RestoreRequest(BaseModel):
    confirm: str


class SchedulePatch(BaseModel):
    values: dict = Field(default_factory=dict)


def _config_payload() -> dict:
    settings = get_settings()
    values = asdict(settings)
    values["home"] = str(settings.home)
    for name in SECRET_FIELDS:
        values.pop(name, None)
    provenance = config_provenance(settings)
    return {
        "values": {name: values[name] for name in sorted(EDITABLE_FIELDS)},
        "home": values["home"],
        "provenance": provenance,
        "read_only": sorted(name for name, source in provenance.items() if source == "env"),
        "secrets": {name: bool(getattr(settings, name)) for name in sorted(SECRET_FIELDS)},
    }


@router.get("/config")
def get_admin_config() -> dict:
    return _config_payload()


@router.patch("/config")
def patch_admin_config(patch: ConfigPatch) -> dict:
    before = get_settings()
    changed = {name for name, value in patch.values.items() if getattr(before, name, object()) != value}
    try:
        update_config(patch.values, confirm_qdrant=patch.confirm_qdrant)
    except ConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _reload_runtime()
    warnings = []
    if "embed_model" in changed or "openai_embed_model" in changed:
        warnings.append("existing corpora remain pinned to their indexed embedding model")
    return {
        **_config_payload(),
        "restart_required": bool(changed & STRUCTURAL_FIELDS),
        "warnings": warnings,
    }


def _reload_runtime() -> None:
    from . import routes

    if routes._SERVICE is not None:
        try:
            routes._SERVICE.store.client.close()
        except Exception:
            pass
        routes._SERVICE = None
    reset_settings()
    try:
        from ...mcp.server import _reset_service_cache

        _reset_service_cache()
    except (ImportError, RuntimeError):
        pass


@router.post("/config/reload")
def reload_admin_config() -> dict:
    _reload_runtime()
    return {**_config_payload(), "reloaded": True}


def _backup_manager():
    from ..backup import BackupManager

    return BackupManager(get_settings())


def _backup_by_id(backup_id: str) -> dict:
    for item in _backup_manager().list():
        if item.get("backup_id") == backup_id:
            return item
    raise HTTPException(status_code=404, detail="Backup not found")


@router.get("/backups")
def list_backups() -> list[dict]:
    return _backup_manager().list()


@router.post("/backups")
def create_backup() -> dict:
    from ..backup import BackupError

    try:
        path = _backup_manager().create(origin="manual")
        return _backup_manager().verify(path)
    except BackupError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/backups/{backup_id}/verify")
def verify_backup(backup_id: str) -> dict:
    from ..backup import BackupError

    try:
        return _backup_manager().verify(Path(_backup_by_id(backup_id)["path"]))
    except BackupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/backups/{backup_id}/restore")
def restore_backup(backup_id: str, body: RestoreRequest) -> dict:
    from ..backup import BackupError

    try:
        result = _backup_manager().restore(
            Path(_backup_by_id(backup_id)["path"]), confirm=body.confirm
        )
        _reload_runtime()
        return result
    except BackupError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/backups/{backup_id}/download")
def download_backup(backup_id: str) -> FileResponse:
    path = Path(_backup_by_id(backup_id)["path"])
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


_SCHEDULE_FIELDS = {
    "backup_enabled",
    "backup_frequency",
    "backup_time",
    "backup_weekday",
    "backup_destination",
    "backup_retention",
}


@router.get("/backup-schedule")
def get_backup_schedule() -> dict:
    settings = get_settings()
    return {name: getattr(settings, name) for name in sorted(_SCHEDULE_FIELDS)}


@router.patch("/backup-schedule")
def patch_backup_schedule(body: SchedulePatch) -> dict:
    if set(body.values) - _SCHEDULE_FIELDS:
        raise HTTPException(status_code=422, detail="Unknown schedule setting")
    try:
        update_config(body.values)
    except ConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    reset_settings()
    return get_backup_schedule()
