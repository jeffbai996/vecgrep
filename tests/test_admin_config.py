from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from vecgrep.backend import config


def test_config_provenance_and_env_precedence(vg_home: Path, monkeypatch) -> None:
    vg_home.mkdir(parents=True)
    (vg_home / "config.json").write_text(
        json.dumps({"ollama_url": "http://from-file:11434", "default_top_k": 12})
    )
    monkeypatch.setenv("VECGREP_OLLAMA_URL", "http://from-env:11434")
    settings = config.load_settings()

    assert settings.ollama_url == "http://from-env:11434"
    assert settings.default_top_k == 12
    assert config.config_provenance(settings)["ollama_url"] == "env"
    assert config.config_provenance(settings)["default_top_k"] == "file"
    assert config.config_provenance(settings)["embed_model"] == "default"


def test_ollama_num_batch_loads_as_an_integer(vg_home: Path, monkeypatch) -> None:
    monkeypatch.setenv("VECGREP_OLLAMA_NUM_BATCH", "2048")

    settings = config.load_settings()

    assert settings.ollama_num_batch == 2048
    assert config.config_provenance(settings)["ollama_num_batch"] == "env"


def test_update_config_is_atomic_and_preserves_unknown_keys(
    vg_home: Path, monkeypatch
) -> None:
    vg_home.mkdir(parents=True)
    path = vg_home / "config.json"
    path.write_text(json.dumps({"future_option": {"enabled": True}, "default_top_k": 5}))

    updated = config.update_config({"default_top_k": 20, "embed_model": "bge-m3"})

    assert updated["future_option"] == {"enabled": True}
    assert updated["default_top_k"] == 20
    assert updated["embed_model"] == "bge-m3"
    assert path.stat().st_mode & 0o777 == 0o600
    assert not list(vg_home.glob(".config.json.*"))


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"ollama_url": "ftp://host/model"}, "http or https"),
        ({"api_port": 70000}, "between 1 and 65535"),
        ({"default_top_k": 0}, "positive"),
        ({"ollama_num_batch": 0}, "positive"),
        ({"embed_model": "  "}, "non-empty"),
        ({"oauth_enabled": True, "oauth_issuer_url": None}, "issuer"),
    ],
)
def test_update_config_validates(vg_home: Path, patch: dict, message: str) -> None:
    with pytest.raises(config.ConfigError, match=message):
        config.update_config(patch)


def test_qdrant_change_requires_typed_confirmation(vg_home: Path) -> None:
    target = "http://localhost:6333"
    with pytest.raises(config.ConfigError, match="confirmation"):
        config.update_config({"qdrant_url": target})
    config.update_config({"qdrant_url": target}, confirm_qdrant=target)
    assert config.load_settings().qdrant_url == target


def _admin_request(peer: str, host: str) -> Request:
    return Request({
        "type": "http",
        "method": "GET",
        "path": "/api/admin/config",
        "headers": [(b"host", host.encode())],
        "client": (peer, 1234),
    })


def _patch_admin_settings(monkeypatch, *, token: str | None = None):
    from vecgrep.backend.api import admin

    settings = config.Settings(admin_token=token)
    monkeypatch.setattr(admin, "get_settings", lambda: settings)
    return admin


def test_admin_auth_allows_only_socket_and_host_loopback(monkeypatch) -> None:
    admin = _patch_admin_settings(monkeypatch)
    admin.require_admin(_admin_request("127.0.0.1", "127.0.0.1:8765"), None)
    with pytest.raises(HTTPException) as public_host:
        admin.require_admin(_admin_request("127.0.0.1", "public.example"), None)
    assert public_host.value.status_code == 401
    with pytest.raises(HTTPException) as public_peer:
        admin.require_admin(_admin_request("192.0.2.10", "127.0.0.1:8765"), None)
    assert public_peer.value.status_code == 401


def test_admin_auth_accepts_separate_bearer_token(monkeypatch) -> None:
    admin = _patch_admin_settings(monkeypatch, token="admin-secret")
    admin.require_admin(
        _admin_request("192.0.2.10", "public.example"),
        "Bearer admin-secret",
    )


def test_admin_patch_reports_restart_and_model_warning(vg_home: Path, monkeypatch) -> None:
    from vecgrep.backend.api import admin

    monkeypatch.setattr(admin, "_reload_runtime", config.reset_settings)
    body = admin.patch_admin_config(
        admin.ConfigPatch(values={"api_port": 9000, "embed_model": "nomic-embed-text"})
    )
    assert body["restart_required"] is True
    assert any("existing corpora" in warning for warning in body["warnings"])
