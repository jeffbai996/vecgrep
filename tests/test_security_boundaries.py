from __future__ import annotations

import json
import os
import stat
from contextlib import nullcontext
from pathlib import Path

import pytest
from click import ClickException
from fastapi.testclient import TestClient

from vecgrep.backend import config as cfg_mod
from vecgrep.backend.config import ConfigError, get_settings
from vecgrep.backend.main import create_app
from vecgrep.cli.main import _require_safe_serve_bind


def test_oauth_fails_closed_without_owner_or_rest_credentials(vg_home, monkeypatch):
    monkeypatch.setenv("VECGREP_OAUTH_ENABLED", "1")
    monkeypatch.setenv("VECGREP_OAUTH_ISSUER_URL", "https://example.test/mcp")
    monkeypatch.delenv("VECGREP_OAUTH_APPROVAL_TOKEN", raising=False)
    monkeypatch.delenv("VECGREP_API_TOKEN", raising=False)
    monkeypatch.setattr(cfg_mod, "_settings", None)
    with pytest.raises(ConfigError, match="OAUTH_APPROVAL_TOKEN"):
        get_settings()


def test_non_loopback_config_fails_closed_when_rest_api_would_be_anonymous(
    vg_home, monkeypatch
):
    monkeypatch.setenv("VECGREP_API_HOST", "0.0.0.0")
    monkeypatch.delenv("VECGREP_API_TOKEN", raising=False)
    monkeypatch.setattr(cfg_mod, "_settings", None)
    with pytest.raises(ConfigError, match="non-loopback api_host"):
        get_settings()


def test_non_loopback_bind_requires_api_token():
    with pytest.raises(ClickException, match="non-loopback bind"):
        _require_safe_serve_bind("0.0.0.0", None)
    _require_safe_serve_bind("0.0.0.0", "configured-token-with-at-least-32-chars")
    _require_safe_serve_bind("127.0.0.1", None)
    _require_safe_serve_bind("::1", None)


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://evil.example"},
        {"Referer": "https://evil.example/form"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site", "Origin": "http://evil.localhost"},
    ],
)
def test_loopback_admin_mutations_reject_foreign_browser_provenance(
    vg_home, headers
):
    with TestClient(create_app()) as client:
        response = client.post("/api/admin/config/reload", headers=headers)
    assert response.status_code == 403


def test_loopback_admin_allows_headerless_cli_and_same_origin_browser(vg_home):
    with TestClient(
        create_app(),
        base_url="http://127.0.0.1",
        client=("127.0.0.1", 50000),
    ) as client:
        assert client.post("/api/admin/config/reload").status_code == 200
        response = client.post(
            "/api/admin/config/reload",
            headers={
                "Origin": "http://127.0.0.1",
                "Sec-Fetch-Site": "same-origin",
            },
        )
    assert response.status_code == 200


def test_runtime_home_is_private_under_permissive_umask(tmp_path, monkeypatch):
    home = tmp_path / "private-vecgrep"
    monkeypatch.setenv("VECGREP_HOME", str(home))
    monkeypatch.setattr(cfg_mod, "_settings", None)
    previous = os.umask(0o000)
    try:
        settings = get_settings()
    finally:
        os.umask(previous)
    assert stat.S_IMODE(settings.home.stat().st_mode) == 0o700
    assert stat.S_IMODE(settings.qdrant_path.stat().st_mode) == 0o700


def test_direct_write_is_mode_0600_under_permissive_umask(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    home = tmp_path / "vg"
    monkeypatch.setenv("VECGREP_HOME", str(home))
    monkeypatch.setenv("VECGREP_DIRECT_WRITE_CORPUS", "external")
    monkeypatch.setattr(cfg_mod, "_settings", None)
    from vecgrep.mcp import server as mcp_server

    class Locks:
        def write(self, _name):
            return nullcontext()

    class Service:
        locks = Locks()

        def _recover_corpus_locked(self, _name):
            return None

        def index(self, *_args, **_kwargs):
            return None

    previous = os.umask(0o000)
    try:
        monkeypatch.setattr(mcp_server, "_svc", lambda: Service())
        result = json.loads(mcp_server._run_direct_write("private text"))
    finally:
        os.umask(previous)
    assert "error" not in result
    path = Path(result["path"])
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
