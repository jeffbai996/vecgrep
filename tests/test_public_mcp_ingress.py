"""Public proxy ingress cannot inherit the private listener's network trust."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("mcp")

from vecgrep.backend import config
from vecgrep.backend.main import create_app
from vecgrep.mcp import server

HEADERS = {"Accept": "application/json, text/event-stream"}
SPOOF = {
    "X-Forwarded-For": "100.64.0.10",
    "Tailscale-User-Login": "owner@example.test",
    "Tailscale-Headers-Info": "https://tailscale.com/s/serve-headers",
}
INITIALIZE = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-03-26", "capabilities": {},
               "clientInfo": {"name": "ingress-test", "version": "1"}},
}


class Listener:
    """Supply the accepted socket address, independently of HTTP Host."""
    def __init__(self, app, port):
        self.app, self.port = app, port

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope = dict(scope, server=("127.0.0.1", self.port))
        await self.app(scope, receive, send)


@pytest.fixture
def ingress_app(vg_home, monkeypatch):
    monkeypatch.setenv("VECGREP_OAUTH_ENABLED", "1")
    monkeypatch.setenv("VECGREP_OAUTH_ISSUER_URL", "https://example.com/mcp")
    monkeypatch.setenv("VECGREP_OAUTH_APPROVAL_TOKEN", "a" * 40)
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(server, "_PROVIDER", None)
    return create_app()


@pytest.mark.parametrize("path", ["/mcp", "/mcp/"])
@pytest.mark.parametrize("headers", [{}, SPOOF, SPOOF | {"Tailscale-Funnel-Request": "?1"}])
def test_public_listener_requires_oauth_even_with_loopback_host(ingress_app, path, headers):
    with TestClient(Listener(ingress_app, 8766), base_url="http://127.0.0.1:8765",
                    client=("127.0.0.1", 40000)) as client:
        response = client.post(path, json=INITIALIZE, headers=HEADERS | headers)
        assert response.status_code == 401
        assert response.headers["www-authenticate"].startswith("Bearer ")


def test_public_identity_cannot_unlock_owner_approval(ingress_app):
    with TestClient(Listener(ingress_app, 8766), base_url="https://example.com",
                    client=("127.0.0.1", 40000)) as client:
        page = client.get("/oauth/unlock", headers=SPOOF)
        assert page.status_code == 200
        assert 'name="tailnet_intent"' not in page.text
        denied = client.post("/oauth/unlock", headers=SPOOF,
                             data={"tailnet_intent": "a" * 64})
        assert denied.status_code == 401


def test_public_and_private_share_tokens_and_private_access(ingress_app):
    with TestClient(Listener(ingress_app, 8766), base_url="https://example.com") as public:
        token = server._shared_provider().store.issue_access_token("test-client", ["read"], ttl_s=60)
        accepted = public.post("/mcp", json=INITIALIZE,
                               headers=HEADERS | {"Authorization": f"Bearer {token.token}"})
        assert accepted.status_code == 200
        denied_write = public.post("/mcp", headers=HEADERS | {
            "Authorization": f"Bearer {token.token}",
        }, json={"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "propose_write", "arguments": {"content": "test", "corpus": "notes"},
        }})
        assert "requires the 'propose' scope" in denied_write.text
        # A second client uses the same lifespan, as do both real sockets.
        private = TestClient(Listener(ingress_app, 8765), base_url="http://127.0.0.1:8765",
                             client=("127.0.0.1", 40000))
        assert private.post("/mcp", json=INITIALIZE, headers=HEADERS).status_code == 200
        assert private.post("/mcp", json=INITIALIZE, headers=HEADERS | SPOOF).status_code == 200


def test_public_listener_does_not_expose_rest(ingress_app):
    with TestClient(Listener(ingress_app, 8766)) as client:
        assert client.get("/api/health").status_code == 404


def test_public_listener_fails_closed_without_oauth(vg_home, monkeypatch):
    monkeypatch.setenv("VECGREP_OAUTH_ENABLED", "0")
    with TestClient(Listener(create_app(), 8766)) as client:
        assert client.post("/mcp", json=INITIALIZE, headers=HEADERS).status_code == 503


def test_missing_ingress_cannot_claim_private_trust():
    from vecgrep.backend.auth.approval import verified_tailnet_login
    scope = {"type": "http", "path": "/", "client": ("127.0.0.1", 1),
             "server": ("127.0.0.1", 8765), "headers": []}
    assert server._is_direct_loopback_mcp(scope) is False
    scope["headers"] = [(k.lower().encode(), v.encode()) for k, v in SPOOF.items()]
    assert verified_tailnet_login(scope) is None


def test_public_duplicate_identity_headers_cannot_acquire_trust(ingress_app):
    with TestClient(Listener(ingress_app, 8766), base_url="http://127.0.0.1:8765",
                    client=("127.0.0.1", 40000)) as client:
        headers = list((HEADERS | SPOOF).items()) + [
            ("Tailscale-User-Login", "another@example.test"),
            ("Host", "127.0.0.1:8765"),
        ]
        assert client.post("/mcp", json=INITIALIZE, headers=headers).status_code == 401


@pytest.mark.parametrize("reload", [False, True])
def test_live_dual_listener_uses_socket_not_host(vg_home, reload):
    import os
    import socket
    import subprocess
    import sys
    import time

    import httpx

    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        private_port = first.getsockname()[1]
        public_port = second.getsockname()[1]
    env = os.environ | {
        "VECGREP_OAUTH_PUBLIC_PORT": str(public_port),
        "VECGREP_OAUTH_ENABLED": "1",
        "VECGREP_OAUTH_ISSUER_URL": "https://example.com/mcp",
        "VECGREP_OAUTH_APPROVAL_TOKEN": "a" * 40,
    }
    log_path = vg_home.parent / "ingress-server.log"
    with log_path.open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", "from vecgrep.cli.main import cli; cli()", "serve",
             "--port", str(private_port)] + (["--reload"] if reload else []),
            env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            with httpx.Client(timeout=2, trust_env=False) as client:
                private = f"http://127.0.0.1:{private_port}"
                public = f"http://127.0.0.1:{public_port}"
                deadline = time.monotonic() + 20
                while True:
                    try:
                        ready = client.get(private + "/api/health")
                        if ready.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    if process.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        pytest.fail("server startup failed: " + log.read())
                    time.sleep(0.05)
                assert client.post(private + "/mcp", json=INITIALIZE, headers=HEADERS).status_code == 200
                spoof = HEADERS | SPOOF | {"Host": f"127.0.0.1:{private_port}"}
                assert client.post(public + "/mcp", json=INITIALIZE, headers=spoof).status_code == 401
                assert client.post(public + "/mcp", json=INITIALIZE, headers=HEADERS).status_code == 401
                assert client.get(public + "/api/health").status_code == 404
                assert client.get(public + "/.well-known/oauth-authorization-server").status_code == 200
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.mark.parametrize("port", [0, 65536, 8765])
def test_invalid_public_listener_configuration_is_rejected(vg_home, monkeypatch, port):
    monkeypatch.setenv("VECGREP_OAUTH_ENABLED", "1")
    monkeypatch.setenv("VECGREP_OAUTH_ISSUER_URL", "https://example.com/mcp")
    monkeypatch.setenv("VECGREP_OAUTH_APPROVAL_TOKEN", "a" * 40)
    monkeypatch.setenv("VECGREP_OAUTH_PUBLIC_PORT", str(port))
    with pytest.raises(config.ConfigError):
        config.load_settings()


def test_oauth_off_custom_api_port_is_unchanged(vg_home, monkeypatch):
    monkeypatch.setenv("VECGREP_OAUTH_ENABLED", "0")
    monkeypatch.setenv("VECGREP_API_PORT", "8766")
    with TestClient(Listener(create_app(), 8766), base_url="http://127.0.0.1:8766") as client:
        assert client.get("/api/health").status_code == 200


def test_oauth_off_cli_port_override_remains_private(vg_home, monkeypatch):
    monkeypatch.setenv("VECGREP_OAUTH_ENABLED", "0")
    monkeypatch.setenv("VECGREP_API_PORT", "8765")
    import uvicorn
    from click.testing import CliRunner
    from vecgrep.cli.main import cli

    def run(_app, **kwargs):
        with TestClient(Listener(create_app(), kwargs["port"]),
                        base_url="http://127.0.0.1:8766") as client:
            assert client.get("/api/health").status_code == 200

    monkeypatch.setattr(uvicorn, "run", run)
    result = CliRunner().invoke(cli, ["serve", "--port", "8766"])
    assert result.exit_code == 0, repr(result.exception)
