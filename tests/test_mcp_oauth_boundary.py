"""OAuth protects proxied MCP while trusted direct-loopback clients still work."""
from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

pytest.importorskip("mcp")

from fastapi.testclient import TestClient

from vecgrep.backend import config as cfg_mod
from vecgrep.backend.main import create_app
from vecgrep.mcp import server as mcp_server
from vecgrep.mcp.server import (
    _has_loopback_server,
    _is_direct_loopback_mcp,
    _is_verified_tailnet_mcp,
)


TEST_ISSUER = "https://example.com:10000/mcp"
TEST_APPROVAL = "a" * 40
MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "boundary-test", "version": "1"},
    },
}


@pytest.fixture
def loopback_client(vg_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("VECGREP_OAUTH_ENABLED", "1")
    monkeypatch.setenv("VECGREP_OAUTH_ISSUER_URL", TEST_ISSUER)
    monkeypatch.setenv("VECGREP_OAUTH_APPROVAL_TOKEN", TEST_APPROVAL)
    monkeypatch.setenv("VECGREP_API_TOKEN", "r" * 40)
    monkeypatch.setattr(cfg_mod, "_settings", None)
    with TestClient(
        create_app(),
        base_url="http://127.0.0.1:8765",
        client=("127.0.0.1", 50000),
    ) as client:
        yield client


def test_direct_loopback_can_initialize_and_list_tools(loopback_client: TestClient) -> None:
    initialized = loopback_client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS)
    assert initialized.status_code == 200
    assert initialized.json()["result"]["serverInfo"]["name"] == "vecgrep"

    listed = loopback_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        headers=MCP_HEADERS,
    )
    assert listed.status_code == 200
    tools = listed.json()["result"]["tools"]
    assert "search" in {tool["name"] for tool in tools}
    search = next(tool for tool in tools if tool["name"] == "search")
    corpora_schema = search["inputSchema"]["properties"]["corpora"]
    assert "array" in json.dumps(corpora_schema)
    assert "string" in json.dumps(corpora_schema)
    assert "list_corpora" in search["description"]


def test_dns_rebinding_host_is_rejected_before_loopback_bypass(
    loopback_client: TestClient,
) -> None:
    response = loopback_client.post(
        "/mcp",
        json=INITIALIZE,
        headers=MCP_HEADERS | {"Host": "attacker.example"},
    )
    assert response.status_code == 421


def test_foreign_browser_origin_is_rejected_on_an_allowed_host(
    loopback_client: TestClient,
) -> None:
    response = loopback_client.post(
        "/mcp",
        json=INITIALIZE,
        headers=MCP_HEADERS | {"Origin": "https://attacker.example"},
    )
    assert response.status_code == 403


@pytest.mark.parametrize(
    "forwarded",
    [
        {"X-Forwarded-For": "1.2.3.4"},
        {"X-Forwarded-Proto": "https"},
    ],
)
def test_proxy_markers_force_oauth(
    loopback_client: TestClient, forwarded: dict[str, str]
) -> None:
    response = loopback_client.post(
        "/mcp", json=INITIALIZE, headers=MCP_HEADERS | forwarded
    )
    assert response.status_code == 401
    assert response.headers["www-authenticate"].startswith("Bearer ")
    assert "resource_metadata=" in response.headers["www-authenticate"]


def test_verified_tailscale_serve_identity_can_list_tools(
    loopback_client: TestClient,
) -> None:
    response = loopback_client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        headers=MCP_HEADERS | {
            "Host": "example.com:10000",
            "X-Forwarded-For": "100.64.0.10",
            "X-Forwarded-Proto": "https",
            "Tailscale-User-Login": "owner@example.test",
            "Tailscale-Headers-Info": "https://tailscale.com/s/serve-headers",
        },
    )
    assert response.status_code == 200
    assert "search" in {tool["name"] for tool in response.json()["result"]["tools"]}


def test_valid_oauth_token_still_works_through_proxy(
    loopback_client: TestClient,
) -> None:
    access = mcp_server._shared_provider().store.issue_access_token(
        "oauth-client", ["read"], ttl_s=60
    )
    response = loopback_client.post(
        "/mcp",
        json=INITIALIZE,
        headers=MCP_HEADERS | {
            "Host": "example.com:10000",
            "Origin": "https://example.com:10000",
            "X-Forwarded-For": "203.0.113.10",
            "X-Forwarded-Proto": "https",
            "Authorization": f"Bearer {access.token}",
        },
    )
    assert response.status_code == 200
    assert response.json()["result"]["serverInfo"]["name"] == "vecgrep"


def test_setting_false_gates_direct_loopback(vg_home, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECGREP_OAUTH_ENABLED", "1")
    monkeypatch.setenv("VECGREP_OAUTH_ISSUER_URL", TEST_ISSUER)
    monkeypatch.setenv("VECGREP_OAUTH_APPROVAL_TOKEN", TEST_APPROVAL)
    monkeypatch.setenv("VECGREP_OAUTH_LOOPBACK_BYPASS", "false")
    monkeypatch.setenv("VECGREP_API_TOKEN", "r" * 40)
    monkeypatch.setattr(cfg_mod, "_settings", None)
    with TestClient(
        create_app(),
        base_url="http://127.0.0.1:8765",
        client=("127.0.0.1", 50000),
    ) as client:
        direct = client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS)
        tailnet = client.post(
            "/mcp",
            json=INITIALIZE,
            headers=MCP_HEADERS | {
                "X-Forwarded-For": "100.64.0.10",
                "Tailscale-User-Login": "owner@example.test",
                "Tailscale-Headers-Info": "https://tailscale.com/s/serve-headers",
            },
        )
    assert direct.status_code == 401
    assert tailnet.status_code == 401


def test_api_health_is_unchanged_by_proxy_classification(
    loopback_client: TestClient,
) -> None:
    assert loopback_client.get("/api/health").status_code == 200
    assert loopback_client.get(
        "/api/health",
        headers={"X-Forwarded-For": "1.2.3.4", "X-Forwarded-Proto": "https"},
    ).status_code == 200


@pytest.mark.parametrize(
    "scope",
    [
        {"type": "http", "path": "/", "headers": []},
        {"type": "http", "path": "/", "client": ("not-an-ip", 1), "headers": []},
        {"type": "http", "path": "/", "client": ("192.0.2.1", 1), "headers": []},
        {"type": "http", "path": "/authorize", "client": ("127.0.0.1", 1), "headers": []},
    ],
)
def test_unclassifiable_or_non_mcp_requests_fail_closed(scope: dict) -> None:
    assert _is_direct_loopback_mcp(scope) is False


@pytest.mark.parametrize(
    "marker",
    [
        (b"x-forwarded-for", b"203.0.113.9"),
        (b"x-forwarded-host", b"node.example.ts.net"),
        (b"forwarded", b"for=203.0.113.9"),
        (b"tailscale-funnel-request", b"?1"),
        (b"tailscale-headers-info", b"https://tailscale.com/s/serve-headers"),
        (b"tailscale-user-login", b"someone@example.test"),
    ],
)
def test_any_proxy_or_tailscale_marker_disables_loopback_bypass(marker) -> None:
    """A Funnel request arrives on the loopback peer via the Windows portproxy;
    the stamped headers are the only thing that says it is not a local call."""
    assert _is_direct_loopback_mcp({
        "type": "http", "path": "/", "client": ("127.0.0.1", 1), "headers": [marker],
    }) is False
    assert _is_direct_loopback_mcp({
        "type": "http", "path": "/", "client": ("127.0.0.1", 1), "headers": [],
    }) is True


def test_tailscale_identity_requires_loopback_listener_and_forwarding_marker() -> None:
    identity = (b"tailscale-user-login", b"owner@example.test")
    forwarded = (b"x-forwarded-for", b"100.64.0.10")
    info = (b"tailscale-headers-info", b"https://tailscale.com/s/serve-headers")
    assert _is_verified_tailnet_mcp({
        "type": "http", "path": "/", "server": ("127.0.0.1", 8765),
        "client": ("100.64.0.10", 1),
        "headers": [identity, forwarded, info],
    }) is True
    assert _is_verified_tailnet_mcp({
        "type": "http", "path": "/", "server": ("192.0.2.2", 8765),
        "client": ("100.64.0.10", 1),
        "headers": [identity, forwarded, info],
    }) is False
    assert _is_verified_tailnet_mcp({
        "type": "http", "path": "/", "server": ("127.0.0.1", 8765),
        "client": ("100.64.0.10", 1),
        "headers": [identity],
    }) is False
    assert _is_verified_tailnet_mcp({
        "type": "http", "path": "/", "server": ("127.0.0.1", 8765),
        "client": ("100.64.0.10", 1),
        "headers": [identity, forwarded],
    }) is False
    assert _is_verified_tailnet_mcp({
        "type": "http", "path": "/", "server": ("127.0.0.1", 8765),
        "client": ("100.64.0.10", 1),
        "headers": [
            identity, forwarded, info, (b"tailscale-funnel-request", b"?1")
        ],
    }) is False


@pytest.mark.parametrize(
    ("server", "expected"),
    [
        (("127.0.0.1", 8765), True),
        (("::1", 8765), True),
        (("localhost", 8765), True),
        (("0.0.0.0", 8765), False),
        (("192.0.2.2", 8765), False),
        (None, False),
    ],
)
def test_loopback_listener_classification(server, expected: bool) -> None:
    scope = {"server": server} if server is not None else {}
    assert _has_loopback_server(scope) is expected


def test_loopback_bypass_config_defaults_true_and_parses_false(
    vg_home, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cfg_mod.load_settings().oauth_loopback_bypass is True
    assert cfg_mod.load_settings().oauth_tailscale_identity_bypass is True
    monkeypatch.setenv("VECGREP_OAUTH_LOOPBACK_BYPASS", "false")
    monkeypatch.setenv("VECGREP_OAUTH_TAILSCALE_IDENTITY_BYPASS", "false")
    assert cfg_mod.load_settings().oauth_loopback_bypass is False
    assert cfg_mod.load_settings().oauth_tailscale_identity_bypass is False
