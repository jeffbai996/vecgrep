"""OAuth protects proxied MCP while trusted direct-loopback clients still work."""
from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

pytest.importorskip("mcp")

from fastapi.testclient import TestClient

from vecgrep.backend import config as cfg_mod
from vecgrep.backend.auth.ingress import PRIVATE_INGRESS
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
    monkeypatch.setattr(mcp_server, "_PROVIDER", None)
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


def test_oauth_tools_advertise_their_required_scopes(loopback_client):
    response = loopback_client.post(
        "/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        headers=MCP_HEADERS,
    )
    tools = {tool["name"]: tool for tool in response.json()["result"]["tools"]}
    for name, scopes in [("search", ["read"]), ("propose_write", ["read", "propose"]),
                         ("write", ["read", "propose"])]:
        expected = [{"type": "oauth2", "scopes": scopes}]
        assert tools[name]["securitySchemes"] == expected
        assert tools[name]["_meta"]["securitySchemes"] == expected


@pytest.mark.parametrize("scopes", [["read"], ["read", "propose"]])
def test_missing_scope_returns_oauth_challenge_without_running_tool(
    loopback_client, monkeypatch, scopes,
):
    calls = []
    monkeypatch.setattr(mcp_server, "_run_propose",
                        lambda *args: calls.append(args) or '{"test": true}')
    access = mcp_server._shared_provider().store.issue_access_token(
        "scope-client", scopes, ttl_s=60, resource=TEST_ISSUER,
    )
    response = loopback_client.post(
        "/mcp", headers=MCP_HEADERS | {
            "X-Forwarded-For": "203.0.113.10",
            "Authorization": f"Bearer {access.token}",
        },
        json={"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "propose_write", "arguments": {"content": "test", "corpus": "notes"},
        }},
    )
    assert response.status_code == 200
    result = response.json()["result"]
    if "propose" in scopes:
        assert len(calls) == 1
        assert result["isError"] is False
        assert "mcp/www_authenticate" not in result.get("_meta", {})
    else:
        assert calls == []
        assert result["isError"] is True
        challenge = result["_meta"]["mcp/www_authenticate"][0]
        assert 'scope="read propose"' in challenge
        assert 'error="insufficient_scope"' in challenge
        assert 'resource_metadata="https://example.com:10000/.well-known/oauth-protected-resource/mcp"' in challenge
        assert access.token not in response.text


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
            "X-Forwarded-For": "192.0.2.10",
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
        "oauth-client", ["read"], ttl_s=60, resource=TEST_ISSUER
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
    monkeypatch.setattr(mcp_server, "_PROVIDER", None)
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
                "X-Forwarded-For": "192.0.2.10",
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
    """Private ingress still rejects proxy markers before the local bypass."""
    assert _is_direct_loopback_mcp({
        PRIVATE_INGRESS: True, "type": "http", "path": "/", "client": ("127.0.0.1", 1), "headers": [marker],
    }) is False
    assert _is_direct_loopback_mcp({
        PRIVATE_INGRESS: True, "type": "http", "path": "/", "client": ("127.0.0.1", 1), "headers": [],
    }) is True


def test_tailscale_identity_requires_loopback_listener_and_forwarding_marker() -> None:
    identity = (b"tailscale-user-login", b"owner@example.test")
    forwarded = (b"x-forwarded-for", b"192.0.2.10")
    info = (b"tailscale-headers-info", b"https://tailscale.com/s/serve-headers")
    assert _is_verified_tailnet_mcp({
        PRIVATE_INGRESS: True, "type": "http", "path": "/", "server": ("127.0.0.1", 8765),
        "client": ("127.0.0.1", 1),
        "headers": [identity, forwarded, info],
    }) is True
    assert _is_verified_tailnet_mcp({
        PRIVATE_INGRESS: True, "type": "http", "path": "/", "server": ("127.0.0.1", 8765),
        "client": ("192.0.2.10", 1),
        "headers": [identity, forwarded, info],
    }) is False
    assert _is_verified_tailnet_mcp({
        PRIVATE_INGRESS: True, "type": "http", "path": "/", "server": ("192.0.2.2", 8765),
        "client": ("192.0.2.10", 1),
        "headers": [identity, forwarded, info],
    }) is False
    assert _is_verified_tailnet_mcp({
        PRIVATE_INGRESS: True, "type": "http", "path": "/", "server": ("127.0.0.1", 8765),
        "client": ("192.0.2.10", 1),
        "headers": [identity],
    }) is False
    assert _is_verified_tailnet_mcp({
        PRIVATE_INGRESS: True, "type": "http", "path": "/", "server": ("127.0.0.1", 8765),
        "client": ("192.0.2.10", 1),
        "headers": [identity, forwarded],
    }) is False
    assert _is_verified_tailnet_mcp({
        PRIVATE_INGRESS: True, "type": "http", "path": "/", "server": ("127.0.0.1", 8765),
        "client": ("192.0.2.10", 1),
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


@pytest.mark.parametrize("resource,expected", [
    (TEST_ISSUER, 200),
    ("https://other.example/mcp", 401),
    (None, 401),
])
def test_public_mcp_enforces_token_resource(loopback_client, resource, expected):
    access = mcp_server._shared_provider().store.issue_access_token(
        "resource-client", ["read"], ttl_s=60, resource=resource,
    )
    for message in (INITIALIZE, {"jsonrpc": "2.0", "id": 2,
                                 "method": "tools/list", "params": {}}):
        response = loopback_client.post(
            "/mcp", json=message,
            headers=MCP_HEADERS | {"X-Forwarded-For": "203.0.113.10",
                                   "Authorization": f"Bearer {access.token}"},
        )
        assert response.status_code == expected
        if expected == 401:
            assert 'error="invalid_token"' in response.headers["www-authenticate"]
            assert access.token not in response.text


@pytest.mark.parametrize("authorization_resource", [None, TEST_ISSUER])
def test_http_code_and_refresh_tokens_initialize_and_list_tools(loopback_client, authorization_resource):
    import base64
    import hashlib
    from urllib.parse import parse_qs, urlsplit

    import anyio
    from mcp.server.auth.provider import AuthorizationParams
    from mcp.shared.auth import OAuthClientInformationFull
    from pydantic import AnyUrl

    provider = mcp_server._shared_provider()
    client = OAuthClientInformationFull(
        client_id="resource-flow", redirect_uris=[AnyUrl("https://client.example/callback")],
        grant_types=["authorization_code", "refresh_token"], response_types=["code"],
        token_endpoint_auth_method="none", scope="read",
    )
    assert client.redirect_uris is not None
    verifier = "v" * 64
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    anyio.run(provider.register_client, client)
    params = AuthorizationParams(
        state="s", scopes=["read"], code_challenge=challenge,
        redirect_uri=client.redirect_uris[0], redirect_uri_provided_explicitly=True,
        resource=authorization_resource,
    )
    redirect = anyio.run(provider.authorize, client, params)
    form = {
        "grant_type": "authorization_code", "client_id": client.client_id,
        "code": parse_qs(urlsplit(redirect).query)["code"][0],
        "redirect_uri": str(client.redirect_uris[0]), "code_verifier": verifier,
    }
    response = loopback_client.post("/token", data=form)
    assert response.status_code == 200
    pair = response.json()
    for grant in ("authorization_code", "refresh_token"):
        if grant == "refresh_token":
            response = loopback_client.post("/token", data={
                "grant_type": grant, "client_id": client.client_id,
                "refresh_token": pair["refresh_token"],
            })
            assert response.status_code == 200
            pair = response.json()
        for message in (INITIALIZE, {"jsonrpc": "2.0", "id": 2,
                                     "method": "tools/list", "params": {}}):
            accepted = loopback_client.post("/mcp", json=message, headers=MCP_HEADERS | {
                "X-Forwarded-For": "203.0.113.10",
                "Authorization": "Bearer " + pair["access_token"],
            })
            assert accepted.status_code == 200
            assert "result" in accepted.json()


TAILSCALE_SERVE = {
    "Host": "example.com:10000",
    "X-Forwarded-For": "192.0.2.10",
    "X-Forwarded-Proto": "https",
    "Tailscale-User-Login": "owner@example.test",
    "Tailscale-Headers-Info": "https://tailscale.com/s/serve-headers",
}
LIST_TOOLS = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}


def _scope(peer: str, headers: dict[str, str] | None = None) -> dict:
    return {
        "type": "http",
        "path": "/",
        PRIVATE_INGRESS: True,
        "client": (peer, 50000),
        "server": ("127.0.0.1", 8765),
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
    }


@pytest.mark.parametrize("peer", ["127.0.0.1", "::1"])
def test_exact_loopback_peers_keep_the_direct_bypass(peer: str) -> None:
    assert _is_direct_loopback_mcp(_scope(peer)) is True


@pytest.mark.parametrize("peer", ["127.0.0.2", "127.1.2.3", "::ffff:127.0.0.1", "testclient", "10.0.0.5"])
def test_other_peers_are_not_direct_loopback_clients(peer: str) -> None:
    assert _is_direct_loopback_mcp(_scope(peer)) is False


@pytest.mark.parametrize("peer", ["127.0.0.1", "::1"])
def test_exact_loopback_peers_keep_the_tailnet_identity(peer: str) -> None:
    assert _is_verified_tailnet_mcp(_scope(peer, TAILSCALE_SERVE)) is True


@pytest.mark.parametrize("peer", ["127.0.0.2", "127.1.2.3"])
def test_tailnet_identity_needs_an_exact_loopback_peer(peer: str) -> None:
    assert _is_verified_tailnet_mcp(_scope(peer, TAILSCALE_SERVE)) is False


def test_serve_headers_from_127_0_0_2_are_routed_to_oauth(vg_home, monkeypatch) -> None:
    monkeypatch.setenv("VECGREP_OAUTH_ENABLED", "1")
    monkeypatch.setenv("VECGREP_OAUTH_ISSUER_URL", TEST_ISSUER)
    monkeypatch.setenv("VECGREP_OAUTH_APPROVAL_TOKEN", TEST_APPROVAL)
    monkeypatch.setenv("VECGREP_API_TOKEN", "r" * 40)
    monkeypatch.setattr(cfg_mod, "_settings", None)
    monkeypatch.setattr(mcp_server, "_PROVIDER", None)
    with TestClient(create_app(), base_url="http://127.0.0.1:8765",
                    client=("127.0.0.2", 50000)) as client:
        served = client.post("/mcp", json=LIST_TOOLS, headers=MCP_HEADERS | TAILSCALE_SERVE)
        assert served.status_code == 401
        assert served.headers["www-authenticate"].startswith("Bearer ")
        unproxied = client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS)
        assert unproxied.status_code == 401


def test_serve_headers_from_ipv6_loopback_keep_the_tailnet_bypass(vg_home, monkeypatch) -> None:
    monkeypatch.setenv("VECGREP_OAUTH_ENABLED", "1")
    monkeypatch.setenv("VECGREP_OAUTH_ISSUER_URL", TEST_ISSUER)
    monkeypatch.setenv("VECGREP_OAUTH_APPROVAL_TOKEN", TEST_APPROVAL)
    monkeypatch.setenv("VECGREP_API_TOKEN", "r" * 40)
    monkeypatch.setattr(cfg_mod, "_settings", None)
    monkeypatch.setattr(mcp_server, "_PROVIDER", None)
    with TestClient(create_app(), base_url="http://127.0.0.1:8765",
                    client=("::1", 50000)) as client:
        assert client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS).status_code == 200
        served = client.post("/mcp", json=LIST_TOOLS, headers=MCP_HEADERS | TAILSCALE_SERVE)
        assert served.status_code == 200

@pytest.mark.parametrize("path", ["/mcp", "/mcp/"])
def test_configured_browser_origin_can_discover_oauth_without_bypassing_it(vg_home, monkeypatch, path):
    origin = "https://client.example"
    monkeypatch.setenv("VECGREP_OAUTH_ENABLED", "1")
    monkeypatch.setenv("VECGREP_OAUTH_ISSUER_URL", TEST_ISSUER)
    monkeypatch.setenv("VECGREP_OAUTH_APPROVAL_TOKEN", TEST_APPROVAL)
    monkeypatch.setenv("VECGREP_MCP_ALLOWED_ORIGINS", origin)
    monkeypatch.setattr(cfg_mod, "_settings", None)
    monkeypatch.setattr(mcp_server, "_PROVIDER", None)
    with TestClient(create_app(), base_url="https://example.com:10000", client=("198.51.100.9", 1234)) as client:
        preflight = client.options(path, headers={"Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type,mcp-protocol-version"})
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == origin
        denied = client.post(path, json=INITIALIZE, headers=MCP_HEADERS | {"Origin": origin})
        assert denied.status_code == 401
        assert denied.headers["access-control-allow-origin"] == origin
        assert "www-authenticate" in denied.headers["access-control-expose-headers"].lower()
        assert "resource_metadata=" in denied.headers["www-authenticate"]
        foreign = client.options(path, headers={"Origin": "https://unlisted.example",
            "Access-Control-Request-Method": "POST"})
        assert foreign.status_code == 400
        assert "access-control-allow-origin" not in foreign.headers


@pytest.mark.parametrize("ro_path", ["/mcp/ro", "/mcp?mode=ro"])
def test_oauth_read_only_route_shares_auth_and_hides_writes(
    loopback_client: TestClient,
    ro_path: str,
) -> None:
    # Unauthenticated proxied request to /mcp/ro requires OAuth
    unauth = loopback_client.post(
        ro_path,
        json=INITIALIZE,
        headers=MCP_HEADERS | {"X-Forwarded-For": "203.0.113.10"},
    )
    assert unauth.status_code == 401
    assert unauth.headers["www-authenticate"].startswith("Bearer ")

    # Protected resource metadata is served for /mcp/ro as well
    meta_ro = loopback_client.get(
        "/.well-known/oauth-protected-resource/mcp/ro",
        headers={"X-Forwarded-For": "203.0.113.10"},
    )
    assert meta_ro.status_code == 200
    assert meta_ro.json()["resource"] == f"{TEST_ISSUER}/ro"
    assert meta_ro.json()["scopes_supported"] == ["read"]

    access = mcp_server._shared_provider().store.issue_access_token(
        "ro-oauth-client", ["read"], ttl_s=60, resource=TEST_ISSUER
    )
    auth_headers = MCP_HEADERS | {
        "X-Forwarded-For": "203.0.113.10",
        "Authorization": f"Bearer {access.token}",
    }
    listed = loopback_client.post(ro_path, json=LIST_TOOLS, headers=auth_headers)
    assert listed.status_code == 200
    tools = listed.json()["result"]["tools"]
    assert {t["name"] for t in tools} == set(mcp_server._READ_ONLY_MCP_TOOLS)
    for t in tools:
        assert t["annotations"] == {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        }
        assert t["securitySchemes"] == [{"type": "oauth2", "scopes": ["read"]}]

    # Write tool call on /mcp/ro is rejected as disabled on read-only endpoint,
    # NOT as an insufficient_scope step-up challenge.
    denied = loopback_client.post(
        ro_path,
        headers=auth_headers,
        json={
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {
                "name": "propose_write",
                "arguments": {"content": "nope", "corpus": "notes"},
            },
        },
    )
    assert denied.status_code == 200
    res = denied.json()["result"]
    assert res["isError"] is True
    assert "mcp/www_authenticate" not in (res.get("_meta") or {})
    assert "disabled on the read-only MCP endpoint" in res["content"][0]["text"]

    # A Google connector token on base /mcp also gets strict read-only tools automatically
    from mcp.shared.auth import OAuthClientInformationFull
    from pydantic import AnyUrl
    g_uri = AnyUrl("https://oauth-redirect.googleusercontent.com/r/custom-mcp")
    store = mcp_server._shared_provider().store
    store.save_client(OAuthClientInformationFull(
        client_id="google-ro-client", redirect_uris=[g_uri], scope="read propose",
    ))
    g_access, _ = store.issue_token_pair("google-ro-client", ["read", "propose"], resource=TEST_ISSUER)
    g_headers = MCP_HEADERS | {
        "X-Forwarded-For": "203.0.113.10",
        "Authorization": f"Bearer {g_access.token}",
    }
    g_listed = loopback_client.post("/mcp", json=LIST_TOOLS, headers=g_headers)
    assert g_listed.status_code == 200
    assert {t["name"] for t in g_listed.json()["result"]["tools"]} == set(mcp_server._READ_ONLY_MCP_TOOLS)
    g_denied = loopback_client.post(
        "/mcp",
        headers=g_headers,
        json={
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "propose_write",
                "arguments": {"content": "nope", "corpus": "notes"},
            },
        },
    )
    assert g_denied.status_code == 200
    assert g_denied.json()["result"]["isError"] is True
    assert "disabled on the read-only MCP endpoint" in g_denied.json()["result"]["content"][0]["text"]
