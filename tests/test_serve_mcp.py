"""Verifies the /mcp HTTP endpoint mount on the FastAPI app.

We only check the auth gate and a single initialize handshake — full
session lifecycle behaviour is the mcp SDK's responsibility.
"""
from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

pytest.importorskip("mcp")

from fastapi.testclient import TestClient

from vecgrep.backend import config as cfg_mod
from vecgrep.backend.main import create_app


@pytest.fixture
def client_with_token(vg_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("VECGREP_API_TOKEN", "test-token")
    monkeypatch.setattr(cfg_mod, "_settings", None)
    app = create_app()
    # TestClient's context manager enters the FastAPI lifespan — which
    # is exactly when the MCP session manager spins up. Without `with`,
    # streamable HTTP requests would 500 on missing task group.
    with TestClient(
        app,
        base_url="http://127.0.0.1:8765",
        client=("127.0.0.1", 50000),
    ) as client:
        yield client


def test_mcp_open_when_oauth_off(client_with_token: TestClient) -> None:
    # The static-token gate on /mcp was RETIRED (OAuth replaces it; /mcp is
    # network-trusted when OAuth is off — reached over localhost/tailnet/secret
    # funnel path). So an unauthenticated POST is no longer a flat 401; it
    # reaches the MCP handler (which may 400 on a malformed body, but is NOT a
    # static-token 401). Auth-on behavior is covered once OAuth mounting lands.
    resp = client_with_token.post("/mcp", json={})
    assert resp.status_code != 401  # static-token gate is gone


def test_mcp_endpoint_responds_to_initialize(client_with_token: TestClient) -> None:
    # Minimal but spec-valid initialize payload. The MCP streamable HTTP
    # transport requires both content types in Accept.
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "0"},
        },
    }
    resp = client_with_token.post(
        "/mcp",
        headers={
            "Authorization": "Bearer test-token",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
        json=payload,
    )
    assert resp.status_code == 200, resp.text
    # Streamable HTTP uses SSE for the response by default. We only
    # verify the framing is sane and the body carries an initialize
    # result — not the full client handshake.
    body = resp.text
    assert "result" in body
    # Either pure JSON or an SSE event with a JSON data line.
    if resp.headers.get("content-type", "").startswith("text/event-stream"):
        data_lines = [
            line[len("data: ") :]
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        assert data_lines, f"no SSE data line in body: {body!r}"
        parsed = json.loads(data_lines[0])
    else:
        parsed = resp.json()
    assert parsed.get("id") == 1
    assert "result" in parsed
    assert "protocolVersion" in parsed["result"]


def test_explicit_private_proxy_host_and_browser_origin_remain_supported(
    vg_home, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VECGREP_MCP_ALLOWED_HOSTS", "private-proxy.example:8443")
    monkeypatch.setenv("VECGREP_MCP_ALLOWED_ORIGINS", "https://app.example")
    monkeypatch.setattr(cfg_mod, "_settings", None)
    with TestClient(
        create_app(),
        base_url="https://private-proxy.example:8443",
        client=("127.0.0.1", 50000),
    ) as client:
        response = client.post(
            "/mcp",
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "Origin": "https://app.example",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "0"},
                },
            },
        )
    assert response.status_code == 200


def test_search_policy_over_http(client_with_token, monkeypatch):
    from vecgrep.mcp import server
    calls = []

    class Service:
        def search_budgeted_with_diagnostics(self, query, **kwargs):
            calls.append(kwargs)
            return [], [], []

    monkeypatch.setattr(server, '_svc', lambda: Service())
    headers = {'Accept': 'application/json, text/event-stream'}
    response = client_with_token.post('/mcp', headers=headers, json={
        'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}})
    search = next(t for t in response.json()['result']['tools'] if t['name'] == 'search')
    assert 'profile' in search['inputSchema']['properties']
    assert search['annotations']['readOnlyHint'] is True
    assert search['annotations']['openWorldHint'] is False
    assert 'response_token_ceiling' in search['inputSchema']['properties']
    response = client_with_token.post('/mcp', headers=headers, json={
        'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {
            'name': 'search', 'arguments': {'query': 'example', 'corpus': 'notes',
                                           'profile': 'explore'}}})
    result = response.json()['result']
    assert not result.get('isError'), result
    payload = json.loads(result['content'][0]['text'])
    assert calls[0]['full_k'] == 6  # wrapper defaults must not override preset
    assert calls[0]['corpus_name'] == 'notes'
    assert payload['search_info']['ignored_parameters'] == []
    assert payload['search_info']['response_token_ceiling'] == 8000


@pytest.mark.parametrize("ro_path", ["/mcp/ro", "/mcp?mode=ro"])
def test_mcp_read_only_route_filters_tools_and_blocks_writes(
    client_with_token: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    ro_path: str,
) -> None:
    from vecgrep.mcp import server

    monkeypatch.setattr(server, "_run_list_corpora", lambda include_hidden=False: '[{"name": "notes"}]')
    monkeypatch.setattr(server, "_run_list_aliases", lambda: '{"gpu": ["3090"]}')

    headers = {"Accept": "application/json, text/event-stream"}
    resp = client_with_token.post(
        ro_path,
        headers=headers,
        json={"jsonrpc": "2.0", "id": 10, "method": "tools/list", "params": {}},
    )
    assert resp.status_code == 200, resp.text
    tools = resp.json()["result"]["tools"]
    names = {t["name"] for t in tools}
    expected_ro = {
        "search",
        "get_chunk",
        "get_source",
        "browse",
        "timeline",
        "incident",
        "related",
        "compare",
        "list_corpora",
        "get_corpus",
        "stats",
        "summarize_corpus",
        "list_aliases",
        "batch_read",
    }
    assert names == expected_ro
    forbidden = {
        "write",
        "write_temp",
        "renew_temp",
        "edit",
        "retire",
        "unretire",
        "propose_write",
        "propose_edit",
        "propose_delete",
        "propose_merge",
    }
    assert names.isdisjoint(forbidden)
    for t in tools:
        assert list(t["annotations"].items()) == [
            ("readOnlyHint", True),
            ("destructiveHint", False),
            ("idempotentHint", True),
            ("openWorldHint", False),
        ]

    # batch_read executes multiple read-only tools in one round-trip
    batch_resp = client_with_token.post(
        ro_path,
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "batch_read",
                "arguments": {
                    "calls": [
                        {"tool": "list_corpora", "arguments": {}},
                        {"tool": "list_aliases"},
                    ]
                },
            },
        },
    )
    assert batch_resp.status_code == 200
    batch_result = batch_resp.json()["result"]
    assert not batch_result.get("isError"), batch_result
    batch_payload = json.loads(batch_result["content"][0]["text"])
    assert batch_payload["results"] == [
        {"tool": "list_corpora", "result": [{"name": "notes"}]},
        {"tool": "list_aliases", "result": {"gpu": ["3090"]}},
    ]

    # Any mutating or propose tool call is rejected on /mcp/ro and ?mode=ro
    for blocked in ("write", "propose_write", "edit", "retire"):
        write_resp = client_with_token.post(
            ro_path,
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 12,
                "method": "tools/call",
                "params": {
                    "name": blocked,
                    "arguments": {"content": "should fail"},
                },
            },
        )
        assert write_resp.status_code == 200
        write_result = write_resp.json()["result"]
        assert write_result["isError"] is True
        assert "disabled on the read-only MCP endpoint" in write_result["content"][0]["text"]
