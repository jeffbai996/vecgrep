"""Verifies the /mcp HTTP endpoint mount on the FastAPI app.

We only check the auth gate and a single initialize handshake — full
session lifecycle behaviour is the mcp SDK's responsibility.
"""
from __future__ import annotations

import json
from typing import Iterator

import pytest

pytest.importorskip("mcp")

from fastapi.testclient import TestClient  # noqa: E402

from vecgrep.backend import config as cfg_mod  # noqa: E402
from vecgrep.backend.main import create_app  # noqa: E402


@pytest.fixture
def client_with_token(vg_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("VECGREP_API_TOKEN", "test-token")
    monkeypatch.setattr(cfg_mod, "_settings", None)
    app = create_app()
    # TestClient's context manager enters the FastAPI lifespan — which
    # is exactly when the MCP session manager spins up. Without `with`,
    # streamable HTTP requests would 500 on missing task group.
    with TestClient(app) as client:
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
