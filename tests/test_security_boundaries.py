"""Security boundary tests for URLAdapter SSRF, remote /api/index, and MCP cross-origin bypass."""
from __future__ import annotations

import asyncio
import pytest

from vecgrep.backend.ingestion.adapters.base import AdapterError
from vecgrep.backend.ingestion.adapters.url import _assert_public_url
from vecgrep.mcp.server import _McpOAuthBoundary


def test_url_adapter_blocks_loopback_private_and_dns_rebind():
    with pytest.raises(AdapterError):
        _assert_public_url("http://127.0.0.1:22019/config/")
    with pytest.raises(AdapterError):
        _assert_public_url("http://localhost:8001/api/query")
    with pytest.raises(AdapterError):
        _assert_public_url("http://10.0.0.5/admin")
    with pytest.raises(AdapterError):
        _assert_public_url("https://box.example.ts.net:8443/")
    with pytest.raises(AdapterError):
        _assert_public_url("file:///etc/passwd")

    priv_resolver = lambda host, port, type=0: [(2, 1, 6, "", ("127.0.0.1", 0))]
    with pytest.raises(AdapterError):
        _assert_public_url("https://rebind.example/doc", resolver=priv_resolver)

    pub_resolver = lambda host, port, type=0: [(2, 1, 6, "", ("93.184.216.34", 0))]
    _assert_public_url("https://public.example/doc", resolver=pub_resolver)


def test_mcp_oauth_boundary_refuses_tailscale_bypass_for_cross_origin_browser():
    called = []

    class _FakeApp:
        def __init__(self, label: str):
            self.label = label
            self.router = None

        async def __call__(self, scope, receive, send):
            called.append(self.label)

    boundary = _McpOAuthBoundary(
        _FakeApp("secured"),
        _FakeApp("direct"),
        tailscale_identity_bypass=True,
    )
    scope = {
        "type": "http",
        "path": "/mcp",
        "client": ("127.0.0.1", 54321),
        "headers": [
            (b"host", b"box.example.ts.net:8444"),
            (b"origin", b"https://gemini.google.com"),
            (b"x-forwarded-for", b"192.0.2.1"),
            (b"tailscale-user-login", b"owner@example.com"),
        ],
    }
    asyncio.run(boundary(scope, None, None))
    assert called == ["secured"]
