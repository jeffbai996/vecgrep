"""MCP server transport-construction tests.

Covers only the wiring done in vecgrep/mcp/server.py — the actual tool
behaviour rides on the service tests. Skipped when the optional `mcp`
extra isn't installed, since these imports would fail at collection.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

mcp = pytest.importorskip("mcp")

from vecgrep.mcp.server import build_http_app, build_mcp_server  # noqa: E402


def test_build_mcp_server_registers_tools() -> None:
    server = build_mcp_server()
    # The Server class stores the registered list_tools handler under
    # request_handlers keyed by the typed request class. Resolving the
    # tools means actually invoking that handler.
    from mcp.types import ListToolsRequest

    handler = server.request_handlers[ListToolsRequest]
    # Build a minimal request shaped like what the runtime would pass.
    req = ListToolsRequest(method="tools/list")
    result = asyncio.run(handler(req))
    tools = {t.name for t in result.root.tools}
    assert tools == {"search", "timeline", "incident", "get_chunk", "list_corpora", "get_corpus"}


def test_build_http_app_returns_asgi_app() -> None:
    app = build_http_app()
    # ASGI callable contract: __call__(scope, receive, send). Starlette
    # apps satisfy this. Checking the signature is enough — actually
    # invoking would need a running event loop and a real scope.
    assert callable(app)
    sig = inspect.signature(app.__call__)
    assert list(sig.parameters)[:3] == ["scope", "receive", "send"]
