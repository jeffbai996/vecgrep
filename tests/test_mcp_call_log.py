"""One log line per MCP tool call, naming the tool, on both transports."""
from __future__ import annotations

import asyncio
import logging

import pytest

pytest.importorskip("mcp")

from mcp.server.fastmcp import FastMCP  # noqa: E402

from vecgrep.mcp import call_log  # noqa: E402


def _server() -> FastMCP:
    server = FastMCP("call-log-test")

    @server.tool(name="echo")
    def echo(text: str) -> str:
        return text

    @server.tool(name="boom")
    def boom() -> str:
        raise RuntimeError("broken")

    return server


def _lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == call_log.LOGGER_NAME]


def test_http_call_is_logged_with_tool_name_and_outcome(caplog) -> None:
    server = _server()
    call_log.install(server)
    with caplog.at_level(logging.INFO, logger=call_log.LOGGER_NAME):
        asyncio.run(server.call_tool("echo", {"text": "hi"}))
    lines = _lines(caplog)
    assert len(lines) == 1 and lines[0].startswith("tool=echo outcome=ok ms=")


def test_failed_call_is_logged_as_raised(caplog) -> None:
    server = _server()
    call_log.install(server)
    with caplog.at_level(logging.INFO, logger=call_log.LOGGER_NAME):
        with pytest.raises(Exception):
            asyncio.run(server.call_tool("boom", {}))
    assert _lines(caplog)[0].startswith("tool=boom outcome=raised ms=")


def test_installing_twice_logs_once(caplog) -> None:
    server = _server()
    call_log.install(server)
    call_log.install(server)
    with caplog.at_level(logging.INFO, logger=call_log.LOGGER_NAME):
        asyncio.run(server.call_tool("echo", {"text": "hi"}))
    assert len(_lines(caplog)) == 1


def test_http_app_installs_the_call_log(monkeypatch) -> None:
    from vecgrep.mcp import server as mcp_server

    installed: list[object] = []
    monkeypatch.setattr(call_log, "install", installed.append)
    mcp_server.build_http_app()
    assert len(installed) == 1
