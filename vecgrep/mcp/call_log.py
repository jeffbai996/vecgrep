"""One INFO line per MCP tool call: the tool name, whether it raised, and how long it took.

The HTTP transport logs each CallToolRequest without the tool name, so usage
per tool was invisible. FastMCP sends every call through its ToolManager, so
wrapping that one method covers every tool on the HTTP endpoint.
"""
from __future__ import annotations

import logging
import time
from typing import Any

LOGGER_NAME = "vecgrep.mcp.calls"
logger = logging.getLogger(LOGGER_NAME)


def install(server: Any) -> None:
    """Log every tool call made through this FastMCP server. Safe to call twice."""
    manager = server._tool_manager
    inner = manager.call_tool
    if getattr(inner, "_call_logged", False):
        return

    async def call_tool(name: str, arguments: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        started = time.monotonic()
        outcome = "raised"
        try:
            result = await inner(name, arguments, *args, **kwargs)
            outcome = "ok"
            return result
        finally:
            logger.info("tool=%s outcome=%s ms=%d", name, outcome,
                        (time.monotonic() - started) * 1000)

    call_tool._call_logged = True  # type: ignore[attr-defined]
    manager.call_tool = call_tool
