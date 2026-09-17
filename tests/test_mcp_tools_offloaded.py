"""Every MCP tool runs off the event loop.

The MCP Python SDK calls a tool function directly when it is not a coroutine:

    mcp/server/fastmcp/utilities/func_metadata.py
        if fn_is_async:  return await fn(...)
        else:            return fn(...)

So a synchronous tool body runs ON the asyncio event loop, and uvicorn can
neither accept a connection nor answer anything while it runs. vecgrep's
search is seconds of embed + qdrant + sqlite + cross-encoder, so the whole
server went dark for the length of one search.

Measured 2026-09-12 against the live server: one MCP search took 17.90s and
`/api/health` — whose entire body is `return {"status": "ok"}` — was blocked
for 17.89s of it, against a 2ms baseline. Three of those in a row is three
watchdog strikes, which is a restart, which is the 504s Jeff saw.

The blocking call had been there all along; 2026-09-10's cf21ce9 (rerank
cross-corpus fan-out by default) made a routine search long enough to cross
the 5s health probe, and the restarts start the next morning.
"""
from __future__ import annotations

import asyncio
import inspect
import threading
import time

import pytest

pytest.importorskip("mcp")

from mcp.server.fastmcp import FastMCP  # noqa: E402

from vecgrep.mcp import server as mcp_server  # noqa: E402
from vecgrep.mcp.server import ToolBusyError  # noqa: E402


def _patched() -> FastMCP:
    fmcp = FastMCP("test")
    mcp_server._offload_sync_tools(fmcp)
    return fmcp


def test_a_sync_tool_is_registered_as_a_coroutine():
    fmcp = _patched()

    @fmcp.tool(description="sync body")
    def probe(query: str, top_k: int = 5) -> str:
        return f"{query}/{top_k}"

    tool = fmcp._tool_manager.get_tool("probe")
    assert tool is not None and tool.is_async, "tool still runs on the loop"


def test_the_signature_survives_so_the_schema_is_unchanged():
    """FastMCP derives each tool's JSON schema from the function signature, so
    the wrapper must not flatten it to (*args, **kwargs)."""
    fmcp = _patched()

    @fmcp.tool(description="sync body")
    def probe(query: str, corpus: str | None = None, top_k: int = 5) -> str:
        return query

    params = fmcp._tool_manager.get_tool("probe").parameters["properties"]
    assert sorted(params) == ["corpus", "query", "top_k"], sorted(params)


def test_an_async_tool_is_left_alone():
    fmcp = _patched()

    @fmcp.tool(description="already async")
    async def probe(query: str) -> str:
        return query

    assert fmcp._tool_manager.get_tool("probe").is_async


def test_a_slow_tool_does_not_block_the_event_loop():
    """The behaviour, not just the flag: while a tool sleeps for 400ms the
    loop must keep running other tasks."""
    fmcp = _patched()

    @fmcp.tool(description="sleeps, on purpose")
    def stall(ms: int = 400) -> str:
        time.sleep(ms / 1000)
        return "done"

    async def drive() -> int:
        ticks = 0

        async def heartbeat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        beat = asyncio.ensure_future(heartbeat())
        await fmcp._tool_manager.call_tool("stall", {"ms": 400})
        beat.cancel()
        return ticks

    ticks = asyncio.run(drive())
    assert ticks >= 5, (
        f"the loop only ticked {ticks} times during a 400ms tool call — "
        "the tool is running on the event loop")


def test_tool_calls_stay_serialized():
    """Today every tool runs one at a time because the loop runs them. Moving
    them to threads must not quietly turn vecgrep into a concurrent server:
    the search path's thread-safety was never designed for that."""
    fmcp = _patched()
    live = {"now": 0, "peak": 0}

    @fmcp.tool(description="records overlap")
    def overlap() -> str:
        live["now"] += 1
        live["peak"] = max(live["peak"], live["now"])
        time.sleep(0.1)
        live["now"] -= 1
        return "ok"

    async def drive():
        await asyncio.gather(*(fmcp._tool_manager.call_tool("overlap", {})
                               for _ in range(4)))

    asyncio.run(drive())
    assert live["peak"] == 1, f"{live['peak']} tool calls ran at once"


def test_the_real_server_applies_it_before_registering_anything():
    """The helper is useless if build_http_app forgets to call it."""
    src = inspect.getsource(mcp_server.build_http_app)
    assert "_offload_sync_tools(fmcp)" in src, \
        "build_http_app does not offload its tools"
    assert src.index("_offload_sync_tools(fmcp)") < src.index("@fmcp.tool("), \
        "tools are registered before the offload patch is applied"


# --------------------------------------------------------------------------
# One stuck tool must not blackhole every other MCP call.
#
# Tool bodies share a single-slot gate so the search path never meets the
# concurrency it was not written for. The wait on that gate had no ceiling:
# on 2026-09-16 a search parked on a saturated disk held it, and every MCP
# call after it -- list_corpora included, whose body does almost nothing --
# queued behind it until the service was restarted. REST was unaffected the
# whole time, which is why the server looked healthy from curl and dead from
# every agent.
# --------------------------------------------------------------------------


def _call(fmcp, name, **kwargs):
    return fmcp._tool_manager.get_tool(name).fn(**kwargs)


def test_a_stuck_tool_does_not_blackhole_the_next_call(monkeypatch):
    monkeypatch.setattr(mcp_server, "MCP_GATE_WAIT_S", 0.3)
    fmcp = _patched()
    stuck = threading.Event()
    entered = threading.Event()

    @fmcp.tool(description="never returns on its own")
    def hold() -> str:
        entered.set()
        stuck.wait(30)
        return "done"

    @fmcp.tool(description="cheap")
    def cheap() -> str:
        return "ok"

    async def scenario():
        held = asyncio.create_task(_call(fmcp, "hold"))
        await asyncio.to_thread(entered.wait, 5)
        started = time.monotonic()
        with pytest.raises(ToolBusyError):
            await _call(fmcp, "cheap")
        assert time.monotonic() - started < 3
        stuck.set()
        await held

    asyncio.run(scenario())


def test_the_gate_still_serializes_tool_bodies(monkeypatch):
    monkeypatch.setattr(mcp_server, "MCP_GATE_WAIT_S", 10)
    fmcp = _patched()
    overlap = []
    inside = []

    @fmcp.tool(description="records overlap")
    def slow(tag: str) -> str:
        inside.append(tag)
        overlap.append(len(inside))
        time.sleep(0.1)
        inside.remove(tag)
        return tag

    async def scenario():
        await asyncio.gather(_call(fmcp, "slow", tag="a"),
                             _call(fmcp, "slow", tag="b"))

    asyncio.run(scenario())
    assert overlap == [1, 1], "tool bodies ran concurrently"


def test_a_raising_tool_hands_the_gate_back(monkeypatch):
    monkeypatch.setattr(mcp_server, "MCP_GATE_WAIT_S", 0.3)
    fmcp = _patched()

    @fmcp.tool(description="raises")
    def boom() -> str:
        raise ValueError("nope")

    @fmcp.tool(description="cheap")
    def cheap() -> str:
        return "ok"

    async def scenario():
        with pytest.raises(ValueError):
            await _call(fmcp, "boom")
        assert await _call(fmcp, "cheap") == "ok"

    asyncio.run(scenario())
