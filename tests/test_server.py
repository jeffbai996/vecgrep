from __future__ import annotations

import pytest

pytest.importorskip("mcp")

import asyncio
import inspect
import logging
import threading
import time

import pytest
from mcp.server.fastmcp import FastMCP

from vecgrep.mcp import call_log, server
from vecgrep.mcp import server as S
from vecgrep.mcp import server as mcp_server
from vecgrep.mcp.server import ToolBusyError, build_http_app, build_mcp_server

# ── test_registration_ingress_replay ──

def test_registration_replay_hands_receive_back_to_the_client_after_the_body():
    """After the buffered body the app must see the client's own receive(), so a
    loop waiting for http.disconnect blocks instead of spinning on synthetic
    empty bodies and starving the event loop."""
    import asyncio

    from vecgrep.mcp.server import _RegistrationIngress

    seen: list[dict] = []

    async def inner(scope, receive, send):
        seen.append(await receive())          # the replayed body
        seen.append(await receive())          # must come from the client
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    messages = iter([
        {"type": "http.request", "body": b'{"client_name": "t"}', "more_body": False},
        {"type": "http.disconnect"},
    ])

    async def receive():
        return next(messages)

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/register",
             "headers": [(b"content-length", b"19")]}
    asyncio.run(_RegistrationIngress(inner)(scope, receive, send))
    assert seen[0]["body"] == b'{"client_name": "t"}'
    assert seen[1] == {"type": "http.disconnect"}
    assert sent[0]["status"] == 200

# ── test_mcp_call_log ──

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
    with (
        caplog.at_level(logging.INFO, logger=call_log.LOGGER_NAME),
        pytest.raises(Exception, match="broken"),
    ):
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

# ── test_registration_rate_limit ──

def test_one_caller_spending_its_budget_does_not_block_another():
    limiter = server._RegistrationRateLimiter(limit=3, window_s=60)
    assert [limiter.allow("a")[0] for _ in range(3)] == [True, True, True]
    allowed, retry_after = limiter.allow("a")
    assert allowed is False and retry_after >= 1
    assert limiter.allow("b") == (True, 0)


def test_the_caller_table_is_bounded():
    limiter = server._RegistrationRateLimiter(limit=1, window_s=60, max_keys=4)
    for i in range(10):
        limiter.allow(f"k{i}")
    assert len(limiter._attempts) <= 4
    assert limiter.allow("k9")[0] is False, "the newest caller keeps its count"


async def _drive(app, forwarded: str) -> int:
    sent: list[dict] = []
    scope = {"type": "http", "method": "POST", "path": "/register",
             "headers": [(b"x-forwarded-for", forwarded.encode()), (b"content-length", b"2")],
             "client": ("127.0.0.1", 50000)}
    messages = iter([{"type": "http.request", "body": b"{}", "more_body": False}])

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


def test_registrations_through_the_ingress_are_counted_per_forwarded_address(monkeypatch):
    monkeypatch.setattr(server, "_REGISTRATION_LIMITER", server._RegistrationRateLimiter(limit=2, window_s=60))

    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 201, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    app = server._RegistrationIngress(inner)
    first = [asyncio.run(_drive(app, "198.51.100.7")) for _ in range(3)]
    assert first[2] == 429
    assert asyncio.run(_drive(app, "203.0.113.9")) != 429


def test_the_default_budget_is_six_registrations_an_hour_per_caller(monkeypatch):
    """A connector registers once; an hour-long window keeps one address from
    churning registrations in minutes (funnel review 2026-11-02)."""
    import time
    clock = [1_000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    limiter = server._RegistrationRateLimiter()
    assert [limiter.allow("a")[0] for _ in range(6)] == [True] * 6
    clock[0] += 59 * 60
    assert limiter.allow("a")[0] is False
    clock[0] += 2 * 60
    assert limiter.allow("a")[0] is True

# ── test_mcp_server ──

mcp = pytest.importorskip("mcp")



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
    assert tools == {
        "search", "timeline", "incident", "browse", "get_chunk", "get_source",
        "list_aliases", "list_corpora", "get_corpus",
        # v1.2 insight tools
        "related", "compare", "stats", "summarize_corpus",
        # batch_read single round-trip read fan-out
        "batch_read",
        # write path — parity with the HTTP transport as of v1.1
        "propose_write", "propose_edit", "propose_delete", "propose_merge",
        # direct (unconfirmed) write path — single operator-designated corpus,
        # inert unless VECGREP_DIRECT_WRITE_CORPUS is set
        "write", "edit",
        # temp corpus: expiring direct writes, inert unless VECGREP_TEMP_CORPUS
        "write_temp", "renew_temp",
        # retire: stale docs leave search, same corpus as direct write
        "retire", "unretire",
    }
    search = next(t for t in result.root.tools if t.name == "search")
    assert search.inputSchema["properties"]["corpora"]["minItems"] == 1
    assert "list_corpora" in search.description
    for tool in result.root.tools:
        if tool.name in {"search", "timeline", "incident", "browse", "get_chunk", "get_source",
                         "list_aliases", "list_corpora", "get_corpus", "related", "compare", "stats",
                         "summarize_corpus", "batch_read"}:
            assert tool.annotations.readOnlyHint is True
            assert tool.annotations.destructiveHint is False
            assert tool.annotations.idempotentHint is True
            assert tool.annotations.openWorldHint is False
        else:
            assert tool.annotations.readOnlyHint is False
            assert tool.annotations.openWorldHint is True


def test_build_http_app_returns_asgi_app() -> None:
    app = build_http_app()
    # ASGI callable contract: __call__(scope, receive, send). Starlette
    # apps satisfy this. Checking the signature is enough — actually
    # invoking would need a running event loop and a real scope.
    assert callable(app)
    sig = inspect.signature(app.__call__)
    assert list(sig.parameters)[:3] == ["scope", "receive", "send"]

# ── test_auto_modes ──

class _C:
    def __init__(self, name, chunk_count):
        self.name = name
        self.chunk_count = chunk_count


class _Svc:
    def list_corpora(self):
        return [_C("big-a", 103_339), _C("big-b", 258_955),
                _C("small-a", 2_436), _C("small-b", 714)]


SVC = _Svc()


@pytest.mark.parametrize("gate", [S._should_rerank, S._should_budget])
@pytest.mark.parametrize("corpus,want", [
    ("big-a", True), ("big-b", True),        # big -> on
    ("small-a", False), ("small-b", False),  # small -> off
])
def test_auto_by_corpus_size(gate, corpus, want):
    assert gate(SVC, {"corpus": corpus}) is want


@pytest.mark.parametrize("gate,key", [(S._should_rerank, "rerank"),
                                      (S._should_budget, "budget")])
def test_explicit_false_beats_auto_on_big_corpus(gate, key):
    # The regression that made the rerank gate a no-op: an explicit False
    # must win, which is exactly why the wrapper must default to None.
    assert gate(SVC, {"corpus": "big-a", key: False}) is False


@pytest.mark.parametrize("gate,key", [(S._should_rerank, "rerank"),
                                      (S._should_budget, "budget")])
def test_explicit_true_beats_auto_on_small_corpus(gate, key):
    assert gate(SVC, {"corpus": "small-a", key: True}) is True


def test_no_corpus_does_not_turn_budget_on_blind():
    # Breadth mode stays scoped-only: see the cross-corpus section below.
    assert S._should_budget(SVC, {}) is False


@pytest.mark.parametrize("gate", [S._should_rerank, S._should_budget])
def test_unreadable_corpus_falls_back_to_off(gate):
    class Boom:
        def list_corpora(self):
            raise RuntimeError("registry down")
    assert gate(Boom(), {"corpus": "big-a"}) is False




# ── cross-corpus fan-out ─────────────────────────────────────────────────────
# The default call names no corpus. It used to skip rerank outright
# ("don't pay it blind"), which meant the highest-traffic call path never got
# the calibration the reranker exists for -- and the eval showed unreranked
# hybrid surfacing a confident top hit for almost every off-topic query. The
# fan-out merges every corpus into ONE candidate pool before the rerank step
# (service.search_with_diagnostics: filter -> dedup -> rerank), so a
# cross-corpus rerank is a single cross-encoder pass over the fused top-50,
# not N per-corpus passes. Sizing it on the summed searchable chunk_count is
# therefore the honest analogue of the single-corpus rule.


class _SvcSearchable:
    """list_corpora() is unfiltered (backup/eval read it); the fan-out uses
    _searchable_corpora(). The gate must size on what actually gets searched,
    so `hidden` is in the registry listing but never in the fan-out."""

    def __init__(self, searchable, hidden=()):
        self._searchable = list(searchable)
        self._hidden = list(hidden)

    def list_corpora(self):
        return self._searchable + self._hidden

    def _searchable_corpora(self):
        return self._searchable


def test_excluded_corpus_does_not_inflate_the_fanout_estimate():
    """An eval-* build copy is in the registry but out of the fan-out. Sizing
    on list_corpora() would rerank a tiny unscoped search on its strength."""
    svc = _SvcSearchable([_C("small-a", 700)], hidden=[_C("eval-big", 300_000)])
    assert S._should_rerank(svc, {}) is False


def test_no_corpus_reranks_when_searchable_corpora_are_big_enough():
    svc = _SvcSearchable([_C("small-a", 6_000), _C("small-b", 5_000)])
    # Neither corpus clears the bar alone; the fused pool they produce does.
    assert S._should_rerank(svc, {}) is True


def test_no_corpus_skips_rerank_when_searchable_corpora_are_tiny():
    svc = _SvcSearchable([_C("small-a", 700), _C("small-b", 300)])
    assert S._should_rerank(svc, {}) is False


def test_empty_corpora_list_argument_is_rejected_not_reranked():
    # `corpora: []` is an error at the service layer, not "search everything".
    svc = _SvcSearchable([_C("big-a", 103_339)])
    assert S._should_rerank(svc, {"corpora": []}) is False


def test_corpora_list_reranks_on_summed_chunk_count():
    svc = _SvcSearchable([_C("big-a", 103_339), _C("small-a", 6_000),
                          _C("small-b", 5_000)])
    assert S._should_rerank(svc, {"corpora": ["small-a", "small-b"]}) is True


def test_corpora_list_of_small_corpora_does_not_rerank():
    svc = _SvcSearchable([_C("big-a", 103_339), _C("small-a", 700),
                          _C("small-b", 300)])
    assert S._should_rerank(svc, {"corpora": ["small-a", "small-b"]}) is False


def test_explicit_false_still_beats_auto_on_cross_corpus():
    svc = _SvcSearchable([_C("big-a", 103_339)])
    assert S._should_rerank(svc, {"rerank": False}) is False


def test_explicit_true_still_beats_auto_on_tiny_cross_corpus():
    svc = _SvcSearchable([_C("small-b", 12)])
    assert S._should_rerank(svc, {"rerank": True}) is True


def test_unreadable_searchable_set_falls_back_to_off():
    class Boom:
        def _searchable_corpora(self):
            raise RuntimeError("registry down")

        def list_corpora(self):
            raise RuntimeError("registry down")

    assert S._should_rerank(Boom(), {}) is False


def test_service_without_searchable_helper_falls_back_to_list_corpora():
    # Older service objects (and the API dict mirror) expose only list_corpora.
    assert S._should_rerank(SVC, {}) is True


def test_budget_still_does_not_fire_blind_on_cross_corpus():
    """Breadth mode is deliberately NOT extended to unscoped search.

    Rerank only reorders a pool the caller already gets; budget changes the
    RESPONSE SHAPE (full hits + a `stubs` tail), so turning it on for every
    unscoped call would silently rewrite what every existing consumer reads.
    Same size rule, different blast radius -- keep it scoped-only."""
    svc = _SvcSearchable([_C("big-a", 103_339), _C("big-b", 258_955)])
    assert S._should_budget(svc, {}) is False
    assert S._should_budget(svc, {"corpora": ["big-a"]}) is False




def test_http_search_tool_defaults_preserve_auto_gates(monkeypatch) -> None:
    import inspect

    from mcp.server.fastmcp import FastMCP

    created: list[FastMCP] = []
    orig_init = FastMCP.__init__

    def capture(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(FastMCP, "__init__", capture)
    S.build_http_app()
    assert created
    tool = created[-1]._tool_manager.get_tool("search")
    assert tool is not None
    params = inspect.signature(tool.fn).parameters
    assert params["budget"].default is None
    assert params["rerank"].default is None
    assert params["include_superseded"].default is False

# ── test_mcp_tools_offloaded ──

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

    tool = fmcp._tool_manager.get_tool("probe")
    assert tool is not None
    params = tool.parameters["properties"]
    assert sorted(params) == ["corpus", "query", "top_k"], sorted(params)


def test_an_async_tool_is_left_alone():
    fmcp = _patched()

    @fmcp.tool(description="already async")
    async def probe(query: str) -> str:
        return query

    tool = fmcp._tool_manager.get_tool("probe")
    assert tool is not None
    assert tool.is_async


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


def test_the_real_server_offloads_every_registered_tool(monkeypatch) -> None:
    created: list[FastMCP] = []
    orig_init = FastMCP.__init__

    def capture(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(FastMCP, "__init__", capture)
    mcp_server.build_http_app()
    assert created
    tools = created[-1]._tool_manager.list_tools()
    assert tools
    assert all(t.is_async for t in tools)
