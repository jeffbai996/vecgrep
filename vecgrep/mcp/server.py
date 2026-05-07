"""MCP server: expose vecgrep as a tool for Claude / Cursor / any MCP client.

Two transports share the same configured Server:

* stdio  — `vecgrep mcp` (default for desktop/CLI MCP clients)
* HTTP   — mounted at /mcp by `vecgrep serve` for remote MCP clients

The MCP SDK is gated behind the `vecgrep[mcp]` extra so the base install
stays slim. Imports stay lazy here on purpose: importing this module must
not require the extra to be installed (the CLI catches the RuntimeError
at the call site).

Tools:
    search          — semantic search across one or all corpora
    list_corpora    — list every persisted corpus and its stats
    get_corpus      — get details for one corpus by name
"""
from __future__ import annotations

import json
from typing import Any

from ..backend.service import VecgrepService


def _require_mcp() -> Any:
    """Lazy-import the mcp SDK pieces we need. Centralised so both the
    stdio and HTTP entrypoints raise the same actionable error when the
    optional extra is missing.
    """
    try:
        from mcp.server import Server
        from mcp.types import TextContent, Tool
    except ImportError as e:
        raise RuntimeError(
            "MCP server requires the 'mcp' extra. "
            "Install with `pip install vecgrep[mcp]`."
        ) from e
    return Server, TextContent, Tool


def build_mcp_server() -> Any:
    """Construct the configured MCP Server with the three tools registered.

    Transport-agnostic: callers pick stdio or HTTP and wrap accordingly.
    """
    Server, TextContent, Tool = _require_mcp()

    server = Server("vecgrep")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="search",
                description=(
                    "Semantic search across vecgrep corpora. Returns ranked chunks with "
                    "surrounding context. Use this instead of dumping documents into "
                    "context — index once, search per question."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language query.",
                        },
                        "corpus": {
                            "type": "string",
                            "description": "Limit to one corpus. Omit to search all corpora.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Max results (default 5).",
                            "default": 5,
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["hybrid", "vector", "bm25"],
                            "description": "Retrieval strategy. Default: hybrid (BM25+vector via RRF).",
                            "default": "hybrid",
                        },
                        "rerank": {
                            "type": "boolean",
                            "description": "Cross-encoder rerank top candidates. Slower, more accurate.",
                            "default": False,
                        },
                        "filters": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional filter expressions. Forms: 'source:<glob>', "
                                "'corpus:<name>', 'meta.<key>=<value>'. All ANDed."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="list_corpora",
                description="List every vecgrep corpus and its stats (doc count, chunk count, embedding model).",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="get_corpus",
                description="Get full metadata for one corpus including its source list.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                    "required": ["name"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None) -> list[TextContent]:
        args = arguments or {}
        svc = VecgrepService()

        if name == "search":
            results = svc.search(
                query=args["query"],
                corpus_name=args.get("corpus"),
                top_k=args.get("top_k"),
                mode=args.get("mode", "hybrid"),
                rerank=bool(args.get("rerank", False)),
                filters=args.get("filters") or None,
            )
            payload = [
                {
                    "similarity_pct": round(r.similarity_pct, 1),
                    "corpus": r.corpus,
                    "source_id": r.source_id,
                    "matched_by": r.matched_by,
                    "chunk": r.chunk,
                    "context_before": r.context_before,
                    "context_after": r.context_after,
                }
                for r in results
            ]
            return [TextContent(type="text", text=json.dumps(payload, indent=2))]

        if name == "list_corpora":
            corpora = [
                {
                    "name": c.name,
                    "embed_backend": c.embed_backend,
                    "embed_model": c.embed_model,
                    "doc_count": c.doc_count,
                    "chunk_count": c.chunk_count,
                    "chunker": c.chunker,
                }
                for c in svc.list_corpora()
            ]
            return [TextContent(type="text", text=json.dumps(corpora, indent=2))]

        if name == "get_corpus":
            target = args.get("name")
            for c in svc.list_corpora():
                if c.name == target:
                    detail = {
                        "name": c.name,
                        "embed_backend": c.embed_backend,
                        "embed_model": c.embed_model,
                        "dim": c.dim,
                        "chunker": c.chunker,
                        "doc_count": c.doc_count,
                        "chunk_count": c.chunk_count,
                        "sources": c.sources,
                        "created_at": c.created_at,
                        "updated_at": c.updated_at,
                    }
                    return [TextContent(type="text", text=json.dumps(detail, indent=2))]
            return [TextContent(type="text", text=f"corpus not found: {target}")]

        return [TextContent(type="text", text=f"unknown tool: {name}")]

    return server


def build_http_app() -> Any:
    """Wrap the configured MCP server in a Starlette ASGI app speaking
    streamable HTTP.

    Mount this app under a path on the FastAPI server (see
    `backend/main.py`). The lifespan context drives the session manager
    — it's required for streamable HTTP to clean up sessions on shutdown.

    Each call returns a fresh Starlette app + session manager pair.
    StreamableHTTPSessionManager is documented as not reusable across
    `.run()` cycles, so building per-app is the safe pattern.
    """
    _require_mcp()  # surface the helpful error before importing the rest
    import contextlib
    from collections.abc import AsyncIterator

    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.routing import Mount

    server = build_mcp_server()
    # stateless=True: each request is independent, no server-side session
    # state to coordinate across the FastAPI process. Simpler for a
    # multi-tenant remote endpoint and avoids tying up resources on a
    # client that disconnects without sending DELETE.
    session_manager = StreamableHTTPSessionManager(app=server, stateless=True)

    async def handle(scope: Any, receive: Any, send: Any) -> None:
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        # session_manager.run() is the documented way to start/stop the
        # background task group that handles transport sessions.
        async with session_manager.run():
            yield

    # Mount at "/" because the parent FastAPI app already mounts this
    # whole Starlette app at /mcp — double-prefixing would give /mcp/mcp.
    return Starlette(
        routes=[Mount("/", app=handle)],
        lifespan=lifespan,
    )


def run() -> None:
    """Entry point for `vecgrep mcp` (stdio transport)."""
    import asyncio

    from mcp.server.stdio import stdio_server

    server = build_mcp_server()

    async def main() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(main())
