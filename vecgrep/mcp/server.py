"""MCP server: expose vecgrep as a tool for Claude / Cursor / any MCP client.

Two transports share the same tool logic:

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

HTTP transport notes
--------------------
The HTTP path uses FastMCP with json_response=True + stateless_http=True.
This makes the server respond with plain JSON (not SSE) so remote clients
like claude.ai work correctly. Their initial probe sends only
"Accept: application/json"; the StreamableHTTPSessionManager-based
approach returned text/event-stream and the client got either a 406 or
a stalled connection.

    json_response=True    → respond with application/json, not SSE
    stateless_http=True   → no server-side session state; each request
                            is independent. Prevents stale session ID
                            bugs and avoids tying up resources when a
                            client disconnects without sending DELETE.
    dns rebinding off     → requests arrive through Tailscale Funnel with
                            a non-localhost Host header; the default
                            allowlist would reject them.
"""
from __future__ import annotations

import json
from typing import Any

from ..backend.service import VecgrepService


def _require_mcp() -> None:
    """Check that the mcp SDK is installed. Raises RuntimeError with an
    actionable message when the optional extra is missing.
    """
    try:
        import mcp  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "MCP server requires the 'mcp' extra. "
            "Install with `pip install vecgrep[mcp]`."
        ) from e


# ---------------------------------------------------------------------------
# Shared tool logic — called by both the stdio and HTTP handlers.
# ---------------------------------------------------------------------------

def _run_search(args: dict) -> str:
    svc = VecgrepService()
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
            # Raw retriever scores for downstream re-calibration.
            "scores": r.explain or {},
        }
        for r in results
    ]
    return json.dumps(payload, indent=2)


def _run_list_corpora() -> str:
    svc = VecgrepService()
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
    return json.dumps(corpora, indent=2)


def _run_get_corpus(name: str) -> str:
    svc = VecgrepService()
    for c in svc.list_corpora():
        if c.name == name:
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
                "decay_half_life_days": c.decay_half_life_days,
                # What `filters` the caller can pass to search — surfaced so
                # filtering by actor/channel/date isn't a guessing game.
                "filterable": svc.filterable_fields(c.name)["filters"],
            }
            return json.dumps(detail, indent=2)
    return json.dumps({"error": f"corpus not found: {name}"})


# ---------------------------------------------------------------------------
# stdio transport — low-level Server (well-tested, leave as-is)
# ---------------------------------------------------------------------------

def build_mcp_server() -> Any:
    """Construct the configured MCP Server with the three tools registered.

    Used by the stdio transport. The HTTP transport uses build_http_app()
    which builds a FastMCP instance independently.
    """
    _require_mcp()
    from mcp.server import Server
    from mcp.types import TextContent, Tool

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
        try:
            if name == "search":
                text = _run_search(args)
            elif name == "list_corpora":
                text = _run_list_corpora()
            elif name == "get_corpus":
                text = _run_get_corpus(args.get("name", ""))
            else:
                text = json.dumps({"error": f"unknown tool: {name}"})
        except Exception as exc:
            text = json.dumps({"error": str(exc)})
        return [TextContent(type="text", text=text)]

    return server


# ---------------------------------------------------------------------------
# HTTP transport — FastMCP with json_response + stateless_http
# ---------------------------------------------------------------------------

# ── OAuth helpers ───────────────────────────────────────────────────────────
# The MCP endpoint lives at <root>/mcp, but OAuth/RFC put discovery .well-known
# at the ORIGIN ROOT. So the auth-SERVER issuer is the root (well-known +
# /authorize + /token + /register land at root, where the SPA-free funnel root
# serves them), while the protected RESOURCE is <root>/mcp (the thing being
# guarded). One shared provider instance backs both the /mcp bearer gate (via
# the sub-app's AuthSettings) and the root discovery routes (mounted by main.py)
# so tokens issued at root validate on /mcp.

_PROVIDER = None  # singleton — same token store for the gate + the root routes


def _shared_provider():
    global _PROVIDER
    if _PROVIDER is None:
        from ..backend.auth.provider import VecgrepOAuthProvider
        _PROVIDER = VecgrepOAuthProvider()
    return _PROVIDER


def _oauth_issuer(issuer_url: str):
    """Auth-server issuer = the ORIGIN ROOT (strip any /mcp path)."""
    from pydantic import AnyHttpUrl
    from urllib.parse import urlsplit
    p = urlsplit(issuer_url)
    return AnyHttpUrl(f"{p.scheme}://{p.netloc}")


def _oauth_resource(issuer_url: str):
    """Protected resource = the MCP endpoint itself (<root>/mcp)."""
    from pydantic import AnyHttpUrl
    from urllib.parse import urlsplit
    p = urlsplit(issuer_url)
    return AnyHttpUrl(f"{p.scheme}://{p.netloc}/mcp")


def _oauth_client_reg():
    """Dynamic Client Registration (RFC 7591): claude.ai self-registers instead
    of needing a pre-shared client_id."""
    from mcp.server.auth.settings import ClientRegistrationOptions
    return ClientRegistrationOptions(
        enabled=True, valid_scopes=["read", "propose"], default_scopes=["read"],
    )


def build_oauth_root_routes(oauth_issuer_url: str) -> list:
    """Return the OAuth discovery + auth routes to mount at the PARENT app ROOT
    (ahead of the SPA catch-all). These serve the .well-known JSON, /authorize,
    /token, /register at the origin root where the SDK advertises them and where
    claude.ai looks. Shares the provider with the /mcp bearer gate."""
    _require_mcp()
    from mcp.server.auth.routes import (
        create_auth_routes, create_protected_resource_routes,
    )
    issuer = _oauth_issuer(oauth_issuer_url)
    resource = _oauth_resource(oauth_issuer_url)
    routes = create_auth_routes(
        provider=_shared_provider(),
        issuer_url=issuer,
        client_registration_options=_oauth_client_reg(),
    )
    routes += create_protected_resource_routes(
        resource_url=resource,
        authorization_servers=[issuer],
        scopes_supported=["read", "propose"],
    )
    return routes


def build_http_app(oauth_issuer_url: str | None = None) -> Any:
    """Build a Starlette ASGI app for the MCP HTTP endpoint.

    oauth_issuer_url: when set, enable the embedded OAuth 2.1 auth server on this
    endpoint (FastMCP mounts the auth routes + bearer middleware). When None, no
    OAuth — the endpoint is network-trusted by the deployment.

    Uses FastMCP instead of the raw StreamableHTTPSessionManager so that:

    1. json_response=True  — respond with application/json, not text/event-stream.
       Remote clients (claude.ai) send "Accept: application/json" on their initial
       probe; the old StreamableHTTPSessionManager returned SSE and the client got
       a stalled or 406 response.

    2. stateless_http=True — no server-side session state. Each HTTP request is
       fully self-contained. Prevents stale session ID bugs where a client caches
       a dead session ID after a server restart and never re-initialises.

    3. DNS rebinding protection disabled — requests arrive through Tailscale Funnel
       with a non-localhost Host header; the default allowlist would reject them.

    The returned Starlette app registers the MCP handler at '/' (via
    streamable_http_path='/'), so the parent FastAPI app's _BearerGatedASGI
    middleware can strip the '/mcp' prefix and delegate correctly.
    """
    _require_mcp()
    from mcp.server.fastmcp import FastMCP
    from mcp.server.fastmcp.server import TransportSecuritySettings

    # OAuth: when an issuer URL is configured, run the embedded auth server.
    # FastMCP, given `auth` + `auth_server_provider`, auto-mounts /authorize,
    # /token, /.well-known and gates the MCP endpoint with bearer middleware.
    # Off (issuer None) → no auth here; /mcp is network-trusted (the parent app
    # reaches it over localhost/tailnet, or via the OAuth-less secret path).
    fmcp_kwargs: dict = {}
    if oauth_issuer_url:
        from mcp.server.auth.settings import AuthSettings
        fmcp_kwargs["auth"] = AuthSettings(
            issuer_url=_oauth_issuer(oauth_issuer_url),
            resource_server_url=_oauth_resource(oauth_issuer_url),
            required_scopes=["read"],
            client_registration_options=_oauth_client_reg(),
        )
        fmcp_kwargs["auth_server_provider"] = _shared_provider()

    fmcp = FastMCP("vecgrep", **fmcp_kwargs)
    fmcp.settings.json_response = True
    fmcp.settings.stateless_http = True
    # Register at '/' so the parent app's prefix-stripping (_BearerGatedASGI
    # rewrites scope['path'] = '/') routes correctly.
    fmcp.settings.streamable_http_path = "/"
    # Tailscale Funnel forwards requests with a non-localhost Host header.
    fmcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    )

    @fmcp.tool(
        description=(
            "Semantic search across vecgrep corpora. Returns ranked chunks with "
            "surrounding context. Use this instead of dumping documents into "
            "context — index once, search per question."
        )
    )
    def search(
        query: str,
        corpus: str | None = None,
        top_k: int = 5,
        mode: str = "hybrid",
        rerank: bool = False,
        filters: list[str] | None = None,
    ) -> str:
        """Natural-language query. corpus: limit to one corpus (omit = all).
        top_k: max results. mode: hybrid|vector|bm25.
        rerank: cross-encoder rerank (slower, more accurate).
        filters: list of 'source:<glob>', 'corpus:<name>', or 'meta.<k>=<v>'."""
        return _run_search({
            "query": query,
            "corpus": corpus,
            "top_k": top_k,
            "mode": mode,
            "rerank": rerank,
            "filters": filters,
        })

    @fmcp.tool(
        description="List every vecgrep corpus and its stats (doc count, chunk count, embedding model)."
    )
    def list_corpora() -> str:
        """Returns JSON array of corpus metadata."""
        return _run_list_corpora()

    @fmcp.tool(
        description="Get full metadata for one corpus including its source list."
    )
    def get_corpus(name: str) -> str:
        """name: corpus name (from list_corpora)."""
        return _run_get_corpus(name)

    return fmcp.streamable_http_app()


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
