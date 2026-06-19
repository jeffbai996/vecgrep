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


def _write_dir(corpus: str):
    from ..backend.config import get_settings
    return get_settings().home / "write" / corpus


def _pending_store():
    from ..backend.config import get_settings
    from ..backend.write import confirm as _C
    return _C.ProposalStore(get_settings().home / "write" / "_pending")


def _run_propose(corpus: str, content: str, edit_id: str | None = None,
                 source_kind: str | None = None, tags: list[str] | None = None,
                 origin: str = "bot-suggested") -> str:
    """PROPOSE an entry (or edit) — WRITES NOTHING. Stores a pending proposal a
    human confirms later (`vecgrep confirm <id>`).

    This is the MCP-facing path: an agent (claude.ai) ingests untrusted content,
    so a direct write would be a prompt-injection → memory-poisoning vector. A
    proposal is inert until a human reviews + confirms it off-protocol. origin
    is 'bot-suggested' here (the wall: bots propose, humans authorize). Returns
    JSON {proposal_id, doc_id, is_edit, preview} or {error}."""
    from ..backend.write import proposal as _P

    corpus_dir = _write_dir(corpus)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    meta = {"origin": origin}
    if source_kind:
        meta["source_kind"] = source_kind
    if tags:
        meta["tags"] = list(tags)
    try:
        pr = _P.propose(corpus, content, corpus_dir, meta=meta, edit_id=edit_id)
    except _P.ProposalError as e:
        return json.dumps({"error": str(e)})
    _pending_store().put(pr)
    result = {
        "proposal_id": pr.proposal_id, "doc_id": pr.doc_id,
        "is_edit": pr.is_edit, "corpus": corpus,
        "status": "pending — a human must run `vecgrep confirm "
                  f"{pr.proposal_id}` to write it",
        "preview": pr.rendered[:500],
    }
    _fire_propose_hook(pr)
    return json.dumps(result)


def _fire_propose_hook(pr) -> None:
    """Optionally notify an external command that a proposal was created.

    If VECGREP_PROPOSE_HOOK is set, run it with the proposal JSON on stdin so a
    deployment can surface pending proposals out-of-band (a Discord card, a
    desktop notification, etc.) without vecgrep itself knowing anything about
    those channels — it stays generic. Strictly best-effort: the proposal is
    already safely stored, so a missing/failing hook must never break propose.
    """
    import os
    hook = os.environ.get("VECGREP_PROPOSE_HOOK", "").strip()
    if not hook:
        return
    import subprocess
    payload = json.dumps({
        "proposal_id": pr.proposal_id, "doc_id": pr.doc_id,
        "corpus": pr.corpus, "is_edit": pr.is_edit,
        "target_path": pr.target_path, "preview": pr.rendered[:1000],
        "meta": pr.meta,
    })
    try:
        subprocess.run(
            hook, shell=True, input=payload, text=True,
            timeout=10, capture_output=True,
        )
    except Exception:
        # Notification is a nicety; the write path's correctness does not
        # depend on it. Swallow everything.
        pass


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

def build_http_app() -> Any:
    """Build a Starlette ASGI app for the MCP HTTP endpoint.

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

    fmcp = FastMCP("vecgrep")
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

    @fmcp.tool(
        description=(
            "PROPOSE a new entry for a corpus. Writes NOTHING — creates a pending "
            "proposal a human reviews + confirms before it's saved. Use for "
            "durable notes/facts/decisions worth recalling later. Returns the "
            "proposal_id; tell the user to confirm it."
        )
    )
    def propose_write(corpus: str, content: str,
                      source_kind: str | None = None,
                      tags: list[str] | None = None) -> str:
        """corpus: target. content: entry text. source_kind: insight|fact|
        correction|journal|decision. tags: optional. Returns the pending
        proposal (nothing is written until a human confirms)."""
        return _run_propose(corpus, content, None, source_kind, tags)

    @fmcp.tool(
        description=(
            "PROPOSE an edit to an existing entry (by id, e.g. notes-007). Writes "
            "NOTHING — creates a pending proposal a human confirms before it "
            "overwrites. Returns the proposal_id."
        )
    )
    def propose_edit(doc_id: str, content: str, corpus: str | None = None,
                     source_kind: str | None = None,
                     tags: list[str] | None = None) -> str:
        """doc_id: existing id. content: new text. corpus: inferred from id
        prefix if omitted. Proposal only — a human must confirm."""
        if corpus is None:
            corpus = doc_id.rsplit("-", 1)[0] if "-" in doc_id else doc_id
        return _run_propose(corpus, content, doc_id, source_kind, tags)

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
