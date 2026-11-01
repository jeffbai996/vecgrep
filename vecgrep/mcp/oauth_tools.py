"""Publish per-tool OAuth scopes and request consent for missing grants."""
from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit


def install(server: Any, issuer_url: str) -> None:
    """Keep existing handlers, but expose recoverable authorization failures.

    Trusted local calls have no OAuth token and keep their existing behavior.
    This never extends a token: only a new owner-approved OAuth flow can do so.
    """
    from mcp.server.auth.middleware.auth_context import get_access_token
    from mcp.types import CallToolRequest, CallToolResult, ServerResult, TextContent

    scopes_by_tool = {
        tool.name: ["read", "propose"]
        if tool.annotations and tool.annotations.readOnlyHint is False
        else ["read"]
        for tool in server._tool_manager.list_tools()
    }
    parsed = urlsplit(issuer_url)
    metadata_url = urlunsplit((
        parsed.scheme, parsed.netloc,
        "/.well-known/oauth-protected-resource" + parsed.path.rstrip("/"),
        "", "",
    ))
    original_list = server.list_tools
    original_call = server._mcp_server.request_handlers[CallToolRequest]

    @server._mcp_server.list_tools()
    async def list_tools():
        tools = await original_list()
        result = []
        for tool in tools:
            schemes = [{"type": "oauth2", "scopes": scopes_by_tool[tool.name]}]
            # Top-level declaration plus the Apps SDK compatibility mirror.
            tool = tool.model_copy(update={
                "securitySchemes": schemes,
                "meta": {**(tool.meta or {}), "securitySchemes": schemes},
            })
            result.append(tool)
        return result

    async def call_tool(request):
        token = get_access_token()
        required = scopes_by_tool.get(request.params.name, [])
        if token is not None and not set(required).issubset(token.scopes):
            description = "Reconnect and approve the scopes required by this tool."
            challenge = (
                f'Bearer resource_metadata="{metadata_url}", '
                'error="insufficient_scope", '
                f'error_description="{description}", '
                f'scope="{" ".join(required)}"'
            )
            return ServerResult(CallToolResult(
                content=[TextContent(
                    type="text",
                    text="OAuth permission required: " + " ".join(required),
                )],
                isError=True,
                _meta={"mcp/www_authenticate": [challenge]},
            ))
        return await original_call(request)

    server._mcp_server.request_handlers[CallToolRequest] = call_tool
