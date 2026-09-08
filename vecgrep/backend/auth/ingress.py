"""Separate public OAuth ingress from private network-trusted access."""
from __future__ import annotations

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

PRIVATE_INGRESS = "vecgrep.private_ingress"


class OAuthIngressMiddleware:
    def __init__(self, app: ASGIApp, *, private_port: int, public_port: int,
                 oauth_ready: bool) -> None:
        self.app = app
        self.private_port = private_port
        self.public_port = public_port
        self.oauth_ready = oauth_ready

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        server = scope.get("server")
        port = server[1] if server else None
        # ASGI server comes from getsockname(), not Host, XFF, or client IP.
        # Unknown ingress never acquires private-network privileges.
        scope = dict(scope)
        scope[PRIVATE_INGRESS] = port == self.private_port
        if port == self.public_port and port != self.private_port:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            if not self.oauth_ready:
                await PlainTextResponse("OAuth ingress unavailable", status_code=503)(scope, receive, send)
                return
            path = scope.get("path", "")
            if not (path in {"/mcp", "/authorize", "/token", "/register", "/oauth/unlock"}
                    or path.startswith(("/mcp/", "/.well-known/"))):
                await PlainTextResponse("Not found", status_code=404)(scope, receive, send)
                return
        await self.app(scope, receive, send)
