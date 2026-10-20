"""Separate public OAuth ingress from private network-trusted access."""
from __future__ import annotations

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

PRIVATE_INGRESS = "vecgrep.private_ingress"

# The only socket peers that count as "this machine". Not ipaddress.is_loopback:
# that admits all of 127.0.0.0/8, so a local forwarder bound to 127.0.0.2 would
# inherit the same-machine trust of the real client.
_SAME_MACHINE_PEERS = frozenset({"127.0.0.1", "::1"})


def has_exact_loopback_peer(scope: Scope) -> bool:
    """True only when the accepted socket's client address is 127.0.0.1 or ::1."""
    client = scope.get("client")
    if not client:
        return False
    try:
        return str(client[0]) in _SAME_MACHINE_PEERS
    except (TypeError, IndexError):
        return False


def has_trusted_proxied_peer(scope: Scope) -> bool:
    """True when uvicorn's proxy-headers middleware rewrote the client.

    uvicorn replaces the client with the first ``X-Forwarded-For`` hop and port
    0 only when the socket peer is in ``forwarded_allow_ips``, ``127.0.0.1``
    alone by default. That shape therefore proves the request arrived through
    the loopback proxy (Tailscale Serve); a forwarder on 127.0.0.2 keeps its
    own address because its forwarded header is not trusted."""
    client = scope.get("client")
    if not client:
        return False
    try:
        host, port = str(client[0]), int(client[1])
    except (TypeError, IndexError, ValueError):
        return False
    if port != 0:
        return False
    first = b""
    for key, value in scope.get("headers") or ():
        if bytes(key).lower() == b"x-forwarded-for":
            first = bytes(value).split(b",")[0].strip()
            break
    return bool(first) and first.decode("ascii", "replace") == host


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
            if not (path in {"/mcp", "/authorize", "/token", "/register", "/oauth/unlock", "/oauth/continue"}
                    or path.startswith(("/mcp/", "/.well-known/"))):
                await PlainTextResponse("Not found", status_code=404)(scope, receive, send)
                return
            # POST-only routes: a GET would otherwise fall through to the UI
            # catch-all and answer with index.html.
            if path in {"/token", "/register"} and scope.get("method") in {"GET", "HEAD"}:
                await PlainTextResponse("Method Not Allowed", status_code=405,
                                        headers={"Allow": "POST"})(scope, receive, send)
                return
        await self.app(scope, receive, send)
