"""Human approval gate for vecgrep's embedded OAuth authorization endpoint.

Dynamic client registration identifies an OAuth client; it does not identify
the vecgrep owner.  A public authorization endpoint therefore needs a separate
owner-presence check before the provider may mint an authorization code.
"""
from __future__ import annotations

import hashlib
import hmac
from urllib.parse import urlencode

from starlette.datastructures import Headers
from starlette.responses import RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send


APPROVAL_COOKIE = "__Host-vecgrep_oauth_approved"
AUTHORIZE_PATHS = frozenset({"/authorize", "/mcp/authorize"})


def approval_cookie_value(secret: str) -> str:
    """Return a non-reversible verifier for the browser approval cookie."""
    return hmac.new(
        secret.strip().encode("utf-8"),
        b"vecgrep-oauth-owner-approval-v1",
        hashlib.sha256,
    ).hexdigest()


def safe_authorize_target(raw: str | None) -> str:
    """Keep post-unlock redirects on one of the two local authorize routes."""
    from urllib.parse import urlsplit

    value = raw or "/authorize"
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or parsed.path not in AUTHORIZE_PATHS:
        return "/authorize"
    return parsed.path + (f"?{parsed.query}" if parsed.query else "")


class OAuthApprovalMiddleware:
    """Require an owner-approved browser before dispatching `/authorize`.

    The cookie contains only an HMAC verifier, never the approval token.  It is
    SameSite=Strict, so a fresh cross-site OAuth navigation intentionally lands
    on the unlock form rather than silently granting a client access.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or scope.get("path") not in AUTHORIZE_PATHS:
            await self.app(scope, receive, send)
            return

        from ..config import get_settings

        settings = get_settings()
        if not settings.oauth_enabled:
            await self.app(scope, receive, send)
            return

        expected = approval_cookie_value(settings.oauth_approval_token or "")
        headers = Headers(scope=scope)
        cookies = {}
        for item in headers.get("cookie", "").split(";"):
            name, sep, value = item.strip().partition("=")
            if sep:
                cookies[name] = value
        provided = cookies.get(APPROVAL_COOKIE, "")
        if not provided or not hmac.compare_digest(provided, expected):
            query = scope.get("query_string", b"").decode("latin-1")
            target = str(scope["path"]) + (f"?{query}" if query else "")
            response = RedirectResponse(
                "/oauth/unlock?" + urlencode({"next": target}),
                status_code=303,
                headers={"Cache-Control": "no-store"},
            )
            await response(scope, receive, send)
            return

        async def no_store(message: dict) -> None:
            if message.get("type") == "http.response.start":
                mutable = list(message.get("headers", []))
                mutable.append((b"cache-control", b"no-store"))
                message = {**message, "headers": mutable}
            await send(message)

        await self.app(scope, receive, no_store)
