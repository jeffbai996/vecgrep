"""Human approval gate for vecgrep's embedded OAuth authorization endpoint.

Dynamic client registration identifies an OAuth client; it does not identify
the vecgrep owner.  A public authorization endpoint therefore needs a separate
owner-presence check before the provider may mint an authorization code.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
from urllib.parse import parse_qs, urlencode, urlsplit

from starlette.datastructures import Headers
from starlette.responses import RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send


APPROVAL_COOKIE = "__Host-vecgrep_oauth_approved"
TAILNET_INTENT_COOKIE = "__Host-vecgrep_oauth_intent"
AUTHORIZE_PATHS = frozenset({"/authorize", "/mcp/authorize"})
_TAILSCALE_HEADERS_INFO = "https://tailscale.com/s/serve-headers"


def approval_cookie_value(secret: str) -> str:
    """Return a non-reversible verifier for the browser approval cookie."""
    return hmac.new(
        secret.strip().encode("utf-8"),
        b"vecgrep-oauth-owner-approval-v1",
        hashlib.sha256,
    ).hexdigest()


def safe_authorize_target(raw: str | None) -> str:
    """Keep post-unlock redirects on one of the two local authorize routes."""
    value = raw or "/authorize"
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or parsed.path not in AUTHORIZE_PATHS:
        return "/authorize"
    return parsed.path + (f"?{parsed.query}" if parsed.query else "")


def oauth_callback_origin(authorize_target: str) -> str | None:
    """Return the registered callback origin for the unlock page's CSP.

    Safari applies ``form-action`` across redirects. The unlock form posts to
    vecgrep and then `/authorize` redirects to the OAuth client's callback, so
    a self-only policy silently blocks an otherwise valid flow. Only a clean
    HTTPS origin (or HTTP loopback for local development) is admitted here;
    the provider still performs the authoritative redirect-URI registration
    check before issuing an authorization code.
    """
    target = urlsplit(safe_authorize_target(authorize_target))
    redirect_uri = (parse_qs(target.query).get("redirect_uri") or [""])[0]
    try:
        parsed = urlsplit(redirect_uri)
        host = parsed.hostname or ""
        port = parsed.port
    except ValueError:
        return None
    if not host or parsed.username or parsed.password:
        return None
    if parsed.scheme == "https":
        pass
    elif parsed.scheme == "http":
        try:
            if host.lower() != "localhost" and not ipaddress.ip_address(host).is_loopback:
                return None
        except ValueError:
            return None
    else:
        return None
    rendered_host = f"[{host}]" if ":" in host else host
    return f"{parsed.scheme}://{rendered_host}" + (f":{port}" if port else "")


def verified_tailnet_login(scope: Scope) -> str | None:
    """Return tailscaled's verified Serve identity, never Funnel input.

    The service's loopback-only listener is what makes proxy-supplied identity
    trustworthy. Tailscale strips incoming identity headers before supplying
    its own on Serve requests; Funnel does not receive an identity header.
    """
    if scope.get("type") != "http":
        return None
    server = scope.get("server")
    if not server:
        return None
    host = str(server[0]).split("%", 1)[0]
    try:
        loopback = (
            host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
        )
    except (ValueError, TypeError, IndexError):
        loopback = False
    if not loopback:
        return None
    headers = Headers(scope=scope)
    login = headers.get("tailscale-user-login", "").strip()
    if (
        not login
        or not headers.get("x-forwarded-for", "").strip()
        or headers.get("tailscale-headers-info", "").strip() != _TAILSCALE_HEADERS_INFO
        or "tailscale-funnel-request" in headers
    ):
        return None
    return login


def tailnet_approval_intent(secret: str, login: str, target: str) -> str:
    """Bind a one-click approval to the verified identity and OAuth request."""
    payload = (
        b"vecgrep-oauth-tailnet-intent-v1\0"
        + login.encode()
        + b"\0"
        + target.encode()
    )
    return hmac.new(secret.strip().encode(), payload, hashlib.sha256).hexdigest()


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
