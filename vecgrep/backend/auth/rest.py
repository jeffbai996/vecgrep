"""Host and write-boundary checks for the tokenless REST API."""
from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlsplit

from starlette.responses import PlainTextResponse

from ..config import get_settings


# These POST routes only query the index. All other unsafe REST methods change
# state and must stay local when the API has no bearer token.
_READ_ONLY_POST_PATHS = frozenset({
    "/api/search", "/api/timeline", "/api/incident", "/api/browse", "/api/compare",
})
_PROXY_TRANSIT_HEADERS = frozenset({
    b"forwarded", b"x-forwarded-for", b"x-forwarded-host", b"tailscale-user-login",
})


def _tokenless_write(scope: dict) -> bool:
    method = str(scope.get("method", "GET")).upper()
    if method in {"GET", "HEAD", "OPTIONS"}:
        return False
    return not (method == "POST" and scope.get("path") in _READ_ONLY_POST_PATHS)


def _direct_local_request(scope: dict, host: str, *, api_port: int) -> bool:
    if not _is_allowed_loopback_host(
        host, api_port=api_port, scheme=scope.get("scheme", "http")
    ):
        return False
    peer = (scope.get("client") or (None,))[0]
    try:
        if not ipaddress.ip_address(peer).is_loopback:
            return False
    except (ValueError, TypeError):
        return False
    return not any(
        name.lower() in _PROXY_TRANSIT_HEADERS for name, _ in scope.get("headers", [])
    )


def _is_allowed_loopback_host(
    value: str, *, api_port: int, scheme: str
) -> bool:
    if not value or value != value.strip() or any(char.isspace() for char in value):
        return False

    parsed = urlsplit(f"//{value}")
    if (
        not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        return False
    try:
        port = parsed.port
    except ValueError:
        return False
    request_port = port if port is not None else (443 if scheme == "https" else 80)
    if request_port != api_port:
        return False

    host = parsed.hostname.lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class TokenlessRestHostMiddleware:
    """Reject rebound hosts and proxied writes before tokenless ``/api`` dispatch."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        path = scope.get("path", "")
        is_rest = path == "/api" or path.startswith("/api/")
        if scope.get("type") == "http" and is_rest:
            settings = get_settings()
            if not (settings.api_token or "").strip():
                hosts = [
                    value.decode("latin-1")
                    for name, value in scope.get("headers", [])
                    if name.lower() == b"host"
                ]
                allowed = len(hosts) == 1 and (
                    _is_allowed_loopback_host(
                        hosts[0],
                        api_port=settings.api_port,
                        scheme=scope.get("scheme", "http"),
                    )
                    or hosts[0].lower()
                    in {host.lower() for host in settings.rest_allowed_hosts}
                )
                if not allowed:
                    response = PlainTextResponse(
                        "Invalid host header", status_code=421
                    )
                    await response(scope, receive, send)
                    return

                if _tokenless_write(scope) and not _direct_local_request(
                    scope, hosts[0], api_port=settings.api_port
                ):
                    response = PlainTextResponse(
                        "Tokenless writes require direct loopback", status_code=403
                    )
                    await response(scope, receive, send)
                    return

        await self.app(scope, receive, send)
