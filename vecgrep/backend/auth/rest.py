"""Host validation for the tokenless local REST API."""
from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlsplit

from starlette.responses import PlainTextResponse

from ..config import get_settings


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
    """Reject rebound Host values before tokenless ``/api`` dispatch."""

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
                if len(hosts) != 1 or not _is_allowed_loopback_host(
                    hosts[0],
                    api_port=settings.api_port,
                    scheme=scope.get("scheme", "http"),
                ):
                    response = PlainTextResponse(
                        "Invalid host header", status_code=421
                    )
                    await response(scope, receive, send)
                    return

        await self.app(scope, receive, send)
