"""FastAPI app. Mounts /api routes; serves the React frontend from
vecgrep/frontend/dist/ when present; mounts /mcp for MCP-over-HTTP when
the optional `mcp` extra is installed.
"""
from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.routing import Route

from .. import __version__
from .api.routes import public_router, require_token, router

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

logger = logging.getLogger(__name__)


def _try_build_mcp_http_app() -> Any | None:
    """Best-effort build of the MCP HTTP sub-app. Returns None when the
    `mcp` extra isn't installed — callers should log and skip mounting.
    """
    try:
        from ..mcp.server import build_http_app
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("MCP HTTP endpoint disabled: %s", e)
        return None
    from .config import get_settings
    s = get_settings()
    issuer = s.oauth_issuer_url if s.oauth_enabled else None
    if s.oauth_enabled and not issuer:
        logger.warning("VECGREP_OAUTH_ENABLED set but no issuer URL; OAuth off.")
    try:
        return build_http_app(oauth_issuer_url=issuer)
    except RuntimeError as e:
        # build_http_app raises RuntimeError when the mcp extra is missing
        # (the helpful "pip install vecgrep[mcp]" message).
        logger.warning("MCP HTTP endpoint disabled: %s", e)
        return None


class _McpPathRewriteASGI:
    """Rewrite the `/mcp` request path to `/` before delegating to the MCP
    sub-app (whose router expects `/`, where build_http_app registered the
    streamable HTTP handler).

    Auth is NOT done here anymore. When OAuth is enabled, FastMCP's own bearer
    middleware (inside the sub-app) gates the endpoint. When OAuth is off, /mcp
    is network-trusted — the deployment reaches it over localhost/tailnet or via
    an unguessable funnel path. (The old static-token check lived here; it was
    retired in favor of OAuth-on-/mcp + network-trust-on-/api.)

    Class-not-function because Starlette only treats class-instance callables as
    raw ASGI; a function is treated as a `func(request)` handler.
    """

    def __init__(self, asgi_app: Callable[..., Any]) -> None:
        self._app = asgi_app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        # Strip the /mcp prefix so the sub-app's `/` route matches. Copy scope to
        # avoid mutating shared state across the request lifecycle.
        sub_scope = dict(scope)
        sub_scope["path"] = "/"
        sub_scope["raw_path"] = b"/"
        await self._app(sub_scope, receive, send)


def create_app() -> FastAPI:
    mcp_http_app = _try_build_mcp_http_app()

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # The MCP streamable HTTP session manager runs as an async task
        # group — its lifespan must be entered for the /mcp endpoint to
        # serve. When mcp_http_app is None (extra not installed), we
        # still need a no-op lifespan so the app boots cleanly.
        if mcp_http_app is not None:
            async with mcp_http_app.router.lifespan_context(mcp_http_app):
                yield
        else:
            yield

    app = FastAPI(title="vecgrep", version=__version__, lifespan=lifespan)

    # CORS is permissive in dev to let the Vite dev server hit the API while
    # we're iterating on the frontend. Lock down in production deployments.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(public_router)
    app.include_router(router)

    if mcp_http_app is not None:
        # Register before the SPA fallback below so it wins routing.
        # Mount at /mcp would 405 on `POST /mcp` (no trailing slash) by
        # falling through to the SPA's GET-only catch-all. Explicit Routes
        # for both the slash and slashless forms keep MCP clients happy
        # regardless of how they normalise their URLs.
        gated = _McpPathRewriteASGI(mcp_http_app)
        app.router.routes.append(
            Route("/mcp", endpoint=gated, methods=["GET", "POST", "DELETE"])
        )
        app.router.routes.append(
            Route("/mcp/", endpoint=gated, methods=["GET", "POST", "DELETE"])
        )
    else:
        logger.warning("vecgrep[mcp] extra not installed; /mcp endpoint not mounted")

    if FRONTEND_DIR.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=FRONTEND_DIR / "assets"),
            name="assets",
        )

        @app.get("/")
        def root() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "index.html")

        @app.get("/{full_path:path}")
        def spa(full_path: str) -> FileResponse:
            # SPA fallback — any unknown path serves index.html so React
            # router can take over. /api routes are matched earlier and
            # never hit this handler.
            return FileResponse(FRONTEND_DIR / "index.html")
    else:
        @app.get("/")
        def no_frontend() -> dict:
            return {
                "message": "vecgrep API is running. Frontend not built. "
                "See README for build instructions or just use the CLI.",
                "version": __version__,
            }

    return app


app = create_app()
