"""FastAPI app. Mounts /api routes; serves the React frontend from
vecgrep/frontend/dist/ when present; mounts /mcp for MCP-over-HTTP when
the optional `mcp` extra is installed.
"""
from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.routing import Mount, Route

from .. import __version__
from .api.routes import public_router, router
from .api.admin import router as admin_router

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


class _McpBareDelegate:
    """Raw-ASGI endpoint for the slashless `/mcp`: rewrite the path to `/` and
    hand the request to the MCP sub-app's root handler. Used ONLY for bare
    `/mcp` (Mount handles `/mcp/...`). Avoids a funnel-breaking redirect.

    Class-not-function: Starlette treats a function endpoint as `func(request)`,
    only a class instance as a raw ASGI `(scope, receive, send)` callable.
    """

    def __init__(self, asgi_app: Callable[..., Any]) -> None:
        self._app = asgi_app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            scope = dict(scope)
            scope["path"] = "/"
            scope["raw_path"] = b"/"
        await self._app(scope, receive, send)


def create_app() -> FastAPI:
    mcp_http_app = _try_build_mcp_http_app()

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # The MCP streamable HTTP session manager runs as an async task
        # group — its lifespan must be entered for the /mcp endpoint to
        # serve. When mcp_http_app is None (extra not installed), we
        # still need a no-op lifespan so the app boots cleanly.
        # AnyIO's default thread-pool limiter (backs FastAPI's run_in_threadpool
        # bridge for any sync route/dependency) defaults to a size meant for a
        # dedicated host. A live instance was observed idling at 63 threads
        # under a few requests/minute — cap it to what this service actually
        # needs. Must be set here, inside the running loop: the limiter is a
        # contextvar-scoped object that doesn't exist until a loop is active,
        # so setting it from serve() before uvicorn starts has no effect.
        import anyio

        from .config import get_settings as _get_settings

        anyio.to_thread.current_default_thread_limiter().total_tokens = (
            _get_settings().thread_pool_size
        )

        # Start loading the cross-encoder now, in the background, so the
        # first search that wants it does not pay the load inside a
        # threadpool slot. Best-effort: a box without the rerank extra logs
        # and carries on serving unreranked.
        try:
            from .rerank import ensure_warm as _warm_reranker

            _warm_reranker()
        except Exception as exc:
            logger.warning("reranker warm-up could not start: %s", exc)

        stop = threading.Event()

        def _backup_loop() -> None:
            from .backup import BackupManager, BackupScheduler
            from .config import get_settings

            while not stop.wait(30):
                settings = get_settings()
                scheduler = BackupScheduler(
                    BackupManager(settings),
                    enabled=settings.backup_enabled,
                    frequency=settings.backup_frequency,
                    local_time=settings.backup_time,
                    weekday=settings.backup_weekday,
                    retention=settings.backup_retention,
                )
                try:
                    scheduler.run_if_due()
                except Exception as exc:
                    logger.error("scheduled backup failed: %s", exc)

        backup_thread = threading.Thread(target=_backup_loop, name="vecgrep-backups", daemon=True)
        backup_thread.start()
        try:
            if mcp_http_app is not None:
                async with mcp_http_app.router.lifespan_context(mcp_http_app):
                    yield
            else:
                yield
        finally:
            stop.set()
            backup_thread.join(timeout=2)

    app = FastAPI(title="vecgrep", version=__version__, lifespan=lifespan)

    app.include_router(public_router)
    app.include_router(router)
    app.include_router(admin_router)

    if mcp_http_app is not None:
        # Mount the MCP sub-app under /mcp so ALL its sub-paths resolve — not
        # just the streamable handler at '/', but the OAuth routes the SDK adds
        # when auth is on (/authorize, /token, /.well-known/...). A path-rewriter
        # that forced everything to '/' (the old single-handler hack) would make
        # those auth routes unreachable. Mount preserves sub-paths: /mcp/ -> the
        # sub-app's '/' (MCP handler), /mcp/authorize -> its '/authorize', etc.
        app.router.routes.append(Mount("/mcp", app=mcp_http_app))
        # Mount matches /mcp/<something> but NOT bare /mcp. A redirect to /mcp/
        # breaks behind the Tailscale Funnel (absolute-from-root, drops the
        # funnel's path prefix → claude.ai loops on 307s). Instead delegate the
        # bare-/mcp request straight into the sub-app's root ('/') by ASGI,
        # rewriting the path in-process — no redirect, funnel-safe. Class
        # instance because Starlette only treats those as raw ASGI endpoints.
        app.router.routes.append(
            Route("/mcp", endpoint=_McpBareDelegate(mcp_http_app),
                  methods=["GET", "POST", "DELETE"])
        )
    else:
        logger.warning("vecgrep[mcp] extra not installed; /mcp endpoint not mounted")

    # OAuth discovery + auth routes at the ROOT, BEFORE the SPA catch-all below.
    # The SDK advertises .well-known/oauth-* at the origin root; without these
    # here the SPA fallback would answer them with HTML and break discovery.
    from .config import get_settings as _gs
    _s = _gs()
    if mcp_http_app is not None and _s.oauth_enabled and _s.oauth_issuer_url:
        try:
            from ..mcp.server import build_oauth_root_routes
            for _r in build_oauth_root_routes(_s.oauth_issuer_url):
                app.router.routes.append(_r)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("OAuth root routes not mounted: %s", e)

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

        @app.get("/{full_path:path}")
        def no_frontend_fallback(full_path: str) -> dict:
            # Mirror the SPA catch-all's shape: any unknown path answers 200
            # with the same message, so client behavior is uniform whether or
            # not the frontend is built. /api and the OAuth root routes are
            # matched earlier and never reach this.
            return no_frontend()

    return app


app = create_app()
