"""FastAPI app. Mounts /api routes; serves the React frontend from
vecgrep/frontend/dist/ when present; mounts /mcp for MCP-over-HTTP when
the optional `mcp` extra is installed.
"""
from __future__ import annotations

import contextlib
import html
import hmac
import logging
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.routing import Mount, Route

from .. import __version__
from .api.routes import public_router, router
from .api.admin import router as admin_router
from .auth.approval import (
    APPROVAL_COOKIE,
    OAuthApprovalMiddleware,
    approval_cookie_value,
    safe_authorize_target,
)

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
        return build_http_app(
            oauth_issuer_url=issuer,
            oauth_loopback_bypass=s.oauth_loopback_bypass,
            oauth_tailscale_identity_bypass=s.oauth_tailscale_identity_bypass,
        )
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
    # Covers both the root SDK authorize route and the duplicate mounted route
    # under /mcp. The MCP sub-app also carries this middleware when constructed
    # directly, so neither integration path can accidentally omit owner approval.
    app.add_middleware(OAuthApprovalMiddleware)

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

        async def _unlock_context(next_target: str) -> dict:
            """Who is asking, and for what — read off the authorize target so
            the owner approves a named client with named permissions, not a
            blank form (Jeff 2026-08-22: "should be as lit as logging into
            Claude Code")."""
            from urllib.parse import parse_qs, urlsplit

            q = parse_qs(urlsplit(next_target).query)
            client_id = (q.get("client_id") or [""])[0]
            scopes = [x for x in (q.get("scope") or [""])[0].split() if x] or ["read"]
            name = ""
            if client_id:
                try:
                    from ..mcp.server import _shared_provider
                    info = await _shared_provider().get_client(client_id)
                    name = (getattr(info, "client_name", None) or "") if info else ""
                except Exception:  # pragma: no cover - a missing name is a generic heading
                    name = ""
            return {"client": name, "scopes": scopes}

        _SCOPE_COPY = {
            "read": ("Search and read", "your indexed corpora"),
            "propose": ("Propose changes", "every write still waits for your confirmation"),
        }

        def _unlock_form(next_target: str, *, error: bool = False, ctx: dict | None = None) -> HTMLResponse:
            safe_next = html.escape(safe_authorize_target(next_target), quote=True)
            ctx = ctx or {"client": "", "scopes": ["read"]}
            client = html.escape(ctx.get("client") or "")
            heading = f"Connect {client} to vecgrep" if client else "Connect to vecgrep"
            from urllib.parse import urlsplit as _us
            host = html.escape(_us(_gs().oauth_issuer_url or "").hostname or "")
            rows = "".join(
                f'<li><span class="dot"></span><span><b>{html.escape(_SCOPE_COPY.get(sc, (sc, ""))[0])}</b>'
                f'<small>{html.escape(_SCOPE_COPY.get(sc, (sc, ""))[1])}</small></span></li>'
                for sc in ctx.get("scopes", ["read"])
            )
            error_html = (
                """<div class="err" role="alert">
<svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
<circle cx="7" cy="7" r="6" stroke="currentColor" stroke-width="1.3"/>
<path d="M7 4v3.4" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
<circle cx="7" cy="9.6" r="0.8" fill="currentColor"/></svg>
<span>That code wasn't accepted.</span></div>"""
                if error else ""
            )
            # No <script> anywhere on purpose: the CSP below is default-src 'none'
            # with only style-src 'unsafe-inline' allowed, so this stays a plain
            # POST form with zero script surface on the one page that gates real
            # write-capable MCP access. All interaction (hover/focus/press) is
            # CSS-only. No web fonts either — default-src 'none' would block the
            # fetch anyway, and the system stack renders instantly with no FOUC.
            # Palette is the ticker-tape board: near-black, amber accent, quiet
            # greys — this page is part of the same product, not a vendor form.
            body = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>{html.escape(heading)}</title>
<style>
:root {{
  --bg: #07080a; --bg-2: #0e1013; --bg-3: #15181d;
  --fg: #e6e8eb; --fg-2: #9aa2ad; --muted: #6b7480; --line: #1f242b;
  --accent: #f59e0b; --accent-hi: #fbbf24; --accent-ring: rgba(245,158,11,0.28); --accent-wash: rgba(245,158,11,0.10);
  --bad: #f85149; --bad-soft: rgba(248,81,73,0.10); --bad-border: rgba(248,81,73,0.35);
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI Variable", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}}
* {{ box-sizing: border-box; }}
html, body {{ height: 100%; }}
body {{
  margin: 0; min-height: 100vh; display: grid; place-items: center; padding: 24px;
  background: var(--bg); color: var(--fg); font-family: var(--font); -webkit-font-smoothing: antialiased;
}}
body::before {{
  content: ""; position: fixed; inset: 0; pointer-events: none;
  background: radial-gradient(60% 40% at 50% 0%, var(--accent-wash), transparent 70%);
}}
main {{ position: relative; width: 100%; max-width: 400px; }}
.brand {{ display: flex; align-items: center; gap: 10px; margin: 0 4px 14px; color: var(--fg-2); font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase; }}
.brand .mark {{ width: 22px; height: 22px; display: grid; place-items: center; border-radius: 6px; background: var(--bg-3); border: 1px solid var(--line); color: var(--accent); }}
.brand b {{ color: var(--fg); font-weight: 600; letter-spacing: 0; text-transform: none; font-size: 0.85rem; }}
.card {{ background: var(--bg-2); border: 1px solid var(--line); border-radius: 14px; padding: 28px 26px 24px; box-shadow: 0 30px 80px -40px rgba(0,0,0,0.8); }}
h1 {{ font-size: 1.15rem; font-weight: 650; margin: 0 0 6px; letter-spacing: -0.01em; }}
p.sub {{ margin: 0 0 18px; color: var(--fg-2); font-size: 0.86rem; line-height: 1.5; }}
ul.scopes {{ list-style: none; margin: 0 0 22px; padding: 12px 14px; border: 1px solid var(--line); border-radius: 10px; background: var(--bg-3); display: grid; gap: 10px; }}
ul.scopes li {{ display: flex; gap: 10px; align-items: flex-start; font-size: 0.84rem; line-height: 1.35; }}
ul.scopes .dot {{ flex: none; width: 7px; height: 7px; margin-top: 6px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 3px var(--accent-wash); }}
ul.scopes b {{ display: block; font-weight: 600; color: var(--fg); }}
ul.scopes small {{ display: block; color: var(--muted); font-size: 0.76rem; }}
label {{ display: block; font-size: 0.7rem; font-weight: 600; color: var(--fg-2); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }}
input[type="password"] {{
  width: 100%; background: var(--bg-3); border: 1px solid var(--line); border-radius: 10px;
  padding: 13px 14px; color: var(--fg); font: inherit; font-family: var(--mono); font-size: 0.95rem; letter-spacing: 0.08em;
  transition: border-color 120ms ease, box-shadow 120ms ease;
}}
input[type="password"]::placeholder {{ color: var(--muted); letter-spacing: 0.2em; }}
input[type="password"]:focus-visible {{ outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-ring); }}
button {{
  width: 100%; margin-top: 14px; padding: 13px 16px; border-radius: 10px; border: none;
  background: var(--accent); color: #0b0804; font: inherit; font-weight: 650; font-size: 0.92rem; letter-spacing: 0.01em;
  cursor: pointer; transition: background-color 120ms ease, transform 60ms ease;
}}
button:hover {{ background: var(--accent-hi); }}
button:active {{ transform: scale(0.985); }}
button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 3px; }}
.err {{ display: flex; gap: 8px; align-items: center; margin: 0 0 16px; background: var(--bad-soft); border: 1px solid var(--bad-border); color: var(--bad); border-radius: 10px; padding: 10px 12px; font-size: 0.8rem; }}
.err svg {{ flex: none; }}
footer {{ margin: 16px 4px 0; display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--muted); letter-spacing: 0.04em; }}
</style>
<body>
<main>
<div class="brand">
<span class="mark"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="13" cy="13" r="5" stroke="currentColor" stroke-width="2.4"/><path d="M17 17l4 4" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/><circle cx="5" cy="6" r="1.6" fill="currentColor" opacity=".6"/><circle cx="11" cy="4" r="1.6" fill="currentColor" opacity=".6"/><circle cx="6" cy="15" r="1.6" fill="currentColor" opacity=".6"/></svg></span>
<b>vecgrep</b><span>{host}</span>
</div>
<div class="card">
<h1>{html.escape(heading)}</h1>
<p class="sub">{("<b>" + client + "</b> is asking for access. ") if client else ""}Enter the owner approval code to allow it.</p>
<ul class="scopes">{rows}</ul>
{error_html}<form method="post" action="/oauth/unlock">
<input type="hidden" name="next" value="{safe_next}">
<label for="token">Owner approval code</label>
<input id="token" name="token" type="password" autocomplete="current-password" placeholder="••••••••" required autofocus>
<button type="submit">Approve &amp; continue</button>
</form>
</div>
<footer><span>OAuth 2.1 · PKCE</span><span>approval lasts 8 hours</span></footer>
</main>
</body></html>"""
            return HTMLResponse(
                body,
                status_code=401 if error else 200,
                headers={
                    "Cache-Control": "no-store",
                    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
                    "Referrer-Policy": "no-referrer",
                    "X-Content-Type-Options": "nosniff",
                },
            )

        @app.get("/oauth/unlock", response_class=HTMLResponse)
        async def oauth_unlock(next: str = "/authorize") -> HTMLResponse:
            target = safe_authorize_target(next)
            return _unlock_form(target, ctx=await _unlock_context(target))

        @app.post("/oauth/unlock")
        async def oauth_unlock_submit(request: Request):
            content_type = request.headers.get("content-type", "").split(";", 1)[0]
            body = await request.body()
            if content_type != "application/x-www-form-urlencoded" or len(body) > 8192:
                return _unlock_form("/authorize", error=True)
            from urllib.parse import parse_qs

            fields = parse_qs(body.decode("utf-8", "replace"), keep_blank_values=True)
            # .strip(): the owner pastes this from a terminal (grep output,
            # a copied env line), and that copy near-universally carries a
            # trailing newline or trailing space. `expected` was already
            # stripped; `provided` wasn't, so a clean correct paste failed
            # hmac.compare_digest on whitespace alone (Jeff 2026-08-22).
            provided = (fields.get("token") or [""])[0].strip()
            next_target = safe_authorize_target((fields.get("next") or [""])[0])
            expected = _gs().oauth_approval_token or ""
            if not provided or not hmac.compare_digest(provided, expected.strip()):
                return _unlock_form(next_target, error=True, ctx=await _unlock_context(next_target))
            response = RedirectResponse(next_target, status_code=303)
            response.set_cookie(
                APPROVAL_COOKIE,
                approval_cookie_value(expected),
                max_age=8 * 60 * 60,
                secure=True,
                httponly=True,
                samesite="strict",
                path="/",
            )
            response.headers["Cache-Control"] = "no-store"
            return response

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
