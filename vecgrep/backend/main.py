"""FastAPI app. Mounts /api routes; serves the React frontend from
vecgrep/frontend/dist/ when present.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from .api.routes import router

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"


def create_app() -> FastAPI:
    app = FastAPI(title="vecgrep", version=__version__)

    # CORS is permissive in dev to let the Vite dev server hit the API while
    # we're iterating on the frontend. Lock down in production deployments.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

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
