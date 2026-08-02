"""AnyIO's default worker-thread limiter is sized for a dedicated host, not
a lightly-loaded background service — a live instance was observed idling
at 63 threads under a few requests/minute (Jeff, 2026-08-02). The lifespan
now caps it to Settings.thread_pool_size on startup.

Entering create_app()'s lifespan directly (rather than routing through
TestClient) avoids needing a portal token just to read the limiter back —
we're testing the lifespan's own effect on its own loop, not a real request.
"""
from __future__ import annotations

import anyio.to_thread
import pytest

from vecgrep.backend import config as cfg_mod
from vecgrep.backend.main import create_app


@pytest.mark.anyio
async def test_thread_limiter_capped_to_default_on_startup(vg_home) -> None:
    app = create_app()
    async with app.router.lifespan_context(app):
        limiter = anyio.to_thread.current_default_thread_limiter()
        assert limiter.total_tokens == 8  # Settings.thread_pool_size default


@pytest.mark.anyio
async def test_thread_limiter_honors_env_override(vg_home, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECGREP_THREAD_POOL_SIZE", "3")
    monkeypatch.setattr(cfg_mod, "_settings", None)
    app = create_app()
    async with app.router.lifespan_context(app):
        limiter = anyio.to_thread.current_default_thread_limiter()
        assert limiter.total_tokens == 3


def test_thread_pool_size_setting_defaults_to_eight() -> None:
    from vecgrep.backend.config import Settings

    assert Settings().thread_pool_size == 8
