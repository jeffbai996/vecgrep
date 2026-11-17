from __future__ import annotations

import anyio.to_thread
import pytest
from fastapi.testclient import TestClient

from vecgrep.backend import config as cfg_mod
from vecgrep.backend.main import create_app

# ── test_cors_boundary ──

def test_foreign_origin_receives_no_cors_read_permission() -> None:
    with TestClient(create_app(), base_url="http://127.0.0.1:8765") as client:
        response = client.get(
            "/api/health",
            headers={"Origin": "https://evil.example"},
        )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers

# ── test_serve_thread_pool ──

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
