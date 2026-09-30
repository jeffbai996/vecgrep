"""Run private and public ingress with one lifespan and OAuth token store."""
from __future__ import annotations

import contextlib
import os

import uvicorn


def serve_http(*, host: str, port: int, public_port: int, reload: bool) -> None:
    if port == public_port:
        raise ValueError("OAuth public and private listeners must use different ports")
    # CLI --port must agree with the socket classification in the app, also
    # after a reload child imports it in a fresh interpreter.
    os.environ["VECGREP_API_PORT"] = str(port)
    from .config import reset_settings

    settings = reset_settings()
    if not settings.oauth_enabled:
        raise ValueError("Public ingress requires OAuth")
    from ..mcp.server import _require_mcp

    _require_mcp()
    config = uvicorn.Config("vecgrep.backend.main:app", host=host, port=port,
                            reload=reload, timeout_keep_alive=900)
    public_config = uvicorn.Config("vecgrep.backend.main:app", host="127.0.0.1",
                                   port=public_port)
    server = uvicorn.Server(config)
    # Bind both before serving either. A failed bind must not leave a partially
    # initialized deployment; ExitStack also closes sockets on reload/shutdown.
    with contextlib.ExitStack() as stack:
        sockets = [stack.enter_context(config.bind_socket()),
                   stack.enter_context(public_config.bind_socket())]
        if config.should_reload:
            from uvicorn.supervisors import ChangeReload

            ChangeReload(config, target=server.run, sockets=sockets).run()
        else:
            server.run(sockets=sockets)
            if not server.started:
                raise RuntimeError("vecgrep HTTP startup failed")
