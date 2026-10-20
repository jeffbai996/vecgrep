"""The registration ingress buffers the body, then gets out of the way.

After the buffered body the application must see the client's own receive(),
so a loop that waits for http.disconnect blocks instead of spinning on
synthetic empty bodies and starving the event loop.
"""
from __future__ import annotations

import pytest

pytest.importorskip("mcp")


def test_registration_replay_hands_receive_back_to_the_client_after_the_body():
    """After the buffered body the app must see the client's own receive(), so a
    loop waiting for http.disconnect blocks instead of spinning on synthetic
    empty bodies and starving the event loop."""
    import asyncio
    from vecgrep.mcp.server import _RegistrationIngress

    seen: list[dict] = []

    async def inner(scope, receive, send):
        seen.append(await receive())          # the replayed body
        seen.append(await receive())          # must come from the client
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    messages = iter([
        {"type": "http.request", "body": b'{"client_name": "t"}', "more_body": False},
        {"type": "http.disconnect"},
    ])

    async def receive():
        return next(messages)

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/register",
             "headers": [(b"content-length", b"19")]}
    asyncio.run(_RegistrationIngress(inner)(scope, receive, send))
    assert seen[0]["body"] == b'{"client_name": "t"}'
    assert seen[1] == {"type": "http.disconnect"}
    assert sent[0]["status"] == 200
