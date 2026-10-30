"""Dynamic client registration is budgeted per caller, not per process."""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp")

from vecgrep.mcp import server


def test_one_caller_spending_its_budget_does_not_block_another():
    limiter = server._RegistrationRateLimiter(limit=3, window_s=60)
    assert [limiter.allow("a")[0] for _ in range(3)] == [True, True, True]
    allowed, retry_after = limiter.allow("a")
    assert allowed is False and retry_after >= 1
    assert limiter.allow("b") == (True, 0)


def test_the_caller_table_is_bounded():
    limiter = server._RegistrationRateLimiter(limit=1, window_s=60, max_keys=4)
    for i in range(10):
        limiter.allow(f"k{i}")
    assert len(limiter._attempts) <= 4
    assert limiter.allow("k9")[0] is False, "the newest caller keeps its count"


async def _drive(app, forwarded: str) -> int:
    sent: list[dict] = []
    scope = {"type": "http", "method": "POST", "path": "/register",
             "headers": [(b"x-forwarded-for", forwarded.encode()), (b"content-length", b"2")],
             "client": ("127.0.0.1", 50000)}
    messages = iter([{"type": "http.request", "body": b"{}", "more_body": False}])

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


def test_registrations_through_the_ingress_are_counted_per_forwarded_address(monkeypatch):
    monkeypatch.setattr(server, "_REGISTRATION_LIMITER", server._RegistrationRateLimiter(limit=2, window_s=60))

    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 201, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    app = server._RegistrationIngress(inner)
    first = [asyncio.run(_drive(app, "198.51.100.7")) for _ in range(3)]
    assert first[2] == 429
    assert asyncio.run(_drive(app, "203.0.113.9")) != 429
