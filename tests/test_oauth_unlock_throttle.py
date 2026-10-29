"""The public owner-approval form bounds failed code attempts per client."""
from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("mcp")

from tests.oauth_flow import submit
from vecgrep.backend import config
from vecgrep.backend.auth import throttle
from vecgrep.backend.auth.approval import APPROVAL_COOKIE, approval_cookie_value
from vecgrep.backend.main import create_app
from vecgrep.mcp import server

CODE = "a" * 40
CLIENT_A = "203.0.113.5"
CLIENT_B = "198.51.100.7"


class Listener:
    """Supply the accepted socket address, independently of HTTP Host."""
    def __init__(self, app, port):
        self.app, self.port = app, port

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope = dict(scope, server=("127.0.0.1", self.port))
        await self.app(scope, receive, send)


@pytest.fixture
def public(vg_home, monkeypatch):
    from vecgrep.backend import rerank
    # Authentication fixtures must not start model loading during app lifespan.
    monkeypatch.setattr(rerank, "ensure_warm", lambda *args, **kwargs: None)
    monkeypatch.setenv("VECGREP_OAUTH_ENABLED", "1")
    monkeypatch.setenv("VECGREP_OAUTH_ISSUER_URL", "https://example.com/mcp")
    monkeypatch.setenv("VECGREP_OAUTH_APPROVAL_TOKEN", CODE)
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(server, "_PROVIDER", None)
    clock = {"now": 1_000.0}
    monkeypatch.setattr(throttle.UNLOCK_FAILURES, "_clock", lambda: clock["now"])
    throttle.UNLOCK_FAILURES.reset()
    with TestClient(Listener(create_app(), 8766), base_url="https://example.com",
                    client=("127.0.0.1", 40000)) as client:
        from mcp.shared.auth import OAuthClientInformationFull
        server._shared_provider().store.save_client(OAuthClientInformationFull(
            client_id="x", redirect_uris=["https://client.example/callback"]))
        yield client, clock


def _attempt(client, code, *, ip=CLIENT_A):
    # Funnel requests reach the app with the real client address in
    # X-Forwarded-For, set by tailscaled; the socket peer is loopback.
    return submit(client, data={"token": code, "next": "/authorize?client_id=x"},
                       headers={"X-Forwarded-For": ip}, follow_redirects=False)


def test_wrong_code_is_refused_with_401(public):
    client, _ = public
    assert _attempt(client, "nope").status_code == 401


def test_correct_code_below_the_limit_redirects_through_approved_continuation(public):
    client, _ = public
    for _ in range(9):
        _attempt(client, "nope")
    ok = _attempt(client, CODE)
    assert ok.status_code == 303
    assert ok.headers["location"] == "/oauth/continue?next=%2Fauthorize%3Fclient_id%3Dx"
    location = urlsplit(ok.headers["location"])
    assert location.path == "/oauth/continue"
    assert parse_qs(location.query) == {"next": ["/authorize?client_id=x"]}
    assert ok.cookies[APPROVAL_COOKIE] == approval_cookie_value(CODE)
    assert ok.cookies[APPROVAL_COOKIE] != CODE
    assert ok.headers["cache-control"] == "no-store"
    cookie = next(cookie for cookie in ok.cookies.jar if cookie.name == APPROVAL_COOKIE)
    assert cookie.secure
    assert cookie.has_nonstandard_attr("HttpOnly")
    assert cookie.get_nonstandard_attr("SameSite") == "strict"

    continuation = client.get(ok.headers["location"], follow_redirects=False)
    assert continuation.status_code == 200
    assert 'content="0;url=/authorize?client_id=x"' in continuation.text
    assert continuation.headers["cache-control"] == "no-store"
    assert continuation.headers["referrer-policy"] == "no-referrer"
    assert continuation.headers["content-security-policy"] == (
        "default-src 'none'; form-action 'none'; "
        "base-uri 'none'; frame-ancestors 'none'"
    )


def test_tenth_failure_in_a_minute_locks_that_client_even_for_the_right_code(public):
    client, _ = public
    for _ in range(10):
        assert _attempt(client, "nope").status_code == 401
    locked = _attempt(client, CODE)
    assert locked.status_code == 429
    assert int(locked.headers["retry-after"]) > 0
    assert "Too many attempts" in locked.text
    assert "set-cookie" not in locked.headers


def test_lock_is_per_client_address(public):
    client, _ = public
    for _ in range(10):
        _attempt(client, "nope", ip=CLIENT_A)
    assert _attempt(client, "nope", ip=CLIENT_B).status_code == 401
    assert _attempt(client, CODE, ip=CLIENT_B).status_code == 303


def test_lock_lifts_after_the_window(public):
    client, clock = public
    for _ in range(10):
        _attempt(client, "nope")
    assert _attempt(client, CODE).status_code == 429
    clock["now"] += 61
    assert _attempt(client, CODE).status_code == 303


def test_locked_attempts_do_not_extend_the_window(public):
    client, clock = public
    for _ in range(10):
        _attempt(client, "nope")
    clock["now"] += 30
    assert _attempt(client, CODE).status_code == 429
    clock["now"] += 31
    assert _attempt(client, CODE).status_code == 303


def test_client_key_falls_back_to_the_socket_peer_without_forwarding():
    assert throttle.client_key({"type": "http", "headers": [], "client": ("10.0.0.9", 1)}) \
        == throttle.client_key({"type": "http", "headers": [], "client": ("10.0.0.9", 2)})
    assert throttle.client_key({"type": "http", "headers": [], "client": ("10.0.0.9", 1)}) \
        != throttle.client_key({"type": "http", "headers": [], "client": ("10.0.0.8", 1)})
    forwarded = [(b"x-forwarded-for", b"203.0.113.5, 10.0.0.1")]
    assert throttle.client_key({"type": "http", "headers": forwarded, "client": ("127.0.0.1", 1)}) \
        == throttle.client_key({"type": "http", "headers": [(b"x-forwarded-for", b"203.0.113.5")],
                                "client": ("127.0.0.1", 2)})
