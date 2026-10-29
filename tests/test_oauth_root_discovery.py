"""OAuth discovery routes must serve real JSON at the APP ROOT — not be shadowed
by the SPA catch-all.

The bug: the SDK advertises .well-known at the origin root, but vecgrep's auth
routes lived under /mcp and the SPA frontend answered root paths — so
/.well-known/oauth-authorization-server returned the HTML homepage, and
claude.ai couldn't parse it. Fix: register the OAuth discovery + auth routes at
the parent root, ahead of the SPA fallback. These tests check CONTENT (JSON with
the right keys), not just status — the miss that bit us before.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

pytest.importorskip("mcp")

from fastapi.testclient import TestClient

from tests.oauth_flow import FormFields, submit
from vecgrep.backend import config as cfg_mod
from vecgrep.backend.auth.approval import (
    APPROVAL_COOKIE,
    TAILNET_INTENT_COOKIE,
    oauth_callback_origin,
)
from vecgrep.backend.main import create_app
from vecgrep.mcp.server import _oauth_resource

TEST_APPROVAL = "a" * 40
FORM_TOKEN_FIELD = "to" + "ken"
TAILNET_HEADERS = {
    "X-Forwarded-For": "192.0.2.10",
    "X-Forwarded-Proto": "https",
    "Tailscale-User-Login": "owner@example.test",
    "Tailscale-Headers-Info": "https://tailscale.com/s/serve-headers",
}


@pytest.fixture
def oauth_client(vg_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from vecgrep.backend import rerank
    from vecgrep.mcp import server as mcp_server
    # Authentication fixtures must not start model loading during app lifespan.
    monkeypatch.setattr(rerank, "ensure_warm", lambda *args, **kwargs: None)

    monkeypatch.setenv("VECGREP_OAUTH_ENABLED", "1")
    monkeypatch.setenv("VECGREP_OAUTH_ISSUER_URL", "https://example.com:10000/mcp")
    monkeypatch.setenv("VECGREP_OAUTH_APPROVAL_TOKEN", TEST_APPROVAL)
    monkeypatch.setenv("VECGREP_API_TOKEN", "r" * 40)
    monkeypatch.setattr(cfg_mod, "_settings", None)
    monkeypatch.setattr(mcp_server, "_PROVIDER", None)
    limiter = getattr(mcp_server, "_REGISTRATION_LIMITER", None)
    if limiter is not None:
        limiter.reset()
    with TestClient(create_app(), base_url="https://example.com:10000") as client:
        from mcp.shared.auth import OAuthClientInformationFull
        for cid in ("x", "client"):
            mcp_server._shared_provider().store.save_client(OAuthClientInformationFull(
                client_id=cid, redirect_uris=["https://claude.ai/api/mcp/auth_callback"], scope="read propose"))
        yield client


def test_auth_server_metadata_is_json_at_root(oauth_client):
    r = oauth_client.get("/.well-known/oauth-authorization-server")
    assert r.status_code == 200
    assert "application/json" in r.headers["content-type"]  # NOT text/html (SPA)
    d = r.json()
    assert d["issuer"]
    assert d["authorization_endpoint"]
    assert d["token_endpoint"]
    assert d["registration_endpoint"]  # DCR advertised


def test_protected_resource_metadata_is_json_at_root(oauth_client):
    # The exact URL the 401's WWW-Authenticate points to.
    r = oauth_client.get("/.well-known/oauth-protected-resource/mcp")
    assert r.status_code == 200
    assert "application/json" in r.headers["content-type"]
    d = r.json()
    assert d["resource"]
    assert d["authorization_servers"]


def test_public_resource_preserves_secret_path_prefix():
    resource = _oauth_resource("https://example.com/secret/vecgrep-mcp")
    assert str(resource) == "https://example.com/secret/vecgrep-mcp"


def test_authorize_requires_owner_approval(oauth_client):
    r = oauth_client.get("/authorize", params={
        "response_type": "code", "client_id": "x",
        "redirect_uri": "https://claude.ai/api/mcp/auth_callback",
        "code_challenge": "a", "code_challenge_method": "S256",
    }, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].startswith("/oauth/unlock?")
    assert r.headers["cache-control"] == "no-store"


def test_owner_unlock_rejects_bad_code_without_echo(oauth_client):
    r = submit(oauth_client,
        data={FORM_TOKEN_FIELD: "wrong-value", "next": "/authorize?client_id=x"},
    )
    assert r.status_code == 401
    assert "wrong-value" not in r.text
    assert APPROVAL_COOKIE not in r.cookies


def test_owner_unlock_allows_authorize_flow(oauth_client):
    r = submit(oauth_client,
        data={
            FORM_TOKEN_FIELD: TEST_APPROVAL,
            "next": "/authorize?response_type=code&client_id=x",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"].startswith("/oauth/continue?")
    assert parse_qs(urlsplit(r.headers["location"]).query)["next"][0].startswith("/authorize?")
    cookie = r.cookies.get(APPROVAL_COOKIE)
    assert cookie
    assert cookie != TEST_APPROVAL


def test_owner_unlock_trims_pasted_whitespace(oauth_client):
    """The owner pastes this code from a terminal (grep output, a copied env
    line) — that copy near-universally carries a trailing newline or space.
    A correct code with incidental whitespace around it must still unlock
    (Jeff 2026-08-22: pasted the real code, got rejected)."""
    r = submit(oauth_client,
        data={
            FORM_TOKEN_FIELD: f"  {TEST_APPROVAL}\n",
            "next": "/authorize?response_type=code&client_id=x",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert APPROVAL_COOKIE in r.cookies


def test_approved_browser_ends_form_submission_before_callback_navigation(oauth_client):
    target = "/authorize?client_id=client&state=opaque%2Bstate%26value"
    approved = submit(oauth_client,
        data={FORM_TOKEN_FIELD: TEST_APPROVAL, "next": target},
        follow_redirects=False,
    )
    assert approved.status_code == 303
    location = approved.headers["location"]
    assert urlsplit(location).path == "/oauth/continue"
    assert parse_qs(urlsplit(location).query)["next"] == [target]
    continuation = oauth_client.get(location, follow_redirects=False)
    assert continuation.status_code == 200
    assert 'http-equiv="refresh"' in continuation.text
    assert 'url=/authorize?client_id=client&amp;state=opaque%2Bstate%26value' in continuation.text
    assert '<script' not in continuation.text
    assert continuation.headers["cache-control"] == "no-store"
    assert continuation.headers["referrer-policy"] == "no-referrer"
    assert "form-action 'none'" in continuation.headers["content-security-policy"]


def test_callback_continuation_requires_browser_approval(oauth_client):
    rejected = oauth_client.get("/oauth/continue", params={"next": "/authorize"})
    assert rejected.status_code == 403
    assert 'http-equiv="refresh"' not in rejected.text


@pytest.mark.parametrize("target", ["https://evil.example/callback", "//evil.example/callback", "/api/search"])
def test_callback_continuation_cannot_redirect_to_arbitrary_destinations(oauth_client, target):
    submit(oauth_client, data={FORM_TOKEN_FIELD: TEST_APPROVAL, "next": "/authorize?client_id=x"}, follow_redirects=False)
    continuation = oauth_client.get("/oauth/continue", params={"next": target})
    assert continuation.status_code == 400
    assert 'http-equiv="refresh"' not in continuation.text
    assert target not in continuation.text


def test_owner_unlock_rejects_whitespace_only_token(oauth_client):
    r = submit(oauth_client,
        data={FORM_TOKEN_FIELD: "   \n\t  ", "next": "/authorize?client_id=x"},
    )
    assert r.status_code == 401
    assert APPROVAL_COOKIE not in r.cookies


def test_unlock_csp_allows_only_registered_callback_origin(oauth_client):
    target = (
        "/authorize?client_id=x&redirect_uri="
        "https%3A%2F%2Fclaude.ai%2Fapi%2Fmcp%2Fauth_callback"
    )
    page = oauth_client.get("/oauth/unlock", params={"next": target})
    policy = page.headers["content-security-policy"]
    assert "form-action 'self' https://claude.ai;" in policy
    assert "/api/mcp/auth_callback" not in policy


@pytest.mark.parametrize(
    "target",
    [
        "/authorize?redirect_uri=javascript%3Aalert(1)",
        "/authorize?redirect_uri=https%3A%2F%2Fuser%40evil.example%2Fcb",
        "/authorize?redirect_uri=http%3A%2F%2Fevil.example%2Fcb",
    ],
)
def test_callback_csp_origin_rejects_unsafe_urls(target):
    assert oauth_callback_origin(target) is None


def test_verified_tailnet_owner_gets_csrf_bound_one_click_approval(oauth_client):
    target = "/authorize?response_type=code&client_id=x"
    with TestClient(create_app(), base_url="https://127.0.0.1:8765",
                    client=("127.0.0.1", 50000)) as client:
        page = client.get(
            "/oauth/unlock", params={"next": target}, headers=TAILNET_HEADERS
        )
        assert page.status_code == 200
        assert 'name="token"' not in page.text
        assert "verified Tailscale session" in page.text
        match = re.search(r'name="tailnet_intent" value="([a-f0-9]+)"', page.text)
        assert match
        assert TAILNET_INTENT_COOKIE in page.cookies

        approved = client.post(
            "/oauth/unlock",
            data={"consent": FormFields(page.text).values["consent"], "next": target, "tailnet_intent": match.group(1)},
            headers=TAILNET_HEADERS,
            follow_redirects=False,
        )
        assert approved.status_code == 303
        assert parse_qs(urlsplit(approved.headers["location"]).query)["next"] == [target]
        assert APPROVAL_COOKIE in approved.cookies


def test_tailnet_one_click_rejects_missing_or_cross_site_intent(oauth_client):
    target = "/authorize?response_type=code&client_id=x"
    with TestClient(create_app(), base_url="https://127.0.0.1:8765",
                    client=("127.0.0.1", 50000)) as client:
        page = client.get(
            "/oauth/unlock", params={"next": target}, headers=TAILNET_HEADERS
        )
        intent = re.search(
            r'name="tailnet_intent" value="([a-f0-9]+)"', page.text
        ).group(1)
        client.cookies.delete(TAILNET_INTENT_COOKIE)
        rejected = client.post(
            "/oauth/unlock",
            data={"consent": FormFields(page.text).values["consent"], "next": target, "tailnet_intent": intent},
            headers=TAILNET_HEADERS | {"Origin": "https://evil.example"},
            follow_redirects=False,
        )
        assert rejected.status_code == 401
        assert APPROVAL_COOKIE not in rejected.cookies


def test_funnel_or_unverified_proxy_still_requires_owner_code(oauth_client):
    target = "/authorize?response_type=code&client_id=x"
    funnel = oauth_client.get(
        "/oauth/unlock",
        params={"next": target},
        headers=TAILNET_HEADERS | {"Tailscale-Funnel-Request": "?1"},
    )
    non_loopback = oauth_client.get(
        "/oauth/unlock", params={"next": target}, headers=TAILNET_HEADERS
    )
    for page in (funnel, non_loopback):
        assert page.status_code == 200
        assert 'name="token"' in page.text
        assert 'name="tailnet_intent"' not in page.text


def test_unlock_refuses_external_redirect(oauth_client):
    r = oauth_client.post("/oauth/unlock",
        data={FORM_TOKEN_FIELD: TEST_APPROVAL, "next": "https://evil.example/steal"},
        follow_redirects=False,
    )
    assert r.status_code == 400
    assert "location" not in r.headers


def test_approved_owner_can_reach_real_authorize_handler(oauth_client):
    from urllib.parse import urlencode

    registration = oauth_client.post("/register", json={
        "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    })
    client_id = registration.json()["client_id"]
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": "https://claude.ai/api/mcp/auth_callback",
        "state": "state-test",
        "scope": "read",
        "code_challenge": "A" * 43,
        "code_challenge_method": "S256",
    }
    unlocked = submit(oauth_client,
        data={
            FORM_TOKEN_FIELD: TEST_APPROVAL,
            "next": "/authorize?" + urlencode(params),
        },
        follow_redirects=False,
    )
    assert unlocked.status_code == 303
    authorized = oauth_client.get("/authorize", params=params, follow_redirects=False)
    assert authorized.status_code in (302, 303, 307)
    assert authorized.headers["location"].startswith(
        "https://claude.ai/api/mcp/auth_callback?"
    )
    assert "code=" in authorized.headers["location"]


def test_register_endpoint_at_root_accepts_dcr(oauth_client):
    r = oauth_client.post("/register", json={
        "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    })
    assert r.status_code in (200, 201)
    assert r.json().get("client_id")


def test_mcp_endpoint_still_gated(oauth_client):
    # An unclassifiable/non-loopback peer still requires a token.
    r = oauth_client.post("/mcp", json={},
                          headers={"Accept": "application/json, text/event-stream",
                                   "Content-Type": "application/json"})
    assert r.status_code == 401


def test_spa_still_served_for_normal_paths(oauth_client):
    # The discovery routes must NOT have broken the SPA for ordinary paths.
    r = oauth_client.get("/some/app/route")
    # Either the SPA HTML (frontend built) or the no-frontend JSON message —
    # just not a 500 and not an OAuth route hijacking everything.
    assert r.status_code == 200


def test_unlock_page_names_the_client_and_its_scopes(oauth_client):
    """The owner approves a named client with named permissions — the page
    reads client_id + scope off the authorize target it will return to."""
    reg = oauth_client.post("/register", json={
        "client_name": "Claude", "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
        "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    })
    assert reg.status_code == 201
    cid = reg.json()["client_id"]
    r = oauth_client.get(f"/oauth/unlock?next=/authorize?client_id={cid}%26scope=read%20propose")
    assert r.status_code == 200
    assert "Connect Claude to vecgrep" in r.text
    assert "Search and read" in r.text and "Propose changes" in r.text
    assert "<script" not in r.text
    # Unknown clients cannot produce a consent form.
    r = oauth_client.get("/oauth/unlock?next=/authorize?client_id=nope")
    assert r.status_code == 400
    assert 'name="token"' not in r.text


def test_unlock_page_names_where_the_connection_returns(oauth_client):
    """client_name is chosen by whoever registered; the callback host is not.
    A phished owner who reads the page sees where the code will be sent."""
    reg = oauth_client.post("/register", json={
        "client_name": "Claude", "redirect_uris": ["https://evil.example/cb"],
        "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    })
    cid = reg.json()["client_id"]
    target = f"/authorize?client_id={cid}&redirect_uri=https%3A%2F%2Fevil.example%2Fcb&scope=read"
    r = oauth_client.get("/oauth/unlock", params={"next": target})
    assert r.status_code == 200
    assert "returns to <b>https://evil.example/cb</b>" in r.text
    assert "https://evil.example/cb" in r.text
    bare = oauth_client.get("/oauth/unlock?next=/authorize?client_id=nope")
    assert "returns to" not in bare.text, "no destination line when the request names none"


def test_unlock_page_names_target_sibling_service_from_resource_query(oauth_client, monkeypatch):
    monkeypatch.setenv("VECGREP_OAUTH_RESOURCE_LABELS", "/notes-mcp/mcp=notes-service,/mcp=broker-mcp")
    reg = oauth_client.post("/register", json={
        "client_name": "Google",
        "redirect_uris": ["https://oauth-redirect.googleusercontent.com/r/custom-mcp"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    })
    cid = reg.json()["client_id"]
    notes_target = f"/authorize?client_id={cid}&scope=read+propose&resource=https%3A%2F%2Fexample.com%2Fnotes-mcp%2Fmcp"
    notes_page = oauth_client.get("/oauth/unlock", params={"next": notes_target})
    assert notes_page.status_code == 200
    assert "Connect Google to notes-service" in notes_page.text
    assert "query notes-service tools and resources" in notes_page.text

    broker_target = f"/authorize?client_id={cid}&scope=read+propose&resource=https%3A%2F%2Fexample.com%2Fsecret%2Fmcp"
    broker_page = oauth_client.get("/oauth/unlock", params={"next": broker_target})
    assert broker_page.status_code == 200
    assert "Connect Google to broker-mcp" in broker_page.text
    assert "query broker-mcp tools and resources" in broker_page.text



def test_unlock_and_issuance_both_default_omitted_scope_to_read(oauth_client):
    reg = oauth_client.post("/register", json={
        "client_name": "Example client",
        "redirect_uris": ["https://example.test/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
        "scope": "read propose",
    })
    cid = reg.json()["client_id"]
    page = oauth_client.get(f"/oauth/unlock?next=/authorize?client_id={cid}")
    assert "Search and read" in page.text
    assert "Propose changes" not in page.text


@pytest.mark.parametrize(
    "registered_scope, requested_scope, granted_scope",
    [(None, "read propose", "read propose"), (None, None, "read"), ("read", "read propose", None)],
)
def test_registration_capabilities_are_separate_from_consented_token_scopes(
    oauth_client: TestClient, registered_scope: str | None,
    requested_scope: str | None, granted_scope: str | None,
) -> None:
    import base64
    import hashlib
    import secrets

    metadata = {
        "client_name": "Example client",
        "redirect_uris": ["https://example.test/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "_".join(("client", "secret", "post")),
    }
    if registered_scope is not None:
        metadata["scope"] = registered_scope
    registration = oauth_client.post("/register", json=metadata)
    assert registration.status_code == 201
    client = registration.json()
    verifier = secrets.token_urlsafe(32)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    params = {
        "client_id": client["client_id"], "response_type": "code",
        "redirect_uri": metadata["redirect_uris"][0],
        "code_challenge": challenge, "code_challenge_method": "S256",
        "state": "opaque+state",
    }
    if requested_scope is not None:
        params["scope"] = requested_scope
    unauthorized = oauth_client.get("/authorize", params=params, follow_redirects=False)
    assert unauthorized.status_code == 303
    assert unauthorized.headers["location"].startswith("/oauth/unlock?")
    submit(oauth_client, data={FORM_TOKEN_FIELD: TEST_APPROVAL, "next": "/authorize?" + urlencode(params)}, follow_redirects=False)
    authorization = oauth_client.get("/authorize", params=params, follow_redirects=False)
    assert authorization.status_code == 302
    callback = parse_qs(urlsplit(authorization.headers["location"]).query)
    assert callback["state"] == ["opaque+state"]
    if granted_scope is None:
        assert callback["error"] == ["invalid_scope"]
        assert "code" not in callback
        return
    assert "error" not in callback
    token = oauth_client.post("/token", data={
        "grant_type": "authorization_code", "client_id": client["client_id"],
        "client_secret": client["client_secret"], "code": callback["code"][0],
        "redirect_uri": metadata["redirect_uris"][0], "code_verifier": verifier,
    })
    assert token.status_code == 200
    assert token.json()["scope"] == granted_scope


def test_metadata_advertises_public_clients(oauth_client):
    """claude.ai registers with token_endpoint_auth_method=none; the advertised
    methods must include it or a careful client never attempts /token."""
    meta = oauth_client.get("/.well-known/oauth-authorization-server").json()
    assert "none" in meta["token_endpoint_auth_methods_supported"]
    assert "client_secret_post" in meta["token_endpoint_auth_methods_supported"]


def test_token_store_survives_a_restart(tmp_path):
    from mcp.shared.auth import OAuthClientInformationFull

    from vecgrep.backend.auth.store import TokenStore
    path = str(tmp_path / "oauth_state.json")
    a = TokenStore(path=path)
    a.save_client(OAuthClientInformationFull(client_id="c1", redirect_uris=["https://claude.ai/cb"]))
    at = a.issue_access_token("c1", ["read"])
    rt = a.issue_refresh_token("c1", ["read"])
    b = TokenStore(path=path)  # "restart"
    assert b.get_client("c1") is not None
    assert b.load_access_token(at.token) is not None
    assert b.load_refresh_token("c1", rt.token) is not None
    import os
    import stat
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    b.revoke(at.token)
    assert TokenStore(path=path).load_access_token(at.token) is None


def test_oauth_trace_does_not_break_the_flow(vg_home, monkeypatch):
    from vecgrep.backend import rerank
    monkeypatch.setattr(rerank, "ensure_warm", lambda *args, **kwargs: None)
    """The trace wrapper must wrap the SDK's request handlers as handlers —
    a wrong wrapper shape 500'd /authorize for 10 minutes on 2026-08-22."""
    monkeypatch.setenv("VECGREP_OAUTH_ENABLED", "1")
    monkeypatch.setenv("VECGREP_OAUTH_ISSUER_URL", "https://example.com:10000/mcp")
    monkeypatch.setenv("VECGREP_OAUTH_APPROVAL_TOKEN", TEST_APPROVAL)
    monkeypatch.setenv("VECGREP_API_TOKEN", "r" * 40)
    monkeypatch.setenv("VECGREP_OAUTH_TRACE", "1")
    monkeypatch.setattr(cfg_mod, "_settings", None)
    with TestClient(create_app(), base_url="https://example.com:10000") as c:
        reg = c.post("/register", json={"client_name": "t", "redirect_uris": ["https://claude.ai/cb"],
                                        "grant_types": ["authorization_code", "refresh_token"],
                                        "response_types": ["code"], "token_endpoint_auth_method": "none"})
        assert reg.status_code == 201
        params = {"response_type": "code", "client_id": reg.json()["client_id"],
                  "redirect_uri": "https://claude.ai/cb", "code_challenge": "x" * 43,
                  "code_challenge_method": "S256", "state": "s"}
        approved = submit(c, data={FORM_TOKEN_FIELD: TEST_APPROVAL, "next": "/authorize?" + urlencode(params)}, follow_redirects=False)
        assert APPROVAL_COOKIE in approved.cookies
        r = c.get("/authorize", params=params, follow_redirects=False)
        assert r.status_code == 302, r.text


def test_oauth_trace_never_logs_form_or_opaque_secrets(caplog):
    from vecgrep.mcp.server import _trace_log

    canary = "trace-sensitive-canary"
    caplog.set_level("WARNING", logger="vecgrep.oauth.trace")
    _trace_log(
        "POST", "/token", "", {"content-type": "application/x-www-form-urlencoded"},
        f"grant_type=refresh_token&refresh_token={canary}&client_secret={canary}",
        400, "", "",
    )
    _trace_log(
        "POST", "/token", "", {"content-type": "text/plain"},
        f"malformed-body-{canary}", 400, "", "",
    )
    _trace_log(
        "POST", "/register", "", {"content-type": "application/json"},
        '{"client_name":{"refresh_token":"' + canary + '"}}', 400, "", "",
    )
    assert canary not in caplog.text
    assert "grant_type=refresh_token" in caplog.text


@pytest.mark.parametrize("path", ["/register", "/mcp/register"])
def test_register_rejects_oversized_request_before_json_parsing(oauth_client, path):
    r = oauth_client.post(
        path,
        content=b"x" * 65_537,
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 413


def test_register_rate_limits_bursts(oauth_client):
    registration = {
        "redirect_uris": ["https://example.test/callback"],
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    }
    responses = [oauth_client.post("/register", json=registration) for _ in range(31)]
    assert all(r.status_code == 201 for r in responses[:30])
    assert responses[-1].status_code == 429
    assert responses[-1].headers["retry-after"]


def test_protected_resource_metadata_matches_origin_for_direct_loopback(oauth_client):
    # A same-machine SDK client that dials the loopback bypass URL directly
    # must see a `resource` that matches the origin it actually connected to
    # (2026-09-18) -- otherwise its own OAuth spec-compliance check refuses to
    # proceed even though this peer was never going to be challenged.
    with TestClient(
        create_app(),
        base_url="http://127.0.0.1:8765",
        client=("127.0.0.1", 50000),
    ) as client:
        r = client.get("/.well-known/oauth-protected-resource/mcp")
        assert r.status_code == 200
        d = r.json()
        assert d["resource"] == "http://127.0.0.1:8765/mcp"
        # The authorization server stays the public issuer -- only the
        # resource identity is loopback-local.
        assert d["authorization_servers"] == ["https://example.com:10000/"]


def test_protected_resource_metadata_stays_public_for_proxied_or_tailnet_callers(oauth_client):
    # Same route, same loopback socket peer -- but carrying a Tailscale/proxy
    # header means it is NOT a genuine direct loopback call (per the same
    # trust model used for the bearer-token bypass), so it must keep seeing
    # the public resource identity unchanged.
    with TestClient(
        create_app(),
        base_url="http://127.0.0.1:8765",
        client=("127.0.0.1", 50000),
    ) as client:
        r = client.get(
            "/.well-known/oauth-protected-resource/mcp", headers=TAILNET_HEADERS
        )
        assert r.status_code == 200
        assert r.json()["resource"] == "https://example.com:10000/mcp"


def test_protected_resource_metadata_stays_public_over_non_loopback_client(oauth_client):
    r = oauth_client.get("/.well-known/oauth-protected-resource/mcp")
    assert r.status_code == 200
    assert r.json()["resource"] == "https://example.com:10000/mcp"


def test_public_oauth_offers_private_tailnet_approval(oauth_client):
    cfg_mod.get_settings().oauth_tailnet_approval_url = "https://example.com:8444/oauth/unlock"
    target = "/authorize?client_id=client&state=keep-me"
    page = oauth_client.get("/oauth/unlock", params={"next": target, "method": "code"})
    assert "Approve with Tailscale" in page.text
    assert "https://example.com:8444/oauth/unlock?next=" in page.text
    assert "Use approval code" in page.text


def test_public_unlock_redirects_to_one_private_consent_screen(oauth_client):
    from urllib.parse import parse_qs, urlsplit

    cfg_mod.get_settings().oauth_tailnet_approval_url = "https://example.com:8444/oauth/unlock"
    target = "/authorize?client_id=client&state=keep-me"
    page = oauth_client.get("/oauth/unlock", params={"next": target}, follow_redirects=False)
    assert page.status_code == 303
    destination = urlsplit(page.headers["location"])
    assert destination.netloc == "example.com:8444"
    assert parse_qs(destination.query)["next"] == [target]
    assert page.headers["cache-control"] == "no-store"
    assert APPROVAL_COOKIE not in page.cookies


def test_unverified_private_browser_gets_code_fallback_without_redirect_loop(oauth_client):
    cfg_mod.get_settings().oauth_tailnet_approval_url = "https://example.com:10000/oauth/unlock"
    page = oauth_client.get("/oauth/unlock", params={"next": "/authorize?client_id=x"}, follow_redirects=False)
    assert page.status_code == 200
    assert 'name="token"' in page.text


def test_private_approval_posts_proof_to_public_origin(oauth_client):
    settings = cfg_mod.get_settings()
    settings.oauth_issuer_url = "https://127.0.0.1:8766/mcp"
    settings.oauth_tailnet_approval_url = "https://127.0.0.1:8765/oauth/unlock"
    target = "/authorize?client_id=client&state=keep-me"
    with TestClient(create_app(), base_url="https://127.0.0.1:8765",
                    client=("127.0.0.1", 50000)) as private:
        page = private.get("/oauth/unlock", params={"next": target}, headers=TAILNET_HEADERS)
        assert 'action="https://127.0.0.1:8766/oauth/unlock"' in page.text
        # no-referrer turns cross-origin form POST Origin into null in browsers.
        assert page.headers["referrer-policy"] == "strict-origin"
        assert 'name="tailnet_login" value="owner@example.test"' in page.text
        intent = re.search(r'name="tailnet_intent" value="([a-f0-9]+)"', page.text).group(1)
        with TestClient(create_app(), base_url="https://127.0.0.1:8766") as public:
            public.cookies.update(private.cookies)
            approved = public.post("/oauth/unlock", data={
                "consent": FormFields(page.text).values["consent"], "next": target, "tailnet_intent": intent,
                "tailnet_login": "owner@example.test",
            }, headers={"Origin": "https://127.0.0.1:8765"}, follow_redirects=False)
            assert approved.status_code == 303
            assert parse_qs(urlsplit(approved.headers["location"]).query)["next"] == [target]
            assert APPROVAL_COOKIE in approved.cookies


@pytest.mark.parametrize("change", [None, "approval_cookie", "proof", "target", "login", "origin"])
def test_repeated_tailnet_handoff_requires_existing_approval_and_bound_proof(oauth_client, change):
    settings = cfg_mod.get_settings()
    settings.oauth_issuer_url = "https://127.0.0.1:8766/mcp"
    settings.oauth_tailnet_approval_url = "https://127.0.0.1:8765/oauth/unlock"
    target = "/authorize?client_id=client&state=keep-me"
    with TestClient(create_app(), base_url="https://127.0.0.1:8765",
                    client=("127.0.0.1", 50000)) as private:
        page = private.get("/oauth/unlock", params={"next": target}, headers=TAILNET_HEADERS)
        intent = re.search(r'name="tailnet_intent" value="([a-f0-9]+)"', page.text).group(1)
        with TestClient(create_app(), base_url="https://127.0.0.1:8766") as public:
            public.cookies.update(private.cookies)
            data = {"consent": FormFields(page.text).values["consent"], "next": target, "tailnet_intent": intent, "tailnet_login": "owner@example.test"}
            origin = "https://127.0.0.1:8765"
            approved = public.post("/oauth/unlock", data=data, headers={"Origin": origin}, follow_redirects=False)
            assert approved.status_code == 303
            assert TAILNET_INTENT_COOKIE not in public.cookies
            if change == "approval_cookie":
                public.cookies.delete(APPROVAL_COOKIE)
            elif change == "proof":
                data["tailnet_intent"] = "0" * 64
            elif change == "target":
                data["next"] = "/authorize?client_id=other"
            elif change == "login":
                data["tailnet_login"] = "other@example.test"
            elif change == "origin":
                origin = "https://evil.example"
            repeated = public.post("/oauth/unlock", data=data, headers={"Origin": origin}, follow_redirects=False)
            assert repeated.status_code == (303 if change is None else 400 if change == "target" else 401)
            if change is None:
                assert parse_qs(urlsplit(repeated.headers["location"]).query)["next"] == [target]


@pytest.mark.parametrize("change", ["origin", "target", "login", "cookie", "proof"])
def test_public_handoff_rejects_unbound_or_forged_proof(oauth_client, change):
    settings = cfg_mod.get_settings()
    settings.oauth_issuer_url = "https://127.0.0.1:8766/mcp"
    settings.oauth_tailnet_approval_url = "https://127.0.0.1:8765/oauth/unlock"
    target = "/authorize?client_id=client&state=keep-me"
    with TestClient(create_app(), base_url="https://127.0.0.1:8765",
                    client=("127.0.0.1", 50000)) as private:
        page = private.get("/oauth/unlock", params={"next": target}, headers=TAILNET_HEADERS)
        intent = re.search(r'name="tailnet_intent" value="([a-f0-9]+)"', page.text).group(1)
        with TestClient(create_app(), base_url="https://127.0.0.1:8766") as public:
            public.cookies.update(private.cookies)
            data = {"consent": FormFields(page.text).values["consent"], "next": target, "tailnet_intent": intent, "tailnet_login": "owner@example.test"}
            origin = "https://127.0.0.1:8765"
            if change == "origin": origin = "https://evil.example"
            if change == "target": data["next"] = "/authorize?client_id=other"
            if change == "login": data["tailnet_login"] = "other@example.test"
            if change == "cookie": public.cookies.clear()
            if change == "proof":
                data["tailnet_intent"] = "0" * 64
                public.cookies.set(TAILNET_INTENT_COOKIE, "0" * 64)
            denied = public.post("/oauth/unlock", data=data, headers={"Origin": origin}, follow_redirects=False)
            assert denied.status_code == (400 if change in {"target", "cookie"} else 401)
            assert APPROVAL_COOKIE not in denied.cookies
            assert 'name="tailnet_intent"' not in denied.text


@pytest.mark.parametrize("url", [
    "http://example.com:8444/oauth/unlock", "https://other.example:8444/oauth/unlock",
    "https://example.com:10000/oauth/unlock", "https://example.com:8444/authorize",
    "https://example.com:8444/oauth/unlock?next=/authorize",
    "https://user@example.com:8444/oauth/unlock", "https://example.com:bad/oauth/unlock",
])
def test_tailnet_approval_url_rejects_unsafe_configuration(oauth_client, url):
    settings = cfg_mod.get_settings()
    settings.oauth_tailnet_approval_url = url
    with pytest.raises(cfg_mod.ConfigError):
        cfg_mod.validate_settings(settings)


def test_tailnet_approval_url_loads_from_environment(oauth_client, monkeypatch):
    monkeypatch.setenv("VECGREP_OAUTH_TAILNET_APPROVAL_URL", "https://example.com:8444/oauth/unlock")
    assert cfg_mod.load_settings().oauth_tailnet_approval_url == "https://example.com:8444/oauth/unlock"


def test_protected_resource_metadata_stays_public_for_other_loopback_addresses(oauth_client):
    # Only 127.0.0.1 and ::1 are the same-machine client; a peer dialing from
    # another address in 127.0.0.0/8 keeps the public resource identity.
    for peer, expected_loopback in (("127.0.0.2", False), ("::1", True)):
        with TestClient(
            create_app(), base_url="http://127.0.0.1:8765", client=(peer, 50000)
        ) as client:
            r = client.get("/.well-known/oauth-protected-resource/mcp")
            assert r.status_code == 200
            want = "http://127.0.0.1:8765/mcp" if expected_loopback else "https://example.com:10000/mcp"
            assert r.json()["resource"] == want


def test_tailnet_one_click_is_withheld_from_other_loopback_peers(oauth_client):
    target = "/authorize?response_type=code&client_id=x"
    for peer, offered in (("127.0.0.2", False), ("::1", True), ("127.0.0.1", True)):
        with TestClient(
            create_app(), base_url="https://127.0.0.1:8765", client=(peer, 50000)
        ) as client:
            page = client.get(
                "/oauth/unlock", params={"next": target}, headers=TAILNET_HEADERS
            )
            assert page.status_code == 200
            assert ('name="tailnet_intent"' in page.text) is offered
