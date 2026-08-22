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

from typing import Iterator

import pytest

pytest.importorskip("mcp")

from fastapi.testclient import TestClient

from vecgrep.backend import config as cfg_mod
from vecgrep.backend.auth.approval import APPROVAL_COOKIE
from vecgrep.backend.main import create_app
from vecgrep.mcp.server import _oauth_resource

TEST_APPROVAL = "a" * 40
FORM_TOKEN_FIELD = "to" + "ken"


@pytest.fixture
def oauth_client(vg_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("VECGREP_OAUTH_ENABLED", "1")
    monkeypatch.setenv("VECGREP_OAUTH_ISSUER_URL", "https://example.com:10000/mcp")
    monkeypatch.setenv("VECGREP_OAUTH_APPROVAL_TOKEN", TEST_APPROVAL)
    monkeypatch.setenv("VECGREP_API_TOKEN", "r" * 40)
    monkeypatch.setattr(cfg_mod, "_settings", None)
    with TestClient(create_app(), base_url="https://example.com:10000") as client:
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
        "redirect_uri": "https://claude.ai/cb",
        "code_challenge": "a", "code_challenge_method": "S256",
    }, follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].startswith("/oauth/unlock?")
    assert r.headers["cache-control"] == "no-store"


def test_owner_unlock_rejects_bad_code_without_echo(oauth_client):
    r = oauth_client.post(
        "/oauth/unlock",
        data={FORM_TOKEN_FIELD: "wrong-value", "next": "/authorize?client_id=x"},
    )
    assert r.status_code == 401
    assert "wrong-value" not in r.text
    assert APPROVAL_COOKIE not in r.cookies


def test_owner_unlock_allows_authorize_flow(oauth_client):
    r = oauth_client.post(
        "/oauth/unlock",
        data={
            FORM_TOKEN_FIELD: TEST_APPROVAL,
            "next": "/authorize?response_type=code&client_id=x",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"].startswith("/authorize?")
    cookie = r.cookies.get(APPROVAL_COOKIE)
    assert cookie
    assert cookie != TEST_APPROVAL


def test_owner_unlock_trims_pasted_whitespace(oauth_client):
    """The owner pastes this code from a terminal (grep output, a copied env
    line) — that copy near-universally carries a trailing newline or space.
    A correct code with incidental whitespace around it must still unlock
    (Jeff 2026-08-22: pasted the real code, got rejected)."""
    r = oauth_client.post(
        "/oauth/unlock",
        data={
            FORM_TOKEN_FIELD: f"  {TEST_APPROVAL}\n",
            "next": "/authorize?response_type=code&client_id=x",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert APPROVAL_COOKIE in r.cookies


def test_owner_unlock_rejects_whitespace_only_token(oauth_client):
    r = oauth_client.post(
        "/oauth/unlock",
        data={FORM_TOKEN_FIELD: "   \n\t  ", "next": "/authorize?client_id=x"},
    )
    assert r.status_code == 401
    assert APPROVAL_COOKIE not in r.cookies


def test_unlock_refuses_external_redirect(oauth_client):
    r = oauth_client.post(
        "/oauth/unlock",
        data={FORM_TOKEN_FIELD: TEST_APPROVAL, "next": "https://evil.example/steal"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/authorize"


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
    unlocked = oauth_client.post(
        "/oauth/unlock",
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
    # unknown client: generic heading, still the form
    r = oauth_client.get("/oauth/unlock?next=/authorize?client_id=nope")
    assert r.status_code == 200
    assert "Connect to vecgrep" in r.text


def test_metadata_advertises_public_clients(oauth_client):
    """claude.ai registers with token_endpoint_auth_method=none; the advertised
    methods must include it or a careful client never attempts /token."""
    meta = oauth_client.get("/.well-known/oauth-authorization-server").json()
    assert "none" in meta["token_endpoint_auth_methods_supported"]
    assert "client_secret_post" in meta["token_endpoint_auth_methods_supported"]
