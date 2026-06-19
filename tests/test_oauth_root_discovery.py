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
from vecgrep.backend.main import create_app


@pytest.fixture
def oauth_client(vg_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("VECGREP_OAUTH_ENABLED", "1")
    monkeypatch.setenv("VECGREP_OAUTH_ISSUER_URL", "https://example.com:10000/mcp")
    monkeypatch.setattr(cfg_mod, "_settings", None)
    with TestClient(create_app()) as client:
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


def test_authorize_endpoint_reachable_at_root(oauth_client):
    # Not the SPA — a real authorize handler (400 on incomplete params, not HTML).
    r = oauth_client.get("/authorize", params={
        "response_type": "code", "client_id": "x",
        "redirect_uri": "https://claude.ai/cb",
        "code_challenge": "a", "code_challenge_method": "S256",
    })
    assert r.status_code != 404
    assert "text/html" not in r.headers.get("content-type", "")


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
    # The /mcp endpoint itself still requires a token (the bearer gate).
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
