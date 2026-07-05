"""OAuth token store — the lifecycle WE own (the SDK does HTTP/PKCE/crypto).

Mint, load, expire, and revoke access tokens, refresh tokens, and authorization
codes, using the MCP SDK's data types so the provider can hand them straight to
the SDK. These tests pin the lifecycle invariants that matter for security:
expiry is enforced, revoked tokens stop loading, a code is single-use, and
scopes round-trip.
"""
from __future__ import annotations

import time

import pytest

from vecgrep.backend.auth.store import TokenStore


@pytest.fixture
def store():
    return TokenStore()


# --- access tokens ---

def test_issue_and_load_access_token(store):
    at = store.issue_access_token(client_id="c1", scopes=["read"], ttl_s=3600)
    loaded = store.load_access_token(at.token)
    assert loaded is not None
    assert loaded.client_id == "c1"
    assert loaded.scopes == ["read"]


def test_expired_access_token_does_not_load(store):
    at = store.issue_access_token(client_id="c1", scopes=["read"], ttl_s=-1)  # already expired
    assert store.load_access_token(at.token) is None


def test_revoked_access_token_does_not_load(store):
    at = store.issue_access_token(client_id="c1", scopes=["read"], ttl_s=3600)
    store.revoke(at.token)
    assert store.load_access_token(at.token) is None


def test_unknown_access_token_is_none(store):
    assert store.load_access_token("never-issued") is None


# --- refresh tokens ---

def test_issue_and_load_refresh_token(store):
    rt = store.issue_refresh_token(client_id="c1", scopes=["read", "propose"])
    loaded = store.load_refresh_token("c1", rt.token)
    assert loaded is not None
    assert set(loaded.scopes) == {"read", "propose"}


def test_refresh_token_wrong_client_rejected(store):
    rt = store.issue_refresh_token(client_id="c1", scopes=["read"])
    assert store.load_refresh_token("other-client", rt.token) is None


def test_revoked_refresh_token_does_not_load(store):
    rt = store.issue_refresh_token(client_id="c1", scopes=["read"])
    store.revoke(rt.token)
    assert store.load_refresh_token("c1", rt.token) is None


# --- authorization codes (single-use) ---

def test_issue_and_consume_authorization_code(store):
    code = store.issue_authorization_code(
        client_id="c1", scopes=["read"], code_challenge="abc",
        redirect_uri="https://claude.ai/cb",
    )
    loaded = store.load_authorization_code("c1", code.code)
    assert loaded is not None
    assert loaded.code_challenge == "abc"


def test_authorization_code_is_single_use(store):
    code = store.issue_authorization_code(
        client_id="c1", scopes=["read"], code_challenge="abc",
        redirect_uri="https://claude.ai/cb",
    )
    assert store.consume_authorization_code("c1", code.code) is not None
    # second consume fails — a replayed code can't mint a second token
    assert store.consume_authorization_code("c1", code.code) is None


def test_expired_authorization_code_does_not_load(store):
    code = store.issue_authorization_code(
        client_id="c1", scopes=["read"], code_challenge="abc",
        redirect_uri="https://claude.ai/cb", ttl_s=-1,
    )
    assert store.load_authorization_code("c1", code.code) is None


# --- scope enforcement helper ---

def test_token_has_scope(store):
    at = store.issue_access_token(client_id="c1", scopes=["read"], ttl_s=3600)
    assert store.token_has_scope(at.token, "read") is True
    assert store.token_has_scope(at.token, "propose") is False


# ----- admin surface (inventory panel) -----

def test_counts_reflect_live_tokens(store):
    store.issue_access_token("c1", ["read"])
    store.issue_access_token("c1", ["read"])
    store.issue_refresh_token("c1", ["read"])
    store.issue_access_token("c2", ["read"])
    c = store.counts()
    assert c["access"] == 3
    assert c["refresh"] == 1
    assert c["by_client"]["c1"]["access"] == 2


def test_revoke_client_kills_all_its_tokens(store):
    a1 = store.issue_access_token("c1", ["read"])
    r1 = store.issue_refresh_token("c1", ["read"])
    a2 = store.issue_access_token("c2", ["read"])
    assert store.revoke_client("c1") == 2
    assert store.load_access_token(a1.token) is None
    assert store.load_refresh_token("c1", r1.token) is None
    assert store.load_access_token(a2.token) is not None
