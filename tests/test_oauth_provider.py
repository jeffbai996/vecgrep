"""OAuth provider — implements the MCP SDK's OAuthAuthorizationServerProvider,
wiring the 9 interface methods to the TokenStore. The SDK's /token route does
PKCE verification before calling exchange_authorization_code, so the provider
just has to store the challenge and mint tokens for a valid code.

These tests drive the provider through a full auth-code flow + a refresh, and
pin the security behaviors: a consumed code can't be exchanged twice, refresh
rotates, revoke kills the token.
"""
from __future__ import annotations

import pytest
from mcp.shared.auth import OAuthClientInformationFull
from mcp.server.auth.provider import AuthorizeError, AuthorizationParams

from vecgrep.backend.auth.provider import VecgrepOAuthProvider


def _client(client_id="claude-ai", redirect="https://claude.ai/api/mcp/auth_callback"):
    return OAuthClientInformationFull(
        client_id=client_id,
        redirect_uris=[redirect],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
        scope="read propose",
    )


@pytest.fixture
def provider():
    return VecgrepOAuthProvider(valid_scopes=["read", "propose"])


@pytest.mark.anyio
async def test_register_and_get_client(provider):
    c = _client()
    await provider.register_client(c)
    got = await provider.get_client("claude-ai")
    assert got is not None and got.client_id == "claude-ai"


@pytest.mark.anyio
async def test_get_unknown_client_is_none(provider):
    assert await provider.get_client("nope") is None


@pytest.mark.anyio
async def test_authorize_returns_redirect_with_code(provider):
    c = _client(); await provider.register_client(c)
    params = AuthorizationParams(
        state="xyz", scopes=["read"], code_challenge="chal",
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        redirect_uri_provided_explicitly=True, resource=None,
    )
    url = await provider.authorize(c, params)
    assert url.startswith("https://claude.ai/api/mcp/auth_callback")
    assert "code=" in url
    assert "state=xyz" in url


@pytest.mark.anyio
async def test_full_auth_code_exchange(provider):
    c = _client(); await provider.register_client(c)
    params = AuthorizationParams(
        state="s", scopes=["read", "propose"], code_challenge="chal",
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        redirect_uri_provided_explicitly=True, resource=None,
    )
    await provider.authorize(c, params)
    # the SDK would parse the code from the redirect; grab it from the store
    code_obj = next(iter(provider.store._codes.values()))
    loaded = await provider.load_authorization_code(c, code_obj.code)
    assert loaded is not None

    tok = await provider.exchange_authorization_code(c, loaded)
    assert tok.access_token and tok.refresh_token
    assert tok.token_type.lower() == "bearer"
    # the access token verifies + carries the scopes
    at = await provider.load_access_token(tok.access_token)
    assert at is not None and set(at.scopes) == {"read", "propose"}


@pytest.mark.anyio
async def test_omitted_scope_defaults_to_read_through_code_and_tokens(provider):
    """A client may register both scopes but omit scope on /authorize. The
    consent page calls that read-only, so every issued artifact must agree."""
    c = _client(); await provider.register_client(c)
    params = AuthorizationParams(
        state="s", scopes=None, code_challenge="chal",
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        redirect_uri_provided_explicitly=True, resource=None,
    )
    await provider.authorize(c, params)
    code_obj = next(iter(provider.store._codes.values()))
    assert code_obj.scopes == ["read"]

    tok = await provider.exchange_authorization_code(c, code_obj)
    assert tok.scope == "read"
    assert (await provider.load_access_token(tok.access_token)).scopes == ["read"]
    assert (await provider.load_refresh_token(c, tok.refresh_token)).scopes == ["read"]


@pytest.mark.anyio
async def test_authorize_rejects_unsupported_scope_instead_of_downgrading(provider):
    c = _client(); await provider.register_client(c)
    params = AuthorizationParams(
        state="s", scopes=["admin"], code_challenge="chal",
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        redirect_uri_provided_explicitly=True, resource=None,
    )
    with pytest.raises(AuthorizeError) as exc:
        await provider.authorize(c, params)
    assert exc.value.error == "invalid_scope"


@pytest.mark.anyio
async def test_code_cannot_be_exchanged_twice(provider):
    c = _client(); await provider.register_client(c)
    params = AuthorizationParams(
        state="s", scopes=["read"], code_challenge="chal",
        redirect_uri="https://claude.ai/api/mcp/auth_callback",
        redirect_uri_provided_explicitly=True, resource=None,
    )
    await provider.authorize(c, params)
    code_obj = next(iter(provider.store._codes.values()))
    loaded = await provider.load_authorization_code(c, code_obj.code)
    await provider.exchange_authorization_code(c, loaded)
    # second load returns None (consumed) — a replayed code is dead
    assert await provider.load_authorization_code(c, code_obj.code) is None


@pytest.mark.anyio
async def test_refresh_token_rotation(provider):
    c = _client(); await provider.register_client(c)
    rt = provider.store.issue_refresh_token("claude-ai", ["read"])
    loaded = await provider.load_refresh_token(c, rt.token)
    assert loaded is not None
    tok = await provider.exchange_refresh_token(c, loaded, ["read"])
    assert tok.access_token
    at = await provider.load_access_token(tok.access_token)
    assert at is not None and at.scopes == ["read"]
    assert await provider.load_refresh_token(c, rt.token) is None


@pytest.mark.anyio
async def test_revoke_kills_access_token(provider):
    c = _client(); await provider.register_client(c)
    at = provider.store.issue_access_token("claude-ai", ["read"], ttl_s=3600)
    assert await provider.load_access_token(at.token) is not None
    await provider.revoke_token(at)
    assert await provider.load_access_token(at.token) is None


def test_read_scope_cannot_invoke_mutation_tools(provider):
    from mcp.server.auth.middleware.auth_context import auth_context_var
    from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
    from vecgrep.mcp.server import _require_remote_scope

    at = provider.store.issue_access_token("claude-ai", ["read"])
    marker = auth_context_var.set(AuthenticatedUser(at))
    try:
        denial = _require_remote_scope("propose")
    finally:
        auth_context_var.reset(marker)
    assert denial is not None
    assert "requires" in denial
    assert at.token not in denial


def test_propose_scope_allows_mutation_tool_boundary(provider):
    from mcp.server.auth.middleware.auth_context import auth_context_var
    from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
    from vecgrep.mcp.server import _require_remote_scope

    at = provider.store.issue_access_token("claude-ai", ["read", "propose"])
    marker = auth_context_var.set(AuthenticatedUser(at))
    try:
        assert _require_remote_scope("propose") is None
    finally:
        auth_context_var.reset(marker)


def test_stdio_without_oauth_token_preserves_local_tools():
    from vecgrep.mcp.server import _require_remote_scope

    assert _require_remote_scope("propose") is None
