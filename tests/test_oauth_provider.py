"""OAuth provider — implements the MCP SDK's OAuthAuthorizationServerProvider,
wiring the 9 interface methods to the TokenStore. The SDK's /token route does
PKCE verification before calling exchange_authorization_code, so the provider
just has to store the challenge and mint tokens for a valid code.

These tests drive the provider through a full auth-code flow + a refresh, and
pin the security behaviors: a consumed code can't be exchanged twice, refresh
rotates, revoke kills the token.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp.server.auth.provider import AuthorizationParams, AuthorizeError
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl

from vecgrep.backend.auth.provider import VecgrepOAuthProvider

CLAUDE_REDIRECT = AnyUrl("https://claude.ai/api/mcp/auth_callback")
GOOGLE_REDIRECT = AnyUrl("https://oauth-redirect.googleusercontent.com/r/custom-mcp")
USER_REDIRECT = AnyUrl("https://oauth-redirect.googleusercontent.com/r/user_bound_custom-mcp-host_example")


def _client(client_id="claude-ai", redirect: str | AnyUrl = CLAUDE_REDIRECT):
    return OAuthClientInformationFull(
        client_id=client_id,
        redirect_uris=[AnyUrl(str(redirect))],
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
        redirect_uri=CLAUDE_REDIRECT,
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
        redirect_uri=CLAUDE_REDIRECT,
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
        redirect_uri=CLAUDE_REDIRECT,
        redirect_uri_provided_explicitly=True, resource=None,
    )
    await provider.authorize(c, params)
    code_obj = next(iter(provider.store._codes.values()))
    assert code_obj.scopes == ["read"]

    tok = await provider.exchange_authorization_code(c, code_obj)
    assert tok.scope == "read"
    assert tok.refresh_token is not None
    at = await provider.load_access_token(tok.access_token)
    assert at is not None and at.scopes == ["read"]
    rt = await provider.load_refresh_token(c, tok.refresh_token)
    assert rt is not None and rt.scopes == ["read"]


@pytest.mark.anyio
async def test_authorize_rejects_unsupported_scope_instead_of_downgrading(provider):
    c = _client(); await provider.register_client(c)
    params = AuthorizationParams(
        state="s", scopes=["admin"], code_challenge="chal",
        redirect_uri=CLAUDE_REDIRECT,
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
        redirect_uri=CLAUDE_REDIRECT,
        redirect_uri_provided_explicitly=True, resource=None,
    )
    await provider.authorize(c, params)
    code_obj = next(iter(provider.store._codes.values()))
    loaded = await provider.load_authorization_code(c, code_obj.code)
    assert loaded is not None
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


@pytest.mark.anyio
@pytest.mark.parametrize(
    "case",
    json.loads(
        (Path(__file__).parent / "fixtures/contracts/oauth-scopes-v1.json").read_text()
    )["refresh_scope_cases"],
    ids=lambda case: case["id"],
)
async def test_shared_refresh_scope_contract(provider, case):
    from mcp.server.auth.provider import TokenError

    client = _client()
    await provider.register_client(client)
    original = provider.store.issue_refresh_token(client.client_id, case["original"])
    if case["error"]:
        with pytest.raises(TokenError) as error:
            await provider.exchange_refresh_token(client, original, case["requested"])
        assert error.value.error == case["error"]
        assert await provider.load_refresh_token(client, original.token) is not None
        return
    result = await provider.exchange_refresh_token(client, original, case["requested"])
    access = await provider.load_access_token(result.access_token)
    successor = await provider.load_refresh_token(client, result.refresh_token)
    assert access.scopes == successor.scopes == case["granted"]


@pytest.mark.anyio
@pytest.mark.parametrize("requested_resource", [None, "https://resource.example/mcp"])
async def test_resource_binding_survives_exchange_refresh_and_reload(tmp_path, monkeypatch, requested_resource):
    from vecgrep.backend.auth.store import TokenStore

    resource = "https://resource.example/mcp"
    path = str(tmp_path / "state.json")
    monkeypatch.setenv("VECGREP_OAUTH_STATE_FILE", path)
    provider = VecgrepOAuthProvider(resource=resource)
    client = _client()
    await provider.register_client(client)
    assert client.redirect_uris is not None
    await provider.authorize(client, AuthorizationParams(
        state="s", scopes=["read"], code_challenge="challenge",
        redirect_uri=client.redirect_uris[0],
        redirect_uri_provided_explicitly=True, resource=requested_resource,
    ))
    code = next(iter(provider.store._codes.values()))
    assert code.resource == resource
    pair = await provider.exchange_authorization_code(client, code)
    assert pair.refresh_token is not None
    provider.store = TokenStore(path=path)
    refresh = await provider.load_refresh_token(client, pair.refresh_token)
    assert refresh is not None
    assert refresh.resource == resource
    successor = await provider.exchange_refresh_token(client, refresh, ["read"])
    assert successor.refresh_token is not None
    access_tok = await provider.load_access_token(successor.access_token)
    assert access_tok is not None
    assert access_tok.resource == resource
    ref_tok = await provider.load_refresh_token(client, successor.refresh_token)
    assert ref_tok is not None
    assert ref_tok.resource == resource
    assert await provider.load_refresh_token(client, pair.refresh_token) is None


@pytest.mark.anyio
async def test_configured_provider_rejects_authorization_for_another_resource():
    provider = VecgrepOAuthProvider(resource="https://resource.example/mcp")
    with pytest.raises(AuthorizeError) as error:
        await provider.authorize(_client(), AuthorizationParams(
            state="s", scopes=["read"], code_challenge="challenge",
            redirect_uri=CLAUDE_REDIRECT,
            redirect_uri_provided_explicitly=True, resource="https://other.example/mcp",
        ))
    assert error.value.error == "invalid_target"
    assert not provider.store._codes


@pytest.mark.anyio
@pytest.mark.parametrize("resource", [None, "https://other.example/mcp"])
async def test_refresh_cannot_upgrade_an_unbound_or_wrong_resource_grant(resource):
    from mcp.server.auth.provider import TokenError

    provider = VecgrepOAuthProvider(resource="https://resource.example/mcp")
    client = _client()
    assert client.client_id is not None
    refresh = provider.store.issue_refresh_token(client.client_id, ["read"], resource=resource)
    with pytest.raises(TokenError) as error:
        await provider.exchange_refresh_token(client, refresh, ["read"])
    assert error.value.error == "invalid_grant"
    assert not provider.store._access
    assert await provider.load_refresh_token(client, refresh.token) is not None


async def _code_pair(provider, client):
    params = AuthorizationParams(
        state="s", scopes=["read"], code_challenge="chal",
        redirect_uri=CLAUDE_REDIRECT,
        redirect_uri_provided_explicitly=True, resource=None,
    )
    await provider.authorize(client, params)
    code_obj = next(iter(provider.store._codes.values()))
    return await provider.exchange_authorization_code(client, code_obj)


@pytest.mark.anyio
async def test_refresh_replay_revokes_the_family_and_is_invalid_grant(provider):
    from mcp.server.auth.provider import TokenError

    c = _client(); await provider.register_client(c)
    first = await _code_pair(provider, c)
    assert first.refresh_token is not None
    old = await provider.load_refresh_token(c, first.refresh_token)
    assert old is not None
    second = await provider.exchange_refresh_token(c, old, ["read"])
    assert second.refresh_token is not None
    assert await provider.load_access_token(second.access_token) is not None

    with pytest.raises(TokenError) as exc:
        await provider.exchange_refresh_token(c, old, ["read"])
    assert exc.value.error == "invalid_grant"
    assert await provider.load_access_token(first.access_token) is None
    assert await provider.load_access_token(second.access_token) is None
    assert await provider.load_refresh_token(c, second.refresh_token) is None


@pytest.mark.anyio
async def test_refresh_replay_through_load_revokes_the_family(provider):
    c = _client(); await provider.register_client(c)
    first = await _code_pair(provider, c)
    assert first.refresh_token is not None
    old = await provider.load_refresh_token(c, first.refresh_token)
    assert old is not None
    second = await provider.exchange_refresh_token(c, old, ["read"])
    assert second.refresh_token is not None
    assert await provider.load_refresh_token(c, first.refresh_token) is None
    assert await provider.load_access_token(second.access_token) is None
    assert await provider.load_refresh_token(c, second.refresh_token) is None


@pytest.mark.anyio
async def test_revoking_a_refresh_token_revokes_its_access_tokens(provider):
    c = _client(); await provider.register_client(c)
    pair = await _code_pair(provider, c)
    assert pair.refresh_token is not None
    refresh = await provider.load_refresh_token(c, pair.refresh_token)
    assert refresh is not None
    await provider.revoke_token(refresh)
    assert await provider.load_access_token(pair.access_token) is None
    assert await provider.load_refresh_token(c, pair.refresh_token) is None


@pytest.mark.anyio
async def test_google_connector_oauth_token_advertises_long_expires_in(provider):
    google = OAuthClientInformationFull(
        client_id="google-ai",
        client_name="Google",
        redirect_uris=[GOOGLE_REDIRECT],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope="read propose",
    )
    await provider.register_client(google)
    await provider.authorize(google, AuthorizationParams(
        state="s", scopes=["read"], code_challenge="chal",
        redirect_uri=GOOGLE_REDIRECT,
        redirect_uri_provided_explicitly=True, resource=None,
    ))
    code_obj = next(iter(provider.store._codes.values()))
    pair = await provider.exchange_authorization_code(google, code_obj)
    assert pair.expires_in is not None and pair.expires_in >= 365 * 86400 - 5
    assert pair.refresh_token is not None
    refresh = await provider.load_refresh_token(google, pair.refresh_token)
    assert refresh is not None
    rotated = await provider.exchange_refresh_token(google, refresh, ["read"])
    assert rotated.expires_in is not None and rotated.expires_in >= 365 * 86400 - 5


@pytest.mark.anyio
async def test_google_connector_accepts_same_origin_sibling_resource_and_normalizes_token_audience():
    provider = VecgrepOAuthProvider(resource="https://host.example/vg-secret/vecgrep-mcp")
    google = OAuthClientInformationFull(
        client_id="google-ai",
        client_name="Google",
        redirect_uris=[USER_REDIRECT],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope="read propose",
    )
    await provider.register_client(google)
    await provider.authorize(google, AuthorizationParams(
        state="s", scopes=["read", "propose"], code_challenge="chal",
        redirect_uri=USER_REDIRECT,
        redirect_uri_provided_explicitly=True,
        resource="https://host.example/notes-mcp/mcp",
    ))
    code_obj = next(iter(provider.store._codes.values()))
    assert code_obj.resource == "https://host.example/vg-secret/vecgrep-mcp"
    pair = await provider.exchange_authorization_code(google, code_obj)
    access = await provider.load_access_token(pair.access_token)
    assert access is not None
    assert access.resource == "https://host.example/vg-secret/vecgrep-mcp"

    with pytest.raises(AuthorizeError) as error:
        await provider.authorize(google, AuthorizationParams(
            state="s", scopes=["read"], code_challenge="chal",
            redirect_uri=USER_REDIRECT,
            redirect_uri_provided_explicitly=True,
            resource="https://other.example/notes-mcp/mcp",
        ))
    assert error.value.error == "invalid_target"


@pytest.mark.anyio
async def test_exchange_consumed_authorization_code_raises_invalid_grant_token_error(provider):
    from mcp.server.auth.provider import TokenError

    c = _client()
    await provider.register_client(c)
    await provider.authorize(c, AuthorizationParams(
        state="s", scopes=["read"], code_challenge="chal",
        redirect_uri=CLAUDE_REDIRECT,
        redirect_uri_provided_explicitly=True, resource=None,
    ))
    code_obj = next(iter(provider.store._codes.values()))
    await provider.exchange_authorization_code(c, code_obj)

    with pytest.raises(TokenError) as exc:
        await provider.exchange_authorization_code(c, code_obj)
    assert exc.value.error == "invalid_grant"



