"""VecgrepOAuthProvider — implements the MCP SDK's OAuthAuthorizationServerProvider.

Wires the 9 SDK interface methods to the TokenStore. The SDK owns the HTTP
routes and PKCE: its /token route verifies the client's code_verifier against
the stored code_challenge BEFORE calling exchange_authorization_code, so this
provider only stores the challenge and mints tokens for a code the SDK already
validated. Dynamic client registration is supported (claude.ai self-registers).
"""
from __future__ import annotations

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from .store import TokenStore, _ACCESS_TTL_S


class VecgrepOAuthProvider(OAuthAuthorizationServerProvider):
    """Embedded auth server behind the HTTP owner-approval middleware.

    Dynamic registration is intentionally open for standards-compliant MCP
    clients, but the web layer must verify the vecgrep owner's approval token
    before this provider is reached. Scope model: read / propose. confirm is
    never a scope — it stays a human action off-protocol.
    """

    def __init__(self, valid_scopes: list[str] | None = None) -> None:
        self.store = TokenStore()
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self.valid_scopes = valid_scopes or ["read", "propose"]

    # ----- client registration -----
    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self._clients[client_info.client_id] = client_info

    # ----- authorize: issue a code, redirect back to the client -----
    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        scopes = [s for s in (params.scopes or self.valid_scopes)
                  if s in self.valid_scopes] or ["read"]
        code = self.store.issue_authorization_code(
            client_id=client.client_id,
            scopes=scopes,
            code_challenge=params.code_challenge,
            redirect_uri=str(params.redirect_uri),
            resource=params.resource,
        )
        # Redirect the resource owner's browser back to the client with the
        # code (+ state for CSRF). The SDK helper builds the query correctly.
        return construct_redirect_uri(
            str(params.redirect_uri), code=code.code, state=params.state
        )

    # ----- authorization code -----
    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        return self.store.load_authorization_code(client.client_id, authorization_code)

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        # Single-use: consume so the code can't be replayed for a second token.
        # (The SDK already verified PKCE before we got here.)
        ac = self.store.consume_authorization_code(client.client_id, authorization_code.code)
        if ac is None:
            from mcp.server.auth.errors import TokenError
            raise TokenError("invalid_grant", "authorization code invalid or already used")
        at = self.store.issue_access_token(client.client_id, ac.scopes, resource=ac.resource)
        rt = self.store.issue_refresh_token(client.client_id, ac.scopes)
        return _oauth_token(at, rt)

    # ----- refresh token -----
    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        return self.store.load_refresh_token(client.client_id, refresh_token)

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        # Narrow to requested scopes (can't widen beyond the refresh token's).
        granted = [s for s in (scopes or refresh_token.scopes) if s in refresh_token.scopes]
        at = self.store.issue_access_token(client.client_id, granted or refresh_token.scopes)
        # Rotate-and-revoke so replaying an already-used refresh token cannot
        # mint a second access token.
        self.store.revoke(refresh_token.token)
        rt = self.store.issue_refresh_token(client.client_id, refresh_token.scopes)
        return _oauth_token(at, rt)

    # ----- access token (also the verify path the bearer middleware uses) -----
    async def load_access_token(self, token: str) -> AccessToken | None:
        return self.store.load_access_token(token)

    # ----- revocation -----
    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        self.store.revoke(token.token)


def _oauth_token(at: AccessToken, rt: RefreshToken) -> OAuthToken:
    return OAuthToken(
        access_token=at.token,
        token_type="Bearer",
        expires_in=_ACCESS_TTL_S,
        scope=" ".join(at.scopes),
        refresh_token=rt.token,
    )
