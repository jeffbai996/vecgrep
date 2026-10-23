"""VecgrepOAuthProvider — implements the MCP SDK's OAuthAuthorizationServerProvider.

Wires the 9 SDK interface methods to the TokenStore. The SDK owns the HTTP
routes and PKCE: its /token route verifies the client's code_verifier against
the stored code_challenge BEFORE calling exchange_authorization_code, so this
provider only stores the challenge and mints tokens for a code the SDK already
validated. Dynamic client registration is supported (claude.ai self-registers).
"""
from __future__ import annotations

from urllib.parse import urlsplit

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizeError,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from .scopes import VALID_SCOPES, UnsupportedScope, effective_scopes
from .store import TokenStore, _ACCESS_TTL_S, _is_google_connector


def _same_origin_resource(requested: str | None, configured: str | None) -> bool:
    if not requested or not configured:
        return False
    req = urlsplit(str(requested))
    cfg = urlsplit(str(configured))
    return bool(
        req.scheme
        and req.netloc
        and (req.scheme.lower(), req.netloc.lower()) == (cfg.scheme.lower(), cfg.netloc.lower())
    )


class VecgrepOAuthProvider(OAuthAuthorizationServerProvider):
    """Embedded auth server behind the HTTP owner-approval middleware.

    Dynamic registration is intentionally open for standards-compliant MCP
    clients, but the web layer must verify the vecgrep owner's approval token
    before this provider is reached. Scope model: read / propose. confirm is
    never a scope — it stays a human action off-protocol.
    """

    def __init__(self, valid_scopes: list[str] | None = None,
                 *, resource: str | None = None) -> None:
        import os
        # persisted when VECGREP_OAUTH_STATE_FILE is set (the service env does)
        self.store = TokenStore(
            path=os.environ.get("VECGREP_OAUTH_STATE_FILE", "").strip() or None,
            default_resource=resource,
        )
        self.resource = resource
        self.valid_scopes = list(VALID_SCOPES if valid_scopes is None else valid_scopes)

    @property
    def _clients(self) -> dict[str, OAuthClientInformationFull]:
        return self.store.clients()

    # ----- client registration -----
    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self.store.get_client(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        self.store.save_client(client_info)

    # ----- authorize: issue a code, redirect back to the client -----
    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        allow_sibling = _is_google_connector(client) and _same_origin_resource(params.resource, self.resource)
        if self.resource and params.resource not in (None, self.resource) and not allow_sibling:
            raise AuthorizeError("invalid_target", "wrong OAuth resource")
        raw_scopes = params.scopes
        if _is_google_connector(client) and raw_scopes:
            mapped = [
                "read" if s.endswith(".read") else ("propose" if s.endswith(".write") else s)
                for s in raw_scopes
                if s != "offline_access"
            ]
            raw_scopes = list(dict.fromkeys(mapped)) or ["read"]
        try:
            scopes = effective_scopes(
                raw_scopes,
                valid_scopes=self.valid_scopes,
            )
        except UnsupportedScope as exc:
            raise AuthorizeError("invalid_scope", str(exc)) from exc
        target_resource = self.resource if allow_sibling else (params.resource or self.resource)
        code = self.store.issue_authorization_code(
            client_id=client.client_id,
            scopes=scopes,
            code_challenge=params.code_challenge,
            redirect_uri=str(params.redirect_uri),
            resource=target_resource,
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
        # The approval starts a token family; every rotation stays in it.
        at, rt = self.store.issue_token_pair(client.client_id, ac.scopes, resource=ac.resource)
        return _oauth_token(at, rt)

    # ----- refresh token -----
    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        # A refresh token this client already rotated revokes its family here,
        # before the SDK answers invalid_grant.
        return self.store.load_refresh_token(client.client_id, refresh_token)

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        resource = getattr(refresh_token, "resource", None)
        if self.resource and resource != self.resource:
            from mcp.server.auth.provider import TokenError

            raise TokenError("invalid_grant", "refresh grant has no matching resource")
        # Reject before minting; filtering an invalid request must never renew
        # the original authority. Successor refresh grants stay narrowed too.
        granted = list(scopes or refresh_token.scopes)
        if (
            not granted
            or not set(granted).issubset(refresh_token.scopes)
            or not set(granted).issubset(self.valid_scopes)
        ):
            from mcp.server.auth.provider import TokenError

            raise TokenError("invalid_scope", "unsupported requested scope")
        # Consume and reissue in one store call. A token already rotated means
        # a second copy exists: the store revokes its whole family and nothing
        # is minted.
        pair = self.store.rotate_refresh_token(client.client_id, refresh_token.token, granted)
        if pair is None:
            from mcp.server.auth.provider import TokenError

            raise TokenError("invalid_grant", "refresh token invalid or already used")
        at, rt = pair
        return _oauth_token(at, rt)

    # ----- access token (also the verify path the bearer middleware uses) -----
    async def load_access_token(self, token: str) -> AccessToken | None:
        return self.store.load_access_token(token)

    # ----- revocation -----
    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        self.store.revoke(token.token)


def _oauth_token(at: AccessToken, rt: RefreshToken) -> OAuthToken:
    from .store import _now
    expires_in = max(1, round(at.expires_at - _now())) if at.expires_at is not None else _ACCESS_TTL_S
    return OAuthToken(
        access_token=at.token,
        token_type="Bearer",
        expires_in=expires_in,
        scope=" ".join(at.scopes),
        refresh_token=rt.token,
    )
