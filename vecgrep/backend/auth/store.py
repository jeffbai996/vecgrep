"""Token store — the OAuth lifecycle vecgrep owns.

Issues, loads, expires, and revokes access tokens, refresh tokens, and
authorization codes, using the MCP SDK's data types so the provider can hand
them straight back to the SDK. In-memory by default (fine for a single-user
homelab where a restart re-auths); a persisted backend can swap in later
without touching the provider.

Security invariants enforced here:
  - expiry is checked on every load (an expired token never loads)
  - revoked tokens stop loading immediately
  - authorization codes are single-use (consume removes them) — a replayed
    code can't mint a second token
  - refresh tokens are bound to their client_id
"""
from __future__ import annotations

import secrets
import time

from mcp.server.auth.provider import AccessToken, AuthorizationCode, RefreshToken

# Default lifetimes. Access tokens are short (the whole point of OAuth); refresh
# tokens are long-lived but revocable; codes are seconds-scale (used immediately).
_ACCESS_TTL_S = 3600        # 1 hour
_REFRESH_TTL_S = 30 * 86400  # 30 days
_CODE_TTL_S = 300           # 5 minutes


def _now() -> float:
    return time.time()


def _new_secret(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(32)}"


class TokenStore:
    def __init__(self) -> None:
        self._access: dict[str, AccessToken] = {}
        self._refresh: dict[str, RefreshToken] = {}
        self._codes: dict[str, AuthorizationCode] = {}

    # ----- access tokens -----
    def issue_access_token(self, client_id: str, scopes: list[str],
                           ttl_s: int = _ACCESS_TTL_S,
                           resource: str | None = None) -> AccessToken:
        at = AccessToken(
            token=_new_secret("vgat"),
            client_id=client_id,
            scopes=list(scopes),
            expires_at=int(_now() + ttl_s),
            resource=resource,
        )
        self._access[at.token] = at
        return at

    def load_access_token(self, token: str) -> AccessToken | None:
        at = self._access.get(token)
        if at is None:
            return None
        if at.expires_at is not None and at.expires_at <= _now():
            return None
        return at

    def token_has_scope(self, token: str, scope: str) -> bool:
        at = self.load_access_token(token)
        return bool(at and scope in at.scopes)

    # ----- refresh tokens -----
    def issue_refresh_token(self, client_id: str, scopes: list[str],
                            ttl_s: int = _REFRESH_TTL_S) -> RefreshToken:
        rt = RefreshToken(
            token=_new_secret("vgrt"),
            client_id=client_id,
            scopes=list(scopes),
            expires_at=int(_now() + ttl_s),
        )
        self._refresh[rt.token] = rt
        return rt

    def load_refresh_token(self, client_id: str, token: str) -> RefreshToken | None:
        rt = self._refresh.get(token)
        if rt is None or rt.client_id != client_id:
            return None
        if rt.expires_at is not None and rt.expires_at <= _now():
            return None
        return rt

    # ----- authorization codes (single-use) -----
    def issue_authorization_code(self, client_id: str, scopes: list[str],
                                 code_challenge: str, redirect_uri: str,
                                 ttl_s: int = _CODE_TTL_S,
                                 resource: str | None = None) -> AuthorizationCode:
        code = AuthorizationCode(
            code=_new_secret("vgac"),
            scopes=list(scopes),
            expires_at=_now() + ttl_s,
            client_id=client_id,
            code_challenge=code_challenge,
            redirect_uri=redirect_uri,
            redirect_uri_provided_explicitly=True,
            resource=resource,
        )
        self._codes[code.code] = code
        return code

    def load_authorization_code(self, client_id: str, code: str) -> AuthorizationCode | None:
        ac = self._codes.get(code)
        if ac is None or ac.client_id != client_id:
            return None
        if ac.expires_at is not None and ac.expires_at <= _now():
            return None
        return ac

    def consume_authorization_code(self, client_id: str, code: str) -> AuthorizationCode | None:
        """Load-and-remove: a code can mint exactly one token, then it's gone."""
        ac = self.load_authorization_code(client_id, code)
        if ac is not None:
            self._codes.pop(code, None)
        return ac

    # ----- revocation -----
    def revoke(self, token: str) -> None:
        """Revoke an access OR refresh token by its value. Idempotent."""
        self._access.pop(token, None)
        self._refresh.pop(token, None)
