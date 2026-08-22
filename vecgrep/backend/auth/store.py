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

import json
import os

from mcp.server.auth.provider import AccessToken, AuthorizationCode, RefreshToken
from mcp.shared.auth import OAuthClientInformationFull

# Default lifetimes. Access tokens are short (the whole point of OAuth); refresh
# tokens are long-lived but revocable; codes are seconds-scale (used immediately).
# A year: the owner approves a client once and expects it to stay connected
# (Jeff 2026-08-22: "indefinite or like a year"); revocation is the lever.
_ACCESS_TTL_S = 365 * 86400
_REFRESH_TTL_S = 365 * 86400
_CODE_TTL_S = 300           # 5 minutes


def _now() -> float:
    return time.time()


def _new_secret(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(32)}"


class TokenStore:
    """Tokens and registered clients, optionally persisted to a 0600 JSON
    file so a service restart no longer revokes every connected client
    (2026-08-22). Codes are never persisted — they live seconds."""

    def __init__(self, path: str | None = None) -> None:
        self._access: dict[str, AccessToken] = {}
        self._refresh: dict[str, RefreshToken] = {}
        self._codes: dict[str, AuthorizationCode] = {}
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._path = path
        self._load()

    # ----- persistence -----
    def _load(self) -> None:
        if not self._path or not os.path.exists(self._path):
            return
        try:
            with open(self._path, encoding="utf-8") as fh:
                raw = json.load(fh)
            self._access = {k: AccessToken.model_validate(v) for k, v in raw.get("access", {}).items()}
            self._refresh = {k: RefreshToken.model_validate(v) for k, v in raw.get("refresh", {}).items()}
            self._clients = {k: OAuthClientInformationFull.model_validate(v) for k, v in raw.get("clients", {}).items()}
        except Exception:
            # a corrupt state file must not take the auth server down; it
            # simply means every client re-approves once
            self._access, self._refresh, self._clients = {}, {}, {}

    def _save(self) -> None:
        if not self._path:
            return
        now = _now()
        data = {
            "access": {k: v.model_dump(mode="json") for k, v in self._access.items()
                       if v.expires_at is None or v.expires_at > now},
            "refresh": {k: v.model_dump(mode="json") for k, v in self._refresh.items()
                        if v.expires_at is None or v.expires_at > now},
            "clients": {k: v.model_dump(mode="json") for k, v in self._clients.items()},
        }
        os.makedirs(os.path.dirname(self._path) or ".", mode=0o700, exist_ok=True)
        tmp = self._path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, self._path)

    # ----- clients -----
    def save_client(self, info: OAuthClientInformationFull) -> None:
        self._clients[info.client_id] = info
        self._save()

    def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._clients.get(client_id)

    def clients(self) -> dict[str, OAuthClientInformationFull]:
        return self._clients

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
        self._save()
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
        self._save()
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
        self._save()

    # ----- admin surface (the /inventory OAuth panel reads these) -----
    def counts(self) -> dict:
        """Live token tallies, total and per-client. Expired tokens still in
        the dicts count — they can't LOAD, and showing them beats hiding a
        pileup from the ops view."""
        by_client: dict[str, dict[str, int]] = {}
        for at in self._access.values():
            by_client.setdefault(at.client_id, {"access": 0, "refresh": 0})
            by_client[at.client_id]["access"] += 1
        for rt in self._refresh.values():
            by_client.setdefault(rt.client_id, {"access": 0, "refresh": 0})
            by_client[rt.client_id]["refresh"] += 1
        return {
            "access": len(self._access),
            "refresh": len(self._refresh),
            "codes": len(self._codes),
            "by_client": by_client,
        }

    def revoke_client(self, client_id: str) -> int:
        """Kill every access + refresh token a client holds (the inventory
        panel's revoke button). Returns how many tokens died."""
        doomed_a = [t for t, at in self._access.items() if at.client_id == client_id]
        doomed_r = [t for t, rt in self._refresh.items() if rt.client_id == client_id]
        for t in doomed_a:
            self._access.pop(t, None)
        for t in doomed_r:
            self._refresh.pop(t, None)
        self._codes = {c: ac for c, ac in self._codes.items()
                       if ac.client_id != client_id}
        self._save()
        return len(doomed_a) + len(doomed_r)
