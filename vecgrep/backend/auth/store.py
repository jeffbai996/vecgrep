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
  - every token descends from one approval (a family); rotation retires the
    presented refresh token, and presenting a retired one again revokes every
    access and refresh token of its family, because only a copy of a token the
    client already rotated can still hold the retired value
  - anonymous dynamic registrations have a TTL and count cap, while clients
    with issued tokens are never evicted automatically
  - Gemini's bearer is shared with the other MCPs on this origin through the
    registry in google_bearers.py: a bearer another MCP issued is accepted
    only under this store's own approval of Gemini (its client, scopes and
    family, no later expiry), and a revocation anywhere holds here
  - Google is recognised by where its logins are delivered (its OAuth redirect
    hosts), never by the client name a registrant typed
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import tempfile
import threading
import time
from urllib.parse import urlsplit

from mcp.server.auth.provider import AccessToken, AuthorizationCode, RefreshToken
from mcp.shared.auth import OAuthClientInformationFull

from . import google_bearers
from .google_bearers import GoogleBearerRegistry

# Default lifetimes. Access tokens are short (the whole point of OAuth); refresh
# tokens are long-lived but revocable; codes are seconds-scale (used immediately).
# The approval lasts a year through the refresh token: the owner approves a
# client once and expects it to stay connected (Jeff 2026-08-22: "indefinite or
# like a year"), and rotation keeps that alive without a new approval. The
# access token a client actually presents on every call lives one hour, so a
# token that leaks from a client, a log or a trace stops working within the
# hour instead of a year (security review 2026-10-19).
_ACCESS_TTL_S = 60 * 60
_REFRESH_TTL_S = 365 * 86400
_CODE_TTL_S = 300           # 5 minutes
_MAX_UNAPPROVED_CLIENTS = 32
# An approved client can rotate as fast as it likes, and rotation leaves the
# earlier access tokens to expire on their own; the cap keeps that from growing
# the state without bound. Matches ibkr-mcp.
_MAX_ACCESS_PER_CLIENT = 16
_UNAPPROVED_CLIENT_TTL_S = 24 * 3600


def _now() -> float:
    return time.time()


def _new_secret(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(32)}"


class ResourceRefreshToken(RefreshToken):
    """Keep the audience with a refresh grant; the SDK base type omits it.
    family names the approval the token descends from."""

    resource: str | None = None
    family: str | None = None


class FamilyAccessToken(AccessToken):
    """An access token that knows the approval it descends from, so revoking
    the approval revokes it too. Tokens minted outside an approval have none."""

    family: str | None = None


def _legacy_key(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# Gemini registers its connector with redirects on Google's production, sandbox
# and test OAuth hosts. Only Google receives what is sent there, while a client
# name is whatever the registrant typed.
_GOOGLE_REDIRECT_HOSTS = frozenset({
    "oauth-redirect.googleusercontent.com",
    "oauth-redirect-sandbox.googleusercontent.com",
    "oauth-redirect-test.googleusercontent.com",
})


def _is_google_connector(client: OAuthClientInformationFull | None) -> bool:
    """Google's Gemini web MCP proxy caches issued bearer tokens across worker
    tasks and does not refresh expired 1-hour access tokens on chat turns."""
    if client is None:
        return False
    uris = [str(u) for u in (getattr(client, "redirect_uris", None) or ())]
    if not uris:
        return False
    for uri in uris:
        parts = urlsplit(uri)
        if parts.scheme != "https" or (parts.hostname or "") not in _GOOGLE_REDIRECT_HOSTS:
            return False
    return True


class TokenStore:
    """Tokens and registered clients, optionally persisted to a 0600 JSON
    file so a service restart no longer revokes every connected client
    (2026-08-22). Codes are never persisted — they live seconds."""

    def __init__(
        self,
        path: str | None = None,
        *,
        max_unapproved_clients: int = _MAX_UNAPPROVED_CLIENTS,
        unapproved_client_ttl_s: int = _UNAPPROVED_CLIENT_TTL_S,
        google_registry_path: str | None = None,
        default_resource: str | None = None,
    ) -> None:
        if max_unapproved_clients < 1:
            raise ValueError("max_unapproved_clients must be positive")
        if unapproved_client_ttl_s < 1:
            raise ValueError("unapproved_client_ttl_s must be positive")
        self._access: dict[str, FamilyAccessToken] = {}
        self._refresh: dict[str, ResourceRefreshToken] = {}
        # family id -> {"client_id", "created_at"}
        self._families: dict[str, dict] = {}
        # sha256 of a rotated pre-family refresh token -> {"client_id", "family"}.
        # Refresh tokens minted with a family carry it under an HMAC, so a
        # retired one is recognised without keeping a record per rotation.
        self._retired_legacy: dict[str, dict] = {}
        self._signing_key = secrets.token_urlsafe(32)
        self._codes: dict[str, AuthorizationCode] = {}
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._path = path
        self._loaded_mtime: float = 0.0
        self._default_resource = default_resource
        if google_registry_path is None and path:
            google_registry_path = os.environ.get(google_bearers.ENV_VAR) or None
        self._google_bearers = (GoogleBearerRegistry(google_registry_path)
                                if google_registry_path else None)
        self._max_unapproved_clients = max_unapproved_clients
        self._unapproved_client_ttl_s = unapproved_client_ttl_s
        # Serialises saves: two concurrent requests used to write the same
        # .tmp path, and the second os.replace found it gone.
        self._save_lock = threading.RLock()
        self._load()

    # ----- persistence -----
    def _load(self) -> None:
        if not self._path or not os.path.exists(self._path):
            return
        try:
            self._loaded_mtime = os.path.getmtime(self._path)
            with open(self._path, encoding="utf-8") as fh:
                raw = json.load(fh)
            access = {k: FamilyAccessToken.model_validate(v) for k, v in raw.get("access", {}).items()}
            refresh = {k: ResourceRefreshToken.model_validate(v) for k, v in raw.get("refresh", {}).items()}
            clients = {k: OAuthClientInformationFull.model_validate(v) for k, v in raw.get("clients", {}).items()}
            families = {str(k): dict(v) for k, v in raw.get("families", {}).items()}
            retired = {str(k): dict(v) for k, v in raw.get("retired_legacy", {}).items()}
            key = raw.get("refresh_signing_key")
        except Exception:  # noqa: BLE001 -- A corrupt state snapshot falls back to owner reapproval rather than crashing startup.
            # a corrupt state file must not take the auth server down; it
            # simply means every client re-approves once
            return
        self._access, self._refresh, self._clients = access, refresh, clients
        self._families, self._retired_legacy = families, retired
        changed = False
        if isinstance(key, str) and key:
            self._signing_key = key
        else:
            changed = True
        changed = self._assign_legacy_families() or changed
        self._clamp_loaded_lifetimes(changed)
        self._record_google_bearers(self._access.values())

    def _assign_legacy_families(self) -> bool:
        """State written before families existed: each refresh token becomes
        its own family (each one came from a separate approval or rotation
        chain). A legacy access token joins its client's family when the
        client holds exactly one; otherwise it stays unattached and expires
        within the hour as before."""
        now = _now()
        changed = False
        for token, rt in list(self._refresh.items()):
            if rt.family is None:
                rt = rt.model_copy(update={"family": self._new_family(rt.client_id)})
                self._refresh[token] = rt
                changed = True
            elif rt.family not in self._families:
                self._families[rt.family] = {"client_id": rt.client_id, "created_at": now}
                changed = True
        by_client: dict[str, set[str]] = {}
        for rt in self._refresh.values():
            by_client.setdefault(rt.client_id, set()).add(rt.family)
        for token, at in list(self._access.items()):
            if at.family is None:
                owned = by_client.get(at.client_id, set())
                if len(owned) == 1:
                    self._access[token] = at.model_copy(update={"family": next(iter(owned))})
                    changed = True
            elif at.family not in self._families:
                self._families[at.family] = {"client_id": at.client_id, "created_at": now}
                changed = True
        return changed

    def _access_ttl_for_client(self, client_id: str) -> int:
        if _is_google_connector(self._clients.get(client_id)):
            return _REFRESH_TTL_S
        return _ACCESS_TTL_S

    def _clamp_loaded_lifetimes(self, changed: bool = False) -> None:
        """Tokens issued under an older, longer lifetime keep working only up to
        the current maximum, counted from this load. A lifetime is never
        extended. Clients hold refresh tokens and renew silently."""
        now = _now()
        refresh_limit = int(now + _REFRESH_TTL_S)
        for token, at in list(self._access.items()):
            access_limit = int(now + self._access_ttl_for_client(at.client_id))
            if at.expires_at is None or at.expires_at > access_limit:
                self._access[token] = at.model_copy(update={"expires_at": access_limit})
                changed = True
        for token, rt in list(self._refresh.items()):
            if rt.expires_at is None or rt.expires_at > refresh_limit:
                self._refresh[token] = rt.model_copy(update={"expires_at": refresh_limit})
                changed = True
        if changed:
            self._save()

    def _prune_families(self) -> None:
        """A family lives while any token of it does; its retired-token
        records go with it."""
        live = {t.family for t in (*self._access.values(), *self._refresh.values())
                if t.family is not None}
        for family in list(self._families):
            if family not in live:
                self._families.pop(family, None)
        self._retired_legacy = {k: v for k, v in self._retired_legacy.items()
                                if v.get("family") in self._families}

    def _save(self) -> None:
        if not self._path:
            return
        with self._save_lock:
            self._save_unlocked()

    def _save_unlocked(self) -> None:
        now = _now()
        self._prune_families()
        access = {k: v for k, v in self._access.items()
                  if v.expires_at is None or v.expires_at > now}
        refresh = {k: v for k, v in self._refresh.items()
                   if v.expires_at is None or v.expires_at > now}
        live = {t.family for t in (*access.values(), *refresh.values())}
        data = {
            "access": {k: v.model_dump(mode="json") for k, v in access.items()},
            "refresh": {k: v.model_dump(mode="json") for k, v in refresh.items()},
            "clients": {k: v.model_dump(mode="json") for k, v in self._clients.items()},
            "families": {k: v for k, v in self._families.items() if k in live},
            "retired_legacy": {k: v for k, v in self._retired_legacy.items()
                               if v.get("family") in live},
            "refresh_signing_key": self._signing_key,
        }
        directory = os.path.dirname(self._path) or "."
        os.makedirs(directory, mode=0o700, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=os.path.basename(self._path) + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(tmp, self._path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        try:
            self._loaded_mtime = os.path.getmtime(self._path)
        except OSError:
            pass

    # ----- Gemini's bearer, shared through the registry -----
    def _google_expiries(self, tokens) -> dict[str, float]:
        now = _now()
        return {google_bearers.digest(at.token): float(at.expires_at or (now + _REFRESH_TTL_S))
                for at in tokens
                if _is_google_connector(self._clients.get(at.client_id))}

    def _record_google_bearers(self, tokens) -> None:
        if self._google_bearers is not None:
            now = _now()
            self._google_bearers.record(
                {k: v for k, v in self._google_expiries(tokens).items() if v > now},
                issuer="vecgrep", now=now)

    def _withdraw_google_bearers(self, tokens) -> None:
        if self._google_bearers is not None:
            self._google_bearers.revoke(self._google_expiries(tokens), now=_now())

    def _approved_google_grant(self) -> ResourceRefreshToken | None:
        """The live refresh token of an approved Google client, latest expiry
        first. A client that only registered (open DCR) holds none."""
        now = _now()
        grants = [rt for rt in self._refresh.values()
                  if _is_google_connector(self._clients.get(rt.client_id))
                  and (rt.expires_at is None or rt.expires_at > now)]
        return max(grants, key=lambda rt: rt.expires_at or float("inf"), default=None)

    def _registered_google_bearer(self, token: str) -> FamilyAccessToken | None:
        """A bearer another MCP on this origin issued to Gemini, accepted under
        this store's own approval of Gemini: its client, scopes and family,
        and no later expiry than that approval. Never stored here, so a
        revocation or expiry in the registry applies on the next call."""
        if self._google_bearers is None:
            return None
        entry = self._google_bearers.lookup(google_bearers.digest(token), now=_now())
        grant = self._approved_google_grant() if entry else None
        if grant is None:
            return None
        expires = float(entry["expires_at"])
        if grant.expires_at is not None:
            expires = min(expires, float(grant.expires_at))
        return FamilyAccessToken(
            token=token,
            client_id=grant.client_id,
            scopes=list(grant.scopes),
            expires_at=int(expires),
            resource=self._default_resource or grant.resource,
            family=grant.family,
        )

    # ----- clients -----
    def save_client(self, info: OAuthClientInformationFull) -> None:
        self._clients[info.client_id] = info
        self._prune_unapproved_clients()
        self._save()

    def _prune_unapproved_clients(self) -> None:
        """Bound anonymous DCR state without disrupting approved clients.

        Possessing any access or refresh token marks a client authorized. Such
        clients are never evicted here, even if a token has expired; explicit
        client revocation remains the operation that makes one pruneable.
        """
        authorized = {
            token.client_id
            for token in (*self._access.values(), *self._refresh.values())
        }
        now = _now()
        unapproved: list[tuple[float, str]] = []
        for client_id, client in list(self._clients.items()):
            if client_id in authorized:
                continue
            issued_at = client.client_id_issued_at
            if (
                issued_at is not None
                and float(issued_at) <= now - self._unapproved_client_ttl_s
            ):
                self._clients.pop(client_id, None)
                continue
            unapproved.append((float(issued_at or now), client_id))

        overflow = len(unapproved) - self._max_unapproved_clients
        for _, client_id in sorted(unapproved)[:max(0, overflow)]:
            self._clients.pop(client_id, None)

    def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._clients.get(client_id)

    def clients(self) -> dict[str, OAuthClientInformationFull]:
        return self._clients

    # ----- families -----
    def _new_family(self, client_id: str) -> str:
        family = _new_secret("vgrf")
        self._families[family] = {"client_id": client_id, "created_at": _now()}
        return family

    def _signature(self, client_id: str, family: str, nonce: str) -> str:
        message = f"{client_id}\0{family}\0{nonce}".encode()
        digest = hmac.new(self._signing_key.encode(), message, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(digest).decode().rstrip("=")

    def _signed_family(self, client_id: str, token: str) -> str | None:
        """The family a signed refresh token was minted for, if the signature
        holds for this client and the family is still alive."""
        parts = token.split(".")
        if len(parts) != 4 or parts[0] != "vgrt" or not all(parts):
            return None
        _prefix, family, nonce, signature = parts
        if not hmac.compare_digest(signature, self._signature(client_id, family, nonce)):
            return None
        meta = self._families.get(family)
        if meta is None or meta.get("client_id") != client_id:
            return None
        return family

    def _replayed_family(self, client_id: str, token: str) -> str | None:
        """The live family of a refresh token this client already rotated."""
        record = self._retired_legacy.get(_legacy_key(token))
        if record is not None:
            family = record.get("family")
            meta = self._families.get(family)
            if record.get("client_id") == client_id and meta and meta.get("client_id") == client_id:
                return family
            return None
        return self._signed_family(client_id, token)

    def _revoke_family(self, family: str) -> None:
        self._withdraw_google_bearers([at for at in self._access.values() if at.family == family])
        self._access = {k: v for k, v in self._access.items() if v.family != family}
        self._refresh = {k: v for k, v in self._refresh.items() if v.family != family}
        self._families.pop(family, None)
        self._retired_legacy = {k: v for k, v in self._retired_legacy.items()
                                if v.get("family") != family}

    def _revoke_if_replayed(self, client_id: str, token: str) -> bool:
        family = self._replayed_family(client_id, token)
        if family is None:
            return False
        self._revoke_family(family)
        self._save()
        return True

    # ----- access tokens -----
    def _trim_access(self, client_id: str) -> None:
        """Drop the client's oldest access tokens so a new one fits the cap."""
        mine = sorted(((at.expires_at or 0, token) for token, at in self._access.items()
                       if at.client_id == client_id))
        drop = [token for _exp, token in mine[:max(0, len(mine) + 1 - _MAX_ACCESS_PER_CLIENT)]]
        if drop:
            self._withdraw_google_bearers([self._access[t] for t in drop])
            for token in drop:
                self._access.pop(token, None)

    def _new_access(self, client_id: str, scopes: list[str], ttl_s: int,
                    resource: str | None, family: str | None) -> FamilyAccessToken:
        self._trim_access(client_id)
        at = FamilyAccessToken(
            token=_new_secret("vgat"),
            client_id=client_id,
            scopes=list(scopes),
            expires_at=int(_now() + ttl_s),
            resource=resource,
            family=family,
        )
        self._access[at.token] = at
        self._record_google_bearers([at])
        return at

    def issue_access_token(self, client_id: str, scopes: list[str],
                           ttl_s: int = _ACCESS_TTL_S,
                           resource: str | None = None,
                           family: str | None = None) -> AccessToken:
        at = self._new_access(client_id, scopes, ttl_s, resource, family)
        self._save()
        return at

    def load_access_token(self, token: str) -> AccessToken | None:
        at = self._access.get(token)
        if at is None and self._path and os.path.exists(self._path):
            try:
                mtime = os.path.getmtime(self._path)
            except OSError:
                mtime = self._loaded_mtime
            if mtime != self._loaded_mtime:
                self._load()
                at = self._access.get(token)
        if at is not None and at.expires_at is not None and at.expires_at <= _now():
            at = None   # another MCP may hold a longer-lived copy of this bearer
        if at is None:
            at = self._registered_google_bearer(token)
            if at is None:
                return None
        if at.expires_at is not None and at.expires_at <= _now():
            return None
        if (self._google_bearers is not None
                and _is_google_connector(self._clients.get(at.client_id))
                and self._google_bearers.is_revoked(google_bearers.digest(at.token))):
            return None
        return at

    def token_has_scope(self, token: str, scope: str) -> bool:
        at = self.load_access_token(token)
        return bool(at and scope in at.scopes)

    # ----- refresh tokens -----
    def _new_refresh(self, client_id: str, scopes: list[str], ttl_s: int,
                     resource: str | None, family: str) -> ResourceRefreshToken:
        nonce = secrets.token_urlsafe(32)
        rt = ResourceRefreshToken(
            token=f"vgrt.{family}.{nonce}.{self._signature(client_id, family, nonce)}",
            client_id=client_id,
            scopes=list(scopes),
            expires_at=int(_now() + ttl_s),
            resource=resource,
            family=family,
        )
        self._refresh[rt.token] = rt
        return rt

    def issue_refresh_token(self, client_id: str, scopes: list[str],
                            ttl_s: int = _REFRESH_TTL_S,
                            resource: str | None = None,
                            family: str | None = None) -> RefreshToken:
        """Without a family this starts a new one, as an approval does."""
        if family is None or family not in self._families:
            family = self._new_family(client_id)
        rt = self._new_refresh(client_id, scopes, ttl_s, resource, family)
        self._save()
        return rt

    def issue_token_pair(self, client_id: str, scopes: list[str],
                         resource: str | None = None) -> tuple[AccessToken, RefreshToken]:
        """An approval: a new family holding one access and one refresh token."""
        family = self._new_family(client_id)
        access_ttl = self._access_ttl_for_client(client_id)
        at = self._new_access(client_id, scopes, access_ttl, resource, family)
        rt = self._new_refresh(client_id, scopes, _REFRESH_TTL_S, resource, family)
        self._save()
        return at, rt

    def load_refresh_token(self, client_id: str, token: str) -> RefreshToken | None:
        """A retired token of this client revokes its family and loads as
        nothing."""
        rt = self._refresh.get(token)
        if rt is None:
            self._revoke_if_replayed(client_id, token)
            return None
        if rt.client_id != client_id:
            return None
        if rt.expires_at is not None and rt.expires_at <= _now():
            return None
        return rt

    def rotate_refresh_token(
        self, client_id: str, token: str, scopes: list[str],
    ) -> tuple[AccessToken, RefreshToken] | None:
        """Consume one refresh token and issue its replacement pair in the same
        family, in one step and one save. A token this client already rotated
        revokes the family. A request for scopes beyond the token's consumes
        nothing."""
        old = self._refresh.get(token)
        if old is None:
            self._revoke_if_replayed(client_id, token)
            return None
        if old.client_id != client_id:
            return None
        if old.expires_at is not None and old.expires_at <= _now():
            return None
        granted = list(scopes or old.scopes)
        if not granted or not set(granted).issubset(old.scopes):
            return None
        family = old.family
        if family is None or family not in self._families:
            family = self._new_family(client_id)
        self._refresh.pop(token, None)
        if self._signed_family(client_id, token) != family:
            # a pre-family token has no signature to recognise it by later
            self._retired_legacy[_legacy_key(token)] = {"client_id": client_id, "family": family}
        access_ttl = self._access_ttl_for_client(client_id)
        at = self._new_access(client_id, granted, access_ttl, old.resource, family)
        rt = self._new_refresh(client_id, granted, _REFRESH_TTL_S, old.resource, family)
        self._save()
        return at, rt

    # ----- authorization codes (single-use) -----
    def issue_authorization_code(self, client_id: str, scopes: list[str],
                                 code_challenge: str, redirect_uri: str,
                                 ttl_s: int = _CODE_TTL_S,
                                 resource: str | None = None,
                                 redirect_uri_provided_explicitly: bool = True) -> AuthorizationCode:
        code = AuthorizationCode(
            code=_new_secret("vgac"),
            scopes=list(scopes),
            expires_at=_now() + ttl_s,
            client_id=client_id,
            code_challenge=code_challenge,
            redirect_uri=redirect_uri,
            redirect_uri_provided_explicitly=redirect_uri_provided_explicitly,
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
        """Revoke an access OR refresh token by its value. Idempotent. A
        refresh token carries the approval, so revoking one revokes its whole
        family; revoking an access token revokes only that token."""
        at = self._access.pop(token, None)
        if at is not None:
            self._withdraw_google_bearers([at])
        elif self._google_bearers is not None:
            # a bearer another MCP issued: revoke it there too, if registered
            self._google_bearers.revoke({google_bearers.digest(token): None}, now=_now())
        rt = self._refresh.pop(token, None)
        if rt is not None and rt.family is not None:
            self._revoke_family(rt.family)
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
            "families": len(self._families),
            "by_client": by_client,
        }

    def revoke_client(self, client_id: str) -> int:
        """Kill every access + refresh token a client holds (the inventory
        panel's revoke button). Returns how many tokens died."""
        doomed_a = [t for t, at in self._access.items() if at.client_id == client_id]
        doomed_r = [t for t, rt in self._refresh.items() if rt.client_id == client_id]
        self._withdraw_google_bearers([self._access[t] for t in doomed_a])
        for t in doomed_a:
            self._access.pop(t, None)
        for t in doomed_r:
            self._refresh.pop(t, None)
        self._codes = {c: ac for c, ac in self._codes.items()
                       if ac.client_id != client_id}
        for family, meta in list(self._families.items()):
            if meta.get("client_id") == client_id:
                self._revoke_family(family)
        self._save()
        return len(doomed_a) + len(doomed_r)
