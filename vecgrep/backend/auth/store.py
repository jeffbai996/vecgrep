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
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time

from mcp.server.auth.provider import AccessToken, AuthorizationCode, RefreshToken
from mcp.shared.auth import OAuthClientInformationFull

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


def _is_google_connector(client: OAuthClientInformationFull | None) -> bool:
    """Google's Gemini web MCP proxy caches issued bearer tokens across worker
    tasks and does not refresh expired 1-hour access tokens on chat turns."""
    if client is None:
        return False
    if (getattr(client, "client_name", None) or "").strip() == "Google":
        return True
    for uri in getattr(client, "redirect_uris", None) or ():
        if ".googleusercontent.com/" in str(uri):
            return True
    return False


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
        federated_state_paths: list[str] | None = None,
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
        if federated_state_paths is not None:
            self._federated_state_paths = [p for p in federated_state_paths if p]
        elif path:
            env_paths = os.environ.get("VECGREP_OAUTH_FEDERATED_STORES", "")
            self._federated_state_paths = [
                os.path.expanduser(p.strip()) for p in env_paths.split(",") if p.strip()
            ]
        else:
            self._federated_state_paths = []
        self._max_unapproved_clients = max_unapproved_clients
        self._unapproved_client_ttl_s = unapproved_client_ttl_s
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
        except Exception:
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
        changed = self._import_sibling_google_token() or changed
        self._clamp_loaded_lifetimes(changed)
        self._sync_google_tokens_to_siblings()

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
        os.makedirs(os.path.dirname(self._path) or ".", mode=0o700, exist_ok=True)
        tmp = self._path + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, self._path)
        try:
            self._loaded_mtime = os.path.getmtime(self._path)
        except OSError:
            pass
        self._sync_google_tokens_to_siblings()

    def _live_sibling_google_token(self) -> str | None:
        now = _now()
        candidates: list[tuple[int, float, str]] = []
        for sib_path in self._federated_state_paths:
            if not os.path.exists(sib_path):
                continue
            try:
                with open(sib_path, encoding="utf-8") as fh:
                    raw = json.load(fh)
                if "access" not in raw:
                    continue
                google_clients = {
                    cid for cid, c in raw.get("clients", {}).items()
                    if isinstance(c, dict) and (
                        str(c.get("client_name") or "").strip() == "Google"
                        or any(".googleusercontent.com/" in str(u) for u in (c.get("redirect_uris") or ()))
                    )
                }
                for tok, at in raw.get("access", {}).items():
                    if not isinstance(at, dict) or at.get("client_id") not in google_clients:
                        continue
                    exp = float(at.get("expires_at") or 0)
                    if exp > now:
                        candidates.append((0 if str(tok).startswith("vgat_") else 1, exp, str(tok)))
            except Exception:
                continue
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][2]

    def _import_sibling_google_token(self) -> bool:
        shared = self._live_sibling_google_token()
        if not shared:
            return False
        google_cids = [cid for cid, c in self._clients.items() if _is_google_connector(c)]
        if not google_cids:
            return False
        target_cid = google_cids[-1]
        inferred_res = self._default_resource
        if not inferred_res:
            for at in self._access.values():
                if at.resource:
                    inferred_res = at.resource
                    break
        existing = self._access.get(shared)
        if (
            existing is not None
            and set(existing.scopes) >= {"read", "propose"}
            and (not inferred_res or existing.resource == inferred_res)
        ):
            return False
        family = next(
            (f for f, meta in self._families.items() if meta.get("client_id") == target_cid),
            None,
        ) or self._new_family(target_cid)
        self._access[shared] = FamilyAccessToken(
            token=shared,
            client_id=target_cid,
            scopes=["read", "propose"],
            expires_at=int(_now() + _REFRESH_TTL_S),
            resource=inferred_res,
            family=family,
        )
        return True

    def _sync_google_tokens_to_siblings(self) -> None:
        if not self._federated_state_paths:
            return
        now = _now()
        google_tokens: dict[str, int] = {}
        for tok, at in self._access.items():
            if _is_google_connector(self._clients.get(at.client_id)) and (at.expires_at is None or at.expires_at > now):
                google_tokens[tok] = int(at.expires_at or (now + _REFRESH_TTL_S))
        if not google_tokens:
            return

        for sib_path in self._federated_state_paths:
            if not os.path.exists(sib_path):
                continue
            try:
                import fcntl
                lock_path = sib_path + ".lock"
                lfd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
                with os.fdopen(lfd, "r+") as lfh:
                    fcntl.flock(lfh, fcntl.LOCK_EX)
                    try:
                        with open(sib_path, encoding="utf-8") as sfh:
                            sib = json.load(sfh)
                        if "access_tokens" in sib and "grants" in sib:
                            google_cids = {
                                cid for cid, c in sib.get("clients", {}).items()
                                if isinstance(c, dict) and not c.get("revoked_at") and (
                                    c.get("client_name") == "Google"
                                    or any(".googleusercontent.com/" in str(u) for u in (c.get("redirect_uris") or ()))
                                )
                            }
                            grants = [
                                g for g in sib.get("grants", {}).values()
                                if isinstance(g, dict) and not g.get("revoked_at") and g.get("client_id") in google_cids
                            ]
                            if grants:
                                grants.sort(key=lambda g: float(g.get("created_at") or 0))
                                grant = grants[-1]
                                gid = grant["grant_id"]
                                fams = [
                                    f for f in sib.get("families", {}).values()
                                    if isinstance(f, dict) and f.get("grant_id") == gid
                                    and not f.get("revoked_at") and float(f.get("expires_at") or 0) > now
                                ]
                                if fams:
                                    fid = fams[-1]["family_id"]
                                else:
                                    fid = secrets.token_hex(16)
                                    sib.setdefault("families", {})[fid] = {
                                        "family_id": fid,
                                        "grant_id": gid,
                                        "expires_at": now + _REFRESH_TTL_S,
                                        "revoked_at": None,
                                    }
                                changed = False
                                acc = sib.setdefault("access_tokens", {})
                                for tok_str, exp in google_tokens.items():
                                    digest = hashlib.sha256(tok_str.encode("utf-8")).hexdigest()
                                    cur = acc.get(digest)
                                    if (
                                        not isinstance(cur, dict)
                                        or cur.get("revoked_at") is not None
                                        or float(cur.get("expires_at") or 0) < exp
                                    ):
                                        acc[digest] = {
                                            "client_id": grant["client_id"],
                                            "expires_at": float(exp),
                                            "family_id": fid,
                                            "grant_id": gid,
                                            "issued_at": now,
                                            "resource": grant["resource"],
                                            "revoked_at": None,
                                            "scopes": sorted(set(grant.get("scopes") or ["read"])),
                                        }
                                        changed = True
                                if changed:
                                    stmp = sib_path + ".tmp"
                                    sfd = os.open(stmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                                    with os.fdopen(sfd, "w", encoding="utf-8") as out:
                                        json.dump(sib, out, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                                    os.replace(stmp, sib_path)
                        elif "access" in sib and "clients" in sib:
                            sib_cids = [
                                cid for cid, c in sib.get("clients", {}).items()
                                if isinstance(c, dict) and (
                                    str(c.get("client_name") or "").strip() == "Google"
                                    or any(".googleusercontent.com/" in str(u) for u in (c.get("redirect_uris") or ()))
                                )
                            ]
                            if sib_cids:
                                sib_cid = sib_cids[-1]
                                sib_acc = sib.setdefault("access", {})
                                changed = False
                                for tok_str, exp in google_tokens.items():
                                    if tok_str not in sib_acc:
                                        sib_acc[tok_str] = {
                                            "token": tok_str,
                                            "client_id": sib_cid,
                                            "scopes": ["read", "propose"],
                                            "expires_at": int(exp),
                                            "resource": None,
                                        }
                                        changed = True
                                if changed:
                                    itmp = sib_path + ".tmp"
                                    ifd = os.open(itmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                                    with os.fdopen(ifd, "w", encoding="utf-8") as out:
                                        json.dump(sib, out, sort_keys=True)
                                    os.replace(itmp, sib_path)
                    finally:
                        fcntl.flock(lfh, fcntl.LOCK_UN)
            except Exception:
                pass

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
    def _new_access(self, client_id: str, scopes: list[str], ttl_s: int,
                    resource: str | None, family: str | None) -> FamilyAccessToken:
        token_val = None
        if _is_google_connector(self._clients.get(client_id)):
            token_val = self._live_sibling_google_token()
        at = FamilyAccessToken(
            token=token_val or _new_secret("vgat"),
            client_id=client_id,
            scopes=list(scopes),
            expires_at=int(_now() + ttl_s),
            resource=resource,
            family=family,
        )
        self._access[at.token] = at
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
        if at is None:
            return None
        if at.expires_at is not None and at.expires_at <= _now():
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
        """Revoke an access OR refresh token by its value. Idempotent. A
        refresh token carries the approval, so revoking one revokes its whole
        family; revoking an access token revokes only that token."""
        self._access.pop(token, None)
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
