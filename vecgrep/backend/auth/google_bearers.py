"""The registry of Gemini's connector bearers, shared by every MCP on this origin.

Gemini keeps one bearer per hostname and presents the one it was issued last
to every MCP behind that hostname. Each MCP still issues and stores its own
tokens; this registry is where they all record the bearers they issue to
Gemini and the ones they revoke. It holds a sha256 digest, an expiry and a
revocation time per bearer, never a token value. An MCP accepts a registered
bearer only under its own approval of Gemini, so the registry says whether a
bearer is live and each MCP decides what it may do with it.

Several MCPs may hold copies of one bearer with different expiries, so an
entry lasts as long as the longest copy any of them records. A revocation
therefore holds for as long as any MCP could still accept that bearer.

Several processes write the file: every change takes `<path>.lock` and
replaces the file atomically. An unreadable registry accepts nothing new and
is never overwritten.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager

log = logging.getLogger(__name__)

ENV_VAR = "OAUTH_GOOGLE_BEARER_REGISTRY"


def digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class RegistryUnreadable(RuntimeError):
    pass


class GoogleBearerRegistry:
    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(os.path.expanduser(path))

    @contextmanager
    def _locked(self):
        os.makedirs(os.path.dirname(self.path), mode=0o700, exist_ok=True)
        fd = os.open(self.path + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(fd, "r+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _read(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as fh:
                state = json.load(fh)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise RegistryUnreadable(str(exc)) from exc
        bearers = state.get("bearers") if isinstance(state, dict) else None
        if not isinstance(bearers, dict):
            raise RegistryUnreadable("no bearers map")
        return bearers

    def _write(self, bearers: dict, now: float) -> None:
        live = {k: v for k, v in bearers.items() if float(v.get("expires_at") or 0) > now}
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path),
                                   prefix=".google-bearers.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"version": 1, "bearers": live}, fh, sort_keys=True)
                fh.flush()
                # a torn file after power loss would stay unreadable for good
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @staticmethod
    def _extend(entry: dict, expires_at: float | None) -> bool:
        if expires_at is None or float(expires_at) <= float(entry.get("expires_at") or 0):
            return False
        entry["expires_at"] = float(expires_at)
        return True

    def record(self, expiries: dict[str, float], *, issuer: str,
               now: float | None = None) -> None:
        """Add bearers an MCP holds for Gemini, as digest -> expiry. An entry
        already present only has its expiry raised to the later one; its
        revocation is never undone."""
        if not expiries:
            return
        now = time.time() if now is None else now
        try:
            with self._locked():
                bearers = self._read()
                changed = False
                for key, expires_at in expiries.items():
                    entry = bearers.get(key)
                    if isinstance(entry, dict):
                        changed = self._extend(entry, expires_at) or changed
                    elif float(expires_at) > now:
                        bearers[key] = {"issuer": issuer, "issued_at": now,
                                        "expires_at": float(expires_at), "revoked_at": None}
                        changed = True
                if changed:
                    self._write(bearers, now)
        except Exception as exc:
            log.error("Google bearer registry was not updated: %s", exc)

    def lookup(self, key: str, now: float | None = None) -> dict | None:
        """The live, unrevoked entry for this digest, or None."""
        now = time.time() if now is None else now
        try:
            with self._locked():
                entry = self._read().get(key)
        except Exception as exc:
            log.error("Google bearer registry was not read: %s", exc)
            return None
        if (not isinstance(entry, dict) or entry.get("revoked_at") is not None
                or float(entry.get("expires_at") or 0) <= now):
            return None
        return dict(entry)

    def is_revoked(self, key: str) -> bool:
        try:
            with self._locked():
                entry = self._read().get(key)
        except Exception as exc:
            log.error("Google bearer registry was not read: %s", exc)
            return False
        return isinstance(entry, dict) and entry.get("revoked_at") is not None

    def revoke(self, expiries: dict[str, float | None], *, now: float | None = None) -> None:
        """Mark bearers revoked for every MCP, as digest -> expiry. An entry
        stays until the latest expiry offered for it, so no MCP records or
        accepts it again; a digest not yet registered is added revoked when
        its expiry is known."""
        if not expiries:
            return
        now = time.time() if now is None else now
        try:
            with self._locked():
                bearers = self._read()
                changed = False
                for key, expires_at in expiries.items():
                    entry = bearers.get(key)
                    if isinstance(entry, dict):
                        changed = self._extend(entry, expires_at) or changed
                        if entry.get("revoked_at") is None:
                            entry["revoked_at"] = now
                            changed = True
                    elif expires_at is not None and float(expires_at) > now:
                        bearers[key] = {"issuer": None, "issued_at": now,
                                        "expires_at": float(expires_at), "revoked_at": now}
                        changed = True
                if changed:
                    self._write(bearers, now)
        except Exception as exc:
            log.error("Google bearer registry revocation was not written: %s", exc)
