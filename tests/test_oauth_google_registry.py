"""Gemini's bearer across the MCPs on one origin, through the shared registry.

Gemini keeps one bearer per hostname, so every MCP behind it must accept the
bearer another one issued. The registry records digests; this store accepts a
registered bearer only under its own approval of Gemini, and a revocation
anywhere holds everywhere. Every store and registry here is a synthetic file
under tmp_path.
"""
from __future__ import annotations

import json
import time

from mcp.shared.auth import OAuthClientInformationFull

from vecgrep.backend.auth.google_bearers import GoogleBearerRegistry, digest
from vecgrep.backend.auth.store import TokenStore

REDIRECT = "https://oauth-redirect.googleusercontent.com/r/custom-mcp"
GEMINI_REDIRECTS = [
    f"https://{host}.googleusercontent.com/{kind}/user_bound_custom-mcp-host_example"
    for kind in ("r", "a")
    for host in ("oauth-redirect-sandbox", "oauth-redirect-test", "oauth-redirect")
]
RESOURCE = "https://host.example/vg/vecgrep-mcp"
DAY = 86400


def _client(client_id="vg-google", name="Google", redirects=(REDIRECT,)):
    return OAuthClientInformationFull(client_id=client_id, client_name=name,
                                      redirect_uris=list(redirects))


def _paths(tmp_path):
    return str(tmp_path / "vecgrep.json"), str(tmp_path / "shared" / "google-bearers.json")


def _store(tmp_path):
    path, registry = _paths(tmp_path)
    return TokenStore(path=path, google_registry_path=registry)


def _approved(tmp_path, scopes=("read",)):
    store = _store(tmp_path)
    store.save_client(_client())
    at, rt = store.issue_token_pair("vg-google", list(scopes), resource=RESOURCE)
    return store, at, rt


def _registry(tmp_path):
    return GoogleBearerRegistry(_paths(tmp_path)[1])


def _registered_elsewhere(tmp_path, token, expires_at):
    _registry(tmp_path).record({digest(token): expires_at}, issuer="another-mcp")


# ── issuing and accepting ───────────────────────────────────────────────────

def test_a_bearer_issued_to_gemini_is_recorded_by_digest(tmp_path):
    _, at, _ = _approved(tmp_path)

    entry = _registry(tmp_path).lookup(digest(at.token))
    assert entry["issuer"] == "vecgrep"
    assert at.token not in open(_paths(tmp_path)[1]).read()


def test_a_bearer_another_mcp_registered_is_accepted_under_this_approval(tmp_path):
    store, _, _ = _approved(tmp_path, scopes=("read",))
    _registered_elsewhere(tmp_path, "from-another-mcp", time.time() + 30 * DAY)

    at = store.load_access_token("from-another-mcp")

    assert at.client_id == "vg-google"
    assert at.scopes == ["read"]
    assert at.resource == RESOURCE


def test_a_registered_bearer_is_refused_without_an_approval_here(tmp_path):
    store = _store(tmp_path)
    store.save_client(_client())                      # registered, never approved
    _registered_elsewhere(tmp_path, "from-another-mcp", time.time() + 30 * DAY)

    assert store.load_access_token("from-another-mcp") is None


def test_a_registered_bearer_never_outlives_the_approval_here(tmp_path):
    store, _, rt = _approved(tmp_path)
    _registered_elsewhere(tmp_path, "from-another-mcp", time.time() + 900 * DAY)

    assert store.load_access_token("from-another-mcp").expires_at <= rt.expires_at


def test_a_registered_bearer_is_refused_once_it_expires(tmp_path, monkeypatch):
    store, _, _ = _approved(tmp_path)
    _registered_elsewhere(tmp_path, "from-another-mcp", time.time() + DAY)
    later = time.time() + 2 * DAY
    monkeypatch.setattr("vecgrep.backend.auth.store._now", lambda: later)

    assert store.load_access_token("from-another-mcp") is None


def test_without_a_registry_a_foreign_bearer_is_refused(tmp_path):
    store = TokenStore(path=str(tmp_path / "vecgrep.json"))
    store.save_client(_client())
    store.issue_token_pair("vg-google", ["read"], resource=RESOURCE)

    assert store.load_access_token("from-another-mcp") is None


def test_ordinary_clients_never_touch_the_registry(tmp_path):
    store = _store(tmp_path)
    store.save_client(_client("notes", "Notes", ["https://notes.example/callback"]))
    store.issue_token_pair("notes", ["read"], resource=RESOURCE)

    import os
    assert not os.path.exists(_paths(tmp_path)[1])


def test_bearers_held_before_the_registry_existed_are_recorded_on_start_up(tmp_path):
    path, registry = _paths(tmp_path)
    before = TokenStore(path=path)
    before.save_client(_client())
    at, _ = before.issue_token_pair("vg-google", ["read"], resource=RESOURCE)

    TokenStore(path=path, google_registry_path=registry)

    assert _registry(tmp_path).lookup(digest(at.token)) is not None


# ── revoking ────────────────────────────────────────────────────────────────

def test_a_bearer_revoked_at_another_mcp_stops_loading_here(tmp_path):
    store, at, _ = _approved(tmp_path)

    _registry(tmp_path).revoke({digest(at.token): None})

    assert store.load_access_token(at.token) is None


def test_revoking_the_bearer_here_revokes_it_in_the_registry(tmp_path):
    store, at, _ = _approved(tmp_path)

    store.revoke(at.token)

    assert _registry(tmp_path).is_revoked(digest(at.token))


def test_revoking_a_registered_bearer_this_store_never_held_revokes_it_everywhere(tmp_path):
    store, _, _ = _approved(tmp_path)
    _registered_elsewhere(tmp_path, "from-another-mcp", time.time() + 30 * DAY)

    store.revoke("from-another-mcp")

    assert store.load_access_token("from-another-mcp") is None
    assert _registry(tmp_path).is_revoked(digest("from-another-mcp"))


def test_revoking_the_google_client_revokes_its_bearers_in_the_registry(tmp_path):
    store, at, _ = _approved(tmp_path)

    store.revoke_client("vg-google")

    assert _registry(tmp_path).is_revoked(digest(at.token))


def test_a_replayed_refresh_token_revokes_the_bearer_in_the_registry(tmp_path):
    store, at, rt = _approved(tmp_path)
    assert store.rotate_refresh_token("vg-google", rt.token, ["read"]) is not None

    assert store.load_refresh_token("vg-google", rt.token) is None   # the replay

    assert _registry(tmp_path).is_revoked(digest(at.token))


def test_a_revocation_holds_across_a_restart(tmp_path):
    store, at, _ = _approved(tmp_path)
    store.revoke(at.token)

    path, registry = _paths(tmp_path)
    reloaded = TokenStore(path=path, google_registry_path=registry)
    assert reloaded.load_access_token(at.token) is None


# ── which client is Google ──────────────────────────────────────────────────

def _one_hour_or_less(at):
    return at.expires_at - time.time() <= 3600 + 5


def test_gemini_registering_its_production_sandbox_and_test_redirects_is_google(tmp_path):
    store = _store(tmp_path)
    store.save_client(_client("gemini", "Google", GEMINI_REDIRECTS))
    at, _ = store.issue_token_pair("gemini", ["read"], resource=RESOURCE)
    assert not _one_hour_or_less(at)


def test_a_client_named_google_with_a_foreign_redirect_gets_an_ordinary_token(tmp_path):
    store = _store(tmp_path)
    store.save_client(_client("impostor", "Google", ["https://evil.example/callback"]))
    at, _ = store.issue_token_pair("impostor", ["read"], resource=RESOURCE)
    assert _one_hour_or_less(at)


def test_a_redirect_that_only_mentions_googleusercontent_in_its_path_is_not_google(tmp_path):
    store = _store(tmp_path)
    store.save_client(_client("impostor", "Notes",
                              ["https://evil.example/x.googleusercontent.com/r/custom-mcp"]))
    at, _ = store.issue_token_pair("impostor", ["read"], resource=RESOURCE)
    assert _one_hour_or_less(at)


def test_a_client_adding_a_foreign_redirect_to_google_ones_is_not_google(tmp_path):
    store = _store(tmp_path)
    store.save_client(_client("impostor", "Google", [REDIRECT, "https://evil.example/callback"]))
    at, _ = store.issue_token_pair("impostor", ["read"], resource=RESOURCE)
    assert _one_hour_or_less(at)


def test_an_impostor_cannot_accept_registered_bearers(tmp_path):
    store = _store(tmp_path)
    store.save_client(_client("impostor", "Google", ["https://evil.example/callback"]))
    store.issue_token_pair("impostor", ["read"], resource=RESOURCE)
    _registered_elsewhere(tmp_path, "from-another-mcp", time.time() + 30 * DAY)

    assert store.load_access_token("from-another-mcp") is None
