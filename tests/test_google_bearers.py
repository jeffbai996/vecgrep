from __future__ import annotations

import json
import os
import time
from pathlib import Path

from mcp.shared.auth import OAuthClientInformationFull

from vecgrep.backend.auth.google_bearers import GoogleBearerRegistry, digest
from vecgrep.backend.auth.store import TokenStore

# ── test_google_bearer_registry ──

DAY = 86400


def _registry(tmp_path):
    return GoogleBearerRegistry(str(tmp_path / "shared" / "google-bearers.json"))


def test_a_recorded_bearer_is_live_until_its_expiry(tmp_path):
    reg = _registry(tmp_path)
    now = time.time()
    reg.record({digest("bearer-1"): now + DAY}, issuer="one", now=now)

    entry = reg.lookup(digest("bearer-1"), now=now)
    assert entry is not None
    assert entry["issuer"] == "one"
    assert reg.lookup(digest("bearer-1"), now=now + 2 * DAY) is None


def test_the_file_holds_digests_never_token_values(tmp_path):
    reg = _registry(tmp_path)
    reg.record({digest("bearer-secret-value"): time.time() + DAY}, issuer="one")

    assert "bearer-secret-value" not in Path(reg.path).read_text()


def test_a_revoked_bearer_is_not_live_and_recording_it_again_does_not_undo_that(tmp_path):
    reg = _registry(tmp_path)
    now = time.time()
    reg.record({digest("bearer-1"): now + DAY}, issuer="one", now=now)

    reg.revoke({digest("bearer-1"): None}, now=now)
    reg.record({digest("bearer-1"): now + DAY}, issuer="two", now=now)

    assert reg.lookup(digest("bearer-1"), now=now) is None
    assert reg.is_revoked(digest("bearer-1"))


def test_revoking_an_unregistered_bearer_with_a_known_expiry_keeps_it_out(tmp_path):
    reg = _registry(tmp_path)
    now = time.time()
    reg.revoke({digest("bearer-1"): now + DAY}, now=now)
    reg.record({digest("bearer-1"): now + DAY}, issuer="one", now=now)

    assert reg.lookup(digest("bearer-1"), now=now) is None


def test_an_entry_lasts_as_long_as_the_longest_copy_recorded(tmp_path):
    reg = _registry(tmp_path)
    now = time.time()
    reg.record({digest("bearer-1"): now + 27 * DAY}, issuer="one", now=now)
    reg.record({digest("bearer-1"): now + 365 * DAY}, issuer="two", now=now)

    assert reg.lookup(digest("bearer-1"), now=now + 100 * DAY) is not None


def test_a_revocation_lasts_as_long_as_the_longest_copy_recorded(tmp_path):
    reg = _registry(tmp_path)
    now = time.time()
    reg.record({digest("bearer-1"): now + 27 * DAY}, issuer="one", now=now)
    reg.revoke({digest("bearer-1"): None}, now=now)
    reg.record({digest("bearer-1"): now + 365 * DAY}, issuer="two", now=now)

    reg.record({digest("other"): now + 365 * DAY}, issuer="one", now=now + 28 * DAY)

    assert reg.is_revoked(digest("bearer-1"))


def test_revoking_with_a_later_expiry_extends_the_revocation(tmp_path):
    reg = _registry(tmp_path)
    now = time.time()
    reg.record({digest("bearer-1"): now + 27 * DAY}, issuer="one", now=now)
    reg.revoke({digest("bearer-1"): now + 365 * DAY}, now=now)

    reg.record({digest("other"): now + 365 * DAY}, issuer="one", now=now + 28 * DAY)

    assert reg.is_revoked(digest("bearer-1"))


def test_revoking_an_unregistered_bearer_without_an_expiry_records_nothing(tmp_path):
    reg = _registry(tmp_path)
    reg.revoke({digest("bearer-1"): None})

    assert not reg.is_revoked(digest("bearer-1"))


def test_an_unreadable_registry_accepts_nothing_and_is_not_overwritten(tmp_path):
    reg = _registry(tmp_path)
    reg.record({digest("bearer-1"): time.time() + DAY}, issuer="one")
    with open(reg.path, "w") as fh:
        fh.write("{torn")

    reg.record({digest("bearer-2"): time.time() + DAY}, issuer="one")

    assert reg.lookup(digest("bearer-1")) is None
    assert reg.lookup(digest("bearer-2")) is None
    assert Path(reg.path).read_text() == "{torn"


def test_expired_entries_are_dropped_on_the_next_write(tmp_path):
    reg = _registry(tmp_path)
    now = time.time()
    reg.record({digest("old"): now + 10}, issuer="one", now=now)
    reg.record({digest("new"): now + DAY}, issuer="one", now=now + 20)

    assert list(json.loads(Path(reg.path).read_text())["bearers"]) == [digest("new")]


def test_the_registry_file_is_private(tmp_path):
    reg = _registry(tmp_path)
    reg.record({digest("bearer-1"): time.time() + DAY}, issuer="one")

    assert os.stat(reg.path).st_mode & 0o777 == 0o600


def test_the_registry_reaches_the_disk_before_it_replaces_the_old_file(tmp_path, monkeypatch):
    reg = _registry(tmp_path)
    calls = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(os, "replace", lambda a, b: (calls.append("replace"), real_replace(a, b))[1])

    reg.record({digest("bearer-1"): time.time() + DAY}, issuer="one")

    assert calls == ["fsync", "replace"]

# ── test_oauth_google_registry ──

REDIRECT = "https://oauth-redirect.googleusercontent.com/r/custom-mcp"
GEMINI_REDIRECTS = [
    f"https://{host}.googleusercontent.com/{kind}/user_bound_custom-mcp-host_example"
    for kind in ("r", "a")
    for host in ("oauth-redirect-sandbox", "oauth-redirect-test", "oauth-redirect")
]
RESOURCE = "https://host.example/vg/vecgrep-mcp"
DAY_oauth_google_registry = 86400


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


def _registry_oauth_google_registry(tmp_path):
    return GoogleBearerRegistry(_paths(tmp_path)[1])


def _registered_elsewhere(tmp_path, token, expires_at):
    _registry_oauth_google_registry(tmp_path).record({digest(token): expires_at}, issuer="another-mcp")


# ── issuing and accepting ───────────────────────────────────────────────────

def test_a_bearer_issued_to_gemini_is_recorded_by_digest(tmp_path):
    _, at, _ = _approved(tmp_path)

    entry = _registry_oauth_google_registry(tmp_path).lookup(digest(at.token))
    assert entry is not None
    assert entry["issuer"] == "vecgrep"
    assert at.token not in Path(_paths(tmp_path)[1]).read_text()


def test_a_bearer_another_mcp_registered_is_accepted_under_this_approval(tmp_path):
    store, _, _ = _approved(tmp_path, scopes=("read",))
    _registered_elsewhere(tmp_path, "from-another-mcp", time.time() + 30 * DAY_oauth_google_registry)

    at = store.load_access_token("from-another-mcp")
    assert at is not None
    assert at.client_id == "vg-google"
    assert at.scopes == ["read"]
    assert at.resource == RESOURCE


def test_a_registered_bearer_is_refused_without_an_approval_here(tmp_path):
    store = _store(tmp_path)
    store.save_client(_client())                      # registered, never approved
    _registered_elsewhere(tmp_path, "from-another-mcp", time.time() + 30 * DAY_oauth_google_registry)

    assert store.load_access_token("from-another-mcp") is None


def test_a_registered_bearer_never_outlives_the_approval_here(tmp_path):
    store, _, rt = _approved(tmp_path)
    _registered_elsewhere(tmp_path, "from-another-mcp", time.time() + 900 * DAY_oauth_google_registry)

    loaded = store.load_access_token("from-another-mcp")
    assert loaded is not None and loaded.expires_at is not None and rt.expires_at is not None
    assert loaded.expires_at <= rt.expires_at


def test_a_registered_bearer_is_refused_once_it_expires(tmp_path, monkeypatch):
    store, _, _ = _approved(tmp_path)
    _registered_elsewhere(tmp_path, "from-another-mcp", time.time() + DAY_oauth_google_registry)
    later = time.time() + 2 * DAY_oauth_google_registry
    monkeypatch.setattr("vecgrep.backend.auth.store._now", lambda: later)

    assert store.load_access_token("from-another-mcp") is None


def test_a_copy_that_expired_here_gives_way_to_a_longer_one_recorded_elsewhere(tmp_path, monkeypatch):
    store, _, _ = _approved(tmp_path)
    short = store.issue_access_token("vg-google", ["read"], ttl_s=DAY_oauth_google_registry, resource=RESOURCE)
    _registered_elsewhere(tmp_path, short.token, time.time() + 30 * DAY_oauth_google_registry)
    later = time.time() + 2 * DAY_oauth_google_registry
    monkeypatch.setattr("vecgrep.backend.auth.store._now", lambda: later)

    at = store.load_access_token(short.token)

    assert at is not None and at.expires_at is not None and at.expires_at > later


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

    assert _registry_oauth_google_registry(tmp_path).lookup(digest(at.token)) is not None


# ── revoking ────────────────────────────────────────────────────────────────

def test_a_bearer_revoked_at_another_mcp_stops_loading_here(tmp_path):
    store, at, _ = _approved(tmp_path)

    _registry_oauth_google_registry(tmp_path).revoke({digest(at.token): None})

    assert store.load_access_token(at.token) is None


def test_revoking_the_bearer_here_revokes_it_in_the_registry(tmp_path):
    store, at, _ = _approved(tmp_path)

    store.revoke(at.token)

    assert _registry_oauth_google_registry(tmp_path).is_revoked(digest(at.token))


def test_revoking_a_registered_bearer_this_store_never_held_revokes_it_everywhere(tmp_path):
    store, _, _ = _approved(tmp_path)
    _registered_elsewhere(tmp_path, "from-another-mcp", time.time() + 30 * DAY_oauth_google_registry)

    store.revoke("from-another-mcp")

    assert store.load_access_token("from-another-mcp") is None
    assert _registry_oauth_google_registry(tmp_path).is_revoked(digest("from-another-mcp"))


def test_revoking_the_google_client_revokes_its_bearers_in_the_registry(tmp_path):
    store, at, _ = _approved(tmp_path)

    store.revoke_client("vg-google")

    assert _registry_oauth_google_registry(tmp_path).is_revoked(digest(at.token))


def test_a_replayed_refresh_token_revokes_the_bearer_in_the_registry(tmp_path):
    store, at, rt = _approved(tmp_path)
    assert store.rotate_refresh_token("vg-google", rt.token, ["read"]) is not None

    assert store.load_refresh_token("vg-google", rt.token) is None   # the replay

    assert _registry_oauth_google_registry(tmp_path).is_revoked(digest(at.token))


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
    _registered_elsewhere(tmp_path, "from-another-mcp", time.time() + 30 * DAY_oauth_google_registry)

    assert store.load_access_token("from-another-mcp") is None


def test_a_google_redirect_that_is_not_the_pinned_connector_is_not_google(tmp_path):
    """Any Google Cloud project can register a redirect on Google's hosts; only
    the owner's pinned connector earns the Gemini lifetime and the registry."""
    store = _store(tmp_path)
    store.save_client(_client("other-project", "Google",
                              ["https://oauth-redirect.googleusercontent.com/r/user_bound_custom-mcp-999-host_example"]))
    at, _ = store.issue_token_pair("other-project", ["read"], resource=RESOURCE)
    assert _one_hour_or_less(at)


def test_no_client_is_google_when_no_connector_is_pinned(tmp_path, monkeypatch):
    monkeypatch.delenv("OAUTH_GOOGLE_REDIRECT_URIS")
    store = _store(tmp_path)
    store.save_client(_client())
    at, _ = store.issue_token_pair("vg-google", ["read"], resource=RESOURCE)
    assert _one_hour_or_less(at)
