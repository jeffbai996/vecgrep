"""OAuth token store — the lifecycle WE own (the SDK does HTTP/PKCE/crypto).

Mint, load, expire, and revoke access tokens, refresh tokens, and authorization
codes, using the MCP SDK's data types so the provider can hand them straight to
the SDK. These tests pin the lifecycle invariants that matter for security:
expiry is enforced, revoked tokens stop loading, a code is single-use, and
scopes round-trip.
"""
from __future__ import annotations

import time

import pytest
from mcp.shared.auth import OAuthClientInformationFull

from vecgrep.backend.auth.store import TokenStore


@pytest.fixture
def store():
    return TokenStore()


def _client(client_id: str, issued_at: int) -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id=client_id,
        client_id_issued_at=issued_at,
        redirect_uris=["https://example.test/callback"],
    )


# --- access tokens ---

def test_issue_and_load_access_token(store):
    at = store.issue_access_token(client_id="c1", scopes=["read"], ttl_s=3600)
    loaded = store.load_access_token(at.token)
    assert loaded is not None
    assert loaded.client_id == "c1"
    assert loaded.scopes == ["read"]


def test_expired_access_token_does_not_load(store):
    at = store.issue_access_token(client_id="c1", scopes=["read"], ttl_s=-1)  # already expired
    assert store.load_access_token(at.token) is None


def test_revoked_access_token_does_not_load(store):
    at = store.issue_access_token(client_id="c1", scopes=["read"], ttl_s=3600)
    store.revoke(at.token)
    assert store.load_access_token(at.token) is None


def test_unknown_access_token_is_none(store):
    assert store.load_access_token("never-issued") is None


def test_access_tokens_default_to_one_hour(store):
    """A leaked access token stops working within the hour; the year-long
    approval lives in the refresh token, which rotates it silently."""
    before = time.time()
    at = store.issue_access_token(client_id="c1", scopes=["read"])
    assert 3600 - 5 <= at.expires_at - before <= 3600 + 5


def test_refresh_tokens_default_to_a_year(store):
    before = time.time()
    rt = store.issue_refresh_token(client_id="c1", scopes=["read"])
    assert 365 * 86400 - 5 <= rt.expires_at - before <= 365 * 86400 + 5


def test_loading_clamps_legacy_access_tokens_to_the_current_lifetime(tmp_path):
    """State written while access tokens lived a year loads with every access
    token due within the hour; nothing is extended, and the file is rewritten
    so a restart cannot bring the old lifetime back."""
    import json
    path = tmp_path / "oauth_state.json"
    first = TokenStore(path=str(path))
    at = first.issue_access_token(client_id="c1", scopes=["read"], ttl_s=300 * 86400)
    rt = first.issue_refresh_token(client_id="c1", scopes=["read"], ttl_s=400 * 86400)
    short = first.issue_access_token(client_id="c1", scopes=["read"], ttl_s=60)
    before = time.time()
    second = TokenStore(path=str(path))
    loaded = second.load_access_token(at.token)
    assert loaded is not None and loaded.expires_at <= before + 3600 + 5
    assert second.load_refresh_token("c1", rt.token).expires_at <= before + 365 * 86400 + 5
    assert second.load_access_token(short.token).expires_at <= before + 60 + 5, "never extended"
    on_disk = json.loads(path.read_text())
    assert on_disk["access"][at.token]["expires_at"] <= before + 3600 + 5


# --- refresh tokens ---

def test_issue_and_load_refresh_token(store):
    rt = store.issue_refresh_token(client_id="c1", scopes=["read", "propose"])
    loaded = store.load_refresh_token("c1", rt.token)
    assert loaded is not None
    assert set(loaded.scopes) == {"read", "propose"}


def test_refresh_token_wrong_client_rejected(store):
    rt = store.issue_refresh_token(client_id="c1", scopes=["read"])
    assert store.load_refresh_token("other-client", rt.token) is None


def test_revoked_refresh_token_does_not_load(store):
    rt = store.issue_refresh_token(client_id="c1", scopes=["read"])
    store.revoke(rt.token)
    assert store.load_refresh_token("c1", rt.token) is None


# --- authorization codes (single-use) ---

def test_issue_and_consume_authorization_code(store):
    code = store.issue_authorization_code(
        client_id="c1", scopes=["read"], code_challenge="abc",
        redirect_uri="https://claude.ai/cb",
    )
    loaded = store.load_authorization_code("c1", code.code)
    assert loaded is not None
    assert loaded.code_challenge == "abc"


def test_authorization_code_is_single_use(store):
    code = store.issue_authorization_code(
        client_id="c1", scopes=["read"], code_challenge="abc",
        redirect_uri="https://claude.ai/cb",
    )
    assert store.consume_authorization_code("c1", code.code) is not None
    # second consume fails — a replayed code can't mint a second token
    assert store.consume_authorization_code("c1", code.code) is None


def test_expired_authorization_code_does_not_load(store):
    code = store.issue_authorization_code(
        client_id="c1", scopes=["read"], code_challenge="abc",
        redirect_uri="https://claude.ai/cb", ttl_s=-1,
    )
    assert store.load_authorization_code("c1", code.code) is None


# --- scope enforcement helper ---

def test_token_has_scope(store):
    at = store.issue_access_token(client_id="c1", scopes=["read"], ttl_s=3600)
    assert store.token_has_scope(at.token, "read") is True
    assert store.token_has_scope(at.token, "propose") is False


# --- dynamic-client retention bounds ---

def test_unapproved_client_count_is_bounded_without_evicting_authorized_clients(monkeypatch):
    monkeypatch.setattr("vecgrep.backend.auth.store._now", lambda: 100.0)
    store = TokenStore(max_unapproved_clients=2, unapproved_client_ttl_s=10_000)
    store.save_client(_client("authorized", 1))
    store.issue_access_token("authorized", ["read"])

    store.save_client(_client("unapproved-old", 2))
    store.save_client(_client("unapproved-new", 3))
    store.save_client(_client("unapproved-newest", 4))

    assert store.get_client("authorized") is not None
    assert store.get_client("unapproved-old") is None
    assert set(store.clients()) == {
        "authorized", "unapproved-new", "unapproved-newest",
    }


def test_expired_unapproved_clients_are_pruned(monkeypatch):
    monkeypatch.setattr("vecgrep.backend.auth.store._now", lambda: 1_000.0)
    store = TokenStore(max_unapproved_clients=5, unapproved_client_ttl_s=60)
    store.save_client(_client("stale", 900))
    store.save_client(_client("fresh", 1_000))

    assert store.get_client("stale") is None
    assert store.get_client("fresh") is not None


# ----- admin surface (inventory panel) -----

def test_counts_reflect_live_tokens(store):
    store.issue_access_token("c1", ["read"])
    store.issue_access_token("c1", ["read"])
    store.issue_refresh_token("c1", ["read"])
    store.issue_access_token("c2", ["read"])
    c = store.counts()
    assert c["access"] == 3
    assert c["refresh"] == 1
    assert c["by_client"]["c1"]["access"] == 2


def test_revoke_client_kills_all_its_tokens(store):
    a1 = store.issue_access_token("c1", ["read"])
    r1 = store.issue_refresh_token("c1", ["read"])
    a2 = store.issue_access_token("c2", ["read"])
    assert store.revoke_client("c1") == 2
    assert store.load_access_token(a1.token) is None
    assert store.load_refresh_token("c1", r1.token) is None
    assert store.load_access_token(a2.token) is not None


# ----- refresh-token families (replay of a rotated token revokes the approval) -----

def test_rotated_refresh_token_replacement_works(store):
    at, rt = store.issue_token_pair("c1", ["read", "propose"])
    pair = store.rotate_refresh_token("c1", rt.token, ["read"])
    assert pair is not None
    new_at, new_rt = pair
    assert store.load_access_token(new_at.token).scopes == ["read"]
    assert store.load_refresh_token("c1", new_rt.token).scopes == ["read"]
    assert store.load_access_token(at.token) is not None, "rotation leaves access tokens alone"
    second = store.rotate_refresh_token("c1", new_rt.token, ["read"])
    assert second is not None, "the replacement rotates in turn"
    # presenting the retired token again is a replay (pinned below)
    assert store.load_refresh_token("c1", rt.token) is None


def test_replaying_a_rotated_refresh_token_revokes_its_family(store):
    at, rt = store.issue_token_pair("c1", ["read"])
    new_at, new_rt = store.rotate_refresh_token("c1", rt.token, ["read"])
    other_at, other_rt = store.issue_token_pair("c1", ["read"])

    assert store.rotate_refresh_token("c1", rt.token, ["read"]) is None
    assert store.load_access_token(at.token) is None
    assert store.load_access_token(new_at.token) is None
    assert store.load_refresh_token("c1", new_rt.token) is None
    # a second approval of the same client is a separate family
    assert store.load_access_token(other_at.token) is not None
    assert store.load_refresh_token("c1", other_rt.token) is not None


def test_loading_a_rotated_refresh_token_revokes_its_family(store):
    """The SDK loads the presented token before it exchanges it, so the replay
    signal is acted on at load time too."""
    _at, rt = store.issue_token_pair("c1", ["read"])
    new_at, new_rt = store.rotate_refresh_token("c1", rt.token, ["read"])
    assert store.load_refresh_token("c1", rt.token) is None
    assert store.load_access_token(new_at.token) is None
    assert store.load_refresh_token("c1", new_rt.token) is None


def test_replay_by_another_client_revokes_nothing(store):
    _at, rt = store.issue_token_pair("c1", ["read"])
    new_at, new_rt = store.rotate_refresh_token("c1", rt.token, ["read"])
    assert store.rotate_refresh_token("c2", rt.token, ["read"]) is None
    assert store.load_refresh_token("c2", rt.token) is None
    assert store.load_access_token(new_at.token) is not None
    assert store.load_refresh_token("c1", new_rt.token) is not None


def test_rotation_cannot_widen_scopes(store):
    _at, rt = store.issue_token_pair("c1", ["read"])
    assert store.rotate_refresh_token("c1", rt.token, ["read", "propose"]) is None
    assert store.load_refresh_token("c1", rt.token) is not None, "a refused widening consumes nothing"


def test_revoking_a_refresh_token_revokes_its_family_access_tokens(store):
    at, rt = store.issue_token_pair("c1", ["read"])
    new_at, new_rt = store.rotate_refresh_token("c1", rt.token, ["read"])
    other_at, other_rt = store.issue_token_pair("c1", ["read"])
    store.revoke(new_rt.token)
    assert store.load_refresh_token("c1", new_rt.token) is None
    assert store.load_access_token(at.token) is None
    assert store.load_access_token(new_at.token) is None
    assert store.load_access_token(other_at.token) is not None
    assert store.load_refresh_token("c1", other_rt.token) is not None


def test_revoking_an_access_token_leaves_its_family(store):
    at, rt = store.issue_token_pair("c1", ["read"])
    store.revoke(at.token)
    assert store.load_access_token(at.token) is None
    assert store.load_refresh_token("c1", rt.token) is not None


def test_access_token_loads_are_unchanged(store):
    at = store.issue_access_token("c1", ["read", "propose"], ttl_s=3600, resource="https://r.test/mcp")
    _pair_at, rt = store.issue_token_pair("c1", ["read"])
    store.rotate_refresh_token("c1", rt.token, ["read"])
    loaded = store.load_access_token(at.token)
    assert loaded is not None
    assert (loaded.token, loaded.client_id, loaded.scopes, loaded.expires_at, loaded.resource) == (
        at.token, "c1", ["read", "propose"], at.expires_at, "https://r.test/mcp")
    assert store.token_has_scope(at.token, "propose") is True


def test_replay_is_detected_after_a_restart(tmp_path):
    path = str(tmp_path / "oauth_state.json")
    first = TokenStore(path=path)
    _at, rt = first.issue_token_pair("c1", ["read"])
    new_at, new_rt = first.rotate_refresh_token("c1", rt.token, ["read"])
    second = TokenStore(path=path)
    assert second.load_refresh_token("c1", new_rt.token) is not None
    assert second.rotate_refresh_token("c1", rt.token, ["read"]) is None
    assert second.load_access_token(new_at.token) is None
    assert second.load_refresh_token("c1", new_rt.token) is None
    third = TokenStore(path=path)
    assert third.load_refresh_token("c1", new_rt.token) is None, "the revocation is persisted"


def _legacy_state(path, now):
    """The state shape written before families existed."""
    import json

    def tok(token, client, ttl):
        return {"token": token, "client_id": client, "scopes": ["read"],
                "expires_at": int(now + ttl), "resource": None}

    state = {
        "access": {"vgat_legacy-a": tok("vgat_legacy-a", "c1", 1800)},
        "refresh": {
            "vgrt_legacy-1": tok("vgrt_legacy-1", "c1", 400 * 86400),
            "vgrt_legacy-2": tok("vgrt_legacy-2", "c2", 30 * 86400),
        },
        "clients": {},
    }
    path.write_text(json.dumps(state))


def test_legacy_state_without_families_loads_rotates_and_is_rewritten(tmp_path):
    import json
    path = tmp_path / "oauth_state.json"
    before = time.time()
    _legacy_state(path, before)

    store = TokenStore(path=str(path))
    assert store.load_access_token("vgat_legacy-a") is not None
    legacy = store.load_refresh_token("c1", "vgrt_legacy-1")
    assert legacy is not None
    assert legacy.expires_at <= before + 365 * 86400 + 5, "the lifetime clamp still applies"
    assert store.load_refresh_token("c2", "vgrt_legacy-2") is not None

    on_disk = json.loads(path.read_text())
    families = on_disk["families"]
    assert len(families) == 2, "each legacy refresh token gets its own family"
    assert {r["family"] for r in on_disk["refresh"].values()} == set(families)
    assert on_disk["access"]["vgat_legacy-a"]["family"] == on_disk["refresh"]["vgrt_legacy-1"]["family"]

    new_at, new_rt = store.rotate_refresh_token("c1", "vgrt_legacy-1", ["read"])
    assert store.load_refresh_token("c1", new_rt.token) is not None

    restarted = TokenStore(path=str(path))
    assert restarted.rotate_refresh_token("c1", "vgrt_legacy-1", ["read"]) is None
    assert restarted.load_access_token(new_at.token) is None
    assert restarted.load_refresh_token("c1", new_rt.token) is None
    assert restarted.load_access_token("vgat_legacy-a") is None
    assert restarted.load_refresh_token("c2", "vgrt_legacy-2") is not None, "other families untouched"


def test_legacy_load_keeps_every_live_token(tmp_path):
    path = tmp_path / "oauth_state.json"
    _legacy_state(path, time.time())
    store = TokenStore(path=str(path))
    counts = store.counts()
    assert (counts["access"], counts["refresh"], counts["families"]) == (1, 2, 2)
    again = TokenStore(path=str(path)).counts()
    assert (again["access"], again["refresh"], again["families"]) == (1, 2, 2)


def test_revoke_client_drops_its_families(store):
    store.issue_token_pair("c1", ["read"])
    store.issue_token_pair("c2", ["read"])
    store.revoke_client("c1")
    assert store.counts()["families"] == 1


def test_google_connector_access_tokens_match_refresh_lifetime(tmp_path):
    """Google's Gemini web MCP proxy caches bearer tokens across worker pools
    without refreshing 1-hour access tokens on chat turns."""
    path = tmp_path / "oauth_state.json"
    store = TokenStore(path=str(path))
    google = OAuthClientInformationFull(
        client_id="google-1",
        client_name="Google",
        redirect_uris=["https://oauth-redirect.googleusercontent.com/r/custom-mcp"],
    )
    store.save_client(google)
    before = time.time()
    at, rt = store.issue_token_pair("google-1", ["read"])
    assert at.expires_at >= before + 365 * 86400 - 5
    rotated_at, _ = store.rotate_refresh_token("google-1", rt.token, ["read"])
    assert rotated_at.expires_at >= before + 365 * 86400 - 5
    reloaded = TokenStore(path=str(path))
    assert reloaded.load_access_token(rotated_at.token).expires_at >= before + 365 * 86400 - 5


def test_load_access_token_reloads_when_state_file_changes_on_disk(tmp_path):
    path = tmp_path / "oauth_state.json"
    writer = TokenStore(path=str(path))
    reader = TokenStore(path=str(path))
    at = writer.issue_access_token("c1", ["read"])
    assert reader.load_access_token(at.token) is not None


def test_google_connector_federates_tokens_across_sibling_stores_and_reuses_live_sibling_token(tmp_path):
    import hashlib
    import json

    vg_path = tmp_path / "vecgrep_oauth.json"
    sib_a_path = tmp_path / "sibling_a_oauth.json"
    sib_b_path = tmp_path / "sibling_b_oauth.json"
    now = time.time()

    sib_a_path.write_text(json.dumps({
        "clients": {
            "sib-google": {
                "client_id": "sib-google",
                "client_name": "Google",
                "redirect_uris": ["https://oauth-redirect.googleusercontent.com/r/custom-mcp"],
            }
        },
        "access": {
            "sib-1": {
                "token": "sib-1",
                "client_id": "sib-google",
                "scopes": ["read"],
                "expires_at": int(now + 30 * 86400),
                "resource": None,
            }
        },
        "refresh": {},
    }))
    sib_b_path.write_text(json.dumps({
        "version": 1,
        "clients": {
            "dcr_google": {
                "client_id": "dcr_google",
                "client_name": "Google",
                "redirect_uris": ["https://oauth-redirect.googleusercontent.com/r/custom-mcp"],
                "created_at": now,
                "registration_expires_at": None,
                "revoked_at": None,
            }
        },
        "grants": {
            "g1": {
                "grant_id": "g1",
                "client_id": "dcr_google",
                "principal_id": "p1",
                "resource": "https://host.example/notes-mcp/mcp",
                "scopes": ["offline_access", "notes.read", "notes.write"],
                "created_at": now,
                "last_used_at": None,
                "revoked_at": None,
            }
        },
        "families": {
            "f1": {
                "family_id": "f1",
                "grant_id": "g1",
                "expires_at": now + 365 * 86400,
                "revoked_at": None,
            }
        },
        "access_tokens": {},
        "refresh_tokens": {},
    }))

    store = TokenStore(
        path=str(vg_path),
        federated_state_paths=[str(sib_a_path), str(sib_b_path)],
    )
    google = OAuthClientInformationFull(
        client_id="vg-google",
        client_name="Google",
        redirect_uris=["https://oauth-redirect.googleusercontent.com/r/custom-mcp"],
    )
    store.save_client(google)
    at, _ = store.issue_token_pair(
        "vg-google",
        ["read", "propose"],
        resource="https://host.example/vg/vecgrep-mcp",
    )
    assert at.token == "sib-1"
    assert store.load_access_token("sib-1") is not None

    sib_b_after = json.loads(sib_b_path.read_text())
    digest = hashlib.sha256(b"sib-1").hexdigest()
    assert digest in sib_b_after["access_tokens"]
    assert sib_b_after["access_tokens"][digest]["resource"] == "https://host.example/notes-mcp/mcp"
    assert set(sib_b_after["access_tokens"][digest]["scopes"]) >= {"notes.read", "notes.write"}


