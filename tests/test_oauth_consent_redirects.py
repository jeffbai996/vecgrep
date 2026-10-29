"""Consent must describe and approve the registered authorization destination."""

from __future__ import annotations

import base64
import hashlib
import html
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

from tests import test_oauth_root_discovery as flow
from tests.oauth_flow import FormFields as Inputs

client = flow.oauth_client
APPROVAL = flow.TEST_APPROVAL
from vecgrep.backend.auth.approval import APPROVAL_COOKIE

PREFIX = ""


def local(value: str) -> str:
    return value


def register(client, redirects: list[str]) -> str:
    response = client.post(
        "/register",
        json={
            "client_name": "<Demo & client>",
            "redirect_uris": redirects,
            "token_endpoint_auth_method": "none",
            "scope": "read",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["client_id"]


def target(cid: str, redirect: str | None = None) -> str:
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(b"v" * 64).digest())
        .rstrip(b"=")
        .decode()
    )
    params = {
        "client_id": cid,
        "response_type": "code",
        "scope": "read",
        "state": "state+&value",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    if redirect is not None:
        params["redirect_uri"] = redirect
    return PREFIX + "/authorize?" + urlencode(params)


def page(client, request: str):
    response = client.get("/oauth/unlock", params={"next": request})
    assert response.status_code == 200, response.text
    return response, Inputs(response.text).values


def approve(client, request: str):
    _, fields = page(client, request)
    return client.post(
        "/oauth/unlock", data={**fields, "token": APPROVAL}, follow_redirects=False
    )


@pytest.mark.parametrize(
    "redirect",
    [
        "https://client.example/callback?x=1&y=%3Cvalue%3E",
        "http://127.0.0.1:53001/callback",
        "http://[::1]:53001/callback",
        "http://localhost:53001/callback",
        "com.example.app:/oauth2redirect",
    ],
)
def test_omitted_single_registered_destination_is_visible_and_redeemable(
    client, redirect: str
) -> None:
    cid = register(client, [redirect])
    request = target(cid)
    response, fields = page(client, request)
    assert html.escape(redirect) in response.text
    assert "&lt;Demo &amp; client&gt;" in response.text
    approved = client.post(
        "/oauth/unlock", data={**fields, "token": APPROVAL}, follow_redirects=False
    )
    assert approved.status_code == 303
    continuation = client.get(local(approved.headers["location"]))
    assert continuation.status_code == 200
    authorized = client.get(local(request), follow_redirects=False)
    assert authorized.status_code == 302
    result = urlsplit(authorized.headers["location"])
    assert result.scheme == urlsplit(redirect).scheme
    assert result.netloc == urlsplit(redirect).netloc
    assert result.path == urlsplit(redirect).path
    values = parse_qs(result.query)
    assert values["state"] == ["state+&value"]
    token = client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "client_id": cid,
            "code": values["code"][0],
            "code_verifier": "v" * 64,
        },
    )
    assert token.status_code == 200, token.text


def test_explicit_registered_destination_preserves_pkce_and_exact_match(client) -> None:
    redirect = "https://client.example/callback"
    cid = register(client, [redirect, "https://other.example/callback"])
    request = target(cid, redirect)
    assert approve(client, request).status_code == 303
    authorized = client.get(local(request), follow_redirects=False)
    values = parse_qs(urlsplit(authorized.headers["location"]).query)
    fields = {
        "grant_type": "authorization_code",
        "client_id": cid,
        "code": values["code"][0],
        "redirect_uri": redirect,
    }
    bad = client.post("/token", data={**fields, "code_verifier": "wrong" * 10})
    assert bad.status_code == 400
    good = client.post("/token", data={**fields, "code_verifier": "v" * 64})
    assert good.status_code == 200
    assert (
        client.post("/token", data={**fields, "code_verifier": "v" * 64}).status_code
        == 400
    )


def test_missing_redirect_with_multiple_registrations_is_rejected(client) -> None:
    cid = register(client, ["https://client.example/a", "https://client.example/b"])
    response = client.get("/oauth/unlock", params={"next": target(cid)})
    assert response.status_code == 400
    assert 'name="token"' not in response.text


@pytest.mark.parametrize(
    "suffix",
    [
        "&redirect_uri=https://evil.example/cb",
        "&client_id=other",
        "&scope=propose",
        "&state=other",
        "&code_challenge=other",
    ],
)
def test_duplicate_authorization_parameters_are_rejected_before_consent(
    client, suffix: str
) -> None:
    cid = register(client, ["https://client.example/callback"])
    request = target(cid, "https://client.example/callback") + suffix
    assert client.get("/oauth/unlock", params={"next": request}).status_code == 400
    assert client.get(local(request), follow_redirects=False).status_code == 400


@pytest.mark.parametrize(
    "redirect", ["https://evil.example/cb", "", "https://client.example/other"]
)
def test_unregistered_or_empty_destination_is_rejected(client, redirect: str) -> None:
    cid = register(client, ["https://client.example/callback"])
    request = target(cid, redirect)
    assert client.get("/oauth/unlock", params={"next": request}).status_code == 400


@pytest.mark.parametrize(
    "redirect",
    [
        "http://remote.example/cb",
        "javascript:alert(1)",
        "https://user:pass@client.example/cb",
        "https://client.example/cb#fragment",
        "https://client.example:bad/cb",
    ],
)
def test_unsafe_registration_is_rejected(client, redirect: str) -> None:
    response = client.post(
        "/register",
        json={"redirect_uris": [redirect], "token_endpoint_auth_method": "none"},
    )
    assert response.status_code == 400


def test_changed_client_or_redirect_cannot_reuse_consent_form(client) -> None:
    cid = register(client, ["https://client.example/a", "https://client.example/b"])
    other = register(client, ["https://other.example/cb"])
    _, fields = page(client, target(cid, "https://client.example/a"))
    for changed in (target(cid, "https://client.example/b"), target(other)):
        response = client.post(
            "/oauth/unlock",
            data={**fields, "next": changed, "token": APPROVAL},
            follow_redirects=False,
        )
        assert response.status_code == 400
        assert APPROVAL_COOKIE not in response.cookies


def test_owner_cookie_and_continuation_cannot_authorize_another_destination(
    client,
) -> None:
    cid = register(client, ["https://client.example/a", "https://client.example/b"])
    first, second = (
        target(cid, "https://client.example/a"),
        target(cid, "https://client.example/b"),
    )
    assert approve(client, first).status_code == 303
    changed = client.get("/oauth/continue", params={"next": second})
    assert changed.status_code == 400
    response = client.get(local(second), follow_redirects=False)
    assert response.status_code == 303
    assert "/oauth/unlock?" in response.headers["location"]


def test_post_authorization_uses_the_same_consent_flow(client) -> None:
    cid = register(client, ["https://client.example/callback"])
    request = target(cid)
    response = client.post(
        "/authorize",
        data={k: v[0] for k, v in parse_qs(urlsplit(request).query).items()},
        follow_redirects=False,
    )
    assert response.status_code == 303
    next_response = client.get(
        local(response.headers["location"]), follow_redirects=False
    )
    assert next_response.status_code == 303
    assert "/oauth/unlock?" in next_response.headers["location"]


def test_sdk_registration_normalization_is_shown_and_used_consistently(client) -> None:
    reg = client.post(
        "/register",
        json={
            "redirect_uris": ["HTTPS://CLIENT.EXAMPLE:443/cb"],
            "token_endpoint_auth_method": "none",
        },
    )
    assert reg.status_code == 201
    uri = reg.json()["redirect_uris"][0]
    assert uri == "https://client.example/cb"
    request = target(reg.json()["client_id"], "HTTPS://CLIENT.EXAMPLE:443/cb")
    response, _ = page(client, request)
    assert uri in response.text
    assert approve(client, request).status_code == 303
    authorized = client.get(local(request), follow_redirects=False)
    assert authorized.headers["location"].startswith(uri + "?")


@pytest.mark.parametrize("changed", ["state", "code_challenge", "scope", "resource"])
def test_consent_binds_all_authorization_parameters(client, changed: str) -> None:
    cid = register(client, ["https://client.example/callback"])
    request = target(cid)
    _, fields = page(client, request)
    params = {k: v[0] for k, v in parse_qs(urlsplit(request).query).items()}
    params[changed] = "changed"
    fields["next"] = PREFIX + "/authorize?" + urlencode(params)
    response = client.post(
        "/oauth/unlock", data={**fields, "token": APPROVAL}, follow_redirects=False
    )
    assert response.status_code == 400


def test_duplicate_outer_next_and_duplicate_form_fields_are_rejected(client) -> None:
    cid = register(client, ["https://client.example/callback"])
    request = target(cid)
    _, fields = page(client, request)
    assert (
        client.get(
            "/oauth/unlock", params=[("next", request), ("next", request)]
        ).status_code
        == 400
    )
    response = client.post(
        "/oauth/unlock",
        content=urlencode({**fields, "token": APPROVAL}) + "&next=" + request,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        follow_redirects=False,
    )
    assert response.status_code == 400


def test_authorize_post_rejects_query_body_client_conflict(client) -> None:
    cid = register(client, ["https://client.example/callback"])
    params = {k: v[0] for k, v in parse_qs(urlsplit(target(cid)).query).items()}
    response = client.post(
        "/authorize?client_id=other", data=params, follow_redirects=False
    )
    assert response.status_code == 400


@pytest.mark.parametrize(
    "path", ["/authorize?client_id=unknown", "/authorize", "/authorize?client_id="]
)
def test_unknown_or_missing_clients_do_not_get_a_consent_form(
    client, path: str
) -> None:
    assert client.get("/oauth/unlock", params={"next": path}).status_code == 400


def test_non_ascii_form_proof_is_rejected_safely(client) -> None:
    cid = register(client, ["https://client.example/callback"])
    _, fields = page(client, target(cid))
    response = client.post(
        "/oauth/unlock", data={**fields, "consent": "無效", "token": APPROVAL}
    )
    assert response.status_code == 400


def test_remembered_owner_can_approve_a_new_request_without_reentering_code(
    client,
) -> None:
    cid = register(client, ["https://client.example/a", "https://client.example/b"])
    assert approve(client, target(cid, "https://client.example/a")).status_code == 303
    response, fields = page(client, target(cid, "https://client.example/b"))
    assert 'name="token"' not in response.text
    approved = client.post("/oauth/unlock", data=fields, follow_redirects=False)
    assert approved.status_code == 303
    assert client.get(local(fields["next"]), follow_redirects=False).status_code == 302


def test_two_open_consent_forms_remain_bound_and_can_complete_in_either_order(
    client,
) -> None:
    cid = register(client, ["https://client.example/a", "https://client.example/b"])
    first, second = (
        target(cid, "https://client.example/a"),
        target(cid, "https://client.example/b"),
    )
    _, fields_a = page(client, first)
    _, fields_b = page(client, second)
    for fields in (fields_b, fields_a):
        approved = client.post(
            "/oauth/unlock", data={**fields, "token": APPROVAL}, follow_redirects=False
        )
        assert approved.status_code == 303
    for request in (first, second):
        assert client.get(local(request), follow_redirects=False).status_code == 302


def test_base_install_can_import_approval_without_optional_mcp(monkeypatch) -> None:
    import builtins
    import importlib.util
    from pathlib import Path

    from vecgrep.backend.auth import redirects

    original_import = builtins.__import__

    def without_mcp(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ModuleNotFoundError("optional mcp is unavailable")
        return original_import(name, *args, **kwargs)

    # Execute fresh module namespaces; cached imports would hide a new hard
    # dependency in an otherwise supported base installation.
    monkeypatch.setattr(builtins, "__import__", without_mcp)
    for name, path in (
        ("vecgrep.backend.auth._base_redirects", Path(redirects.__file__)),
        (
            "vecgrep.backend.auth._base_approval",
            Path(redirects.__file__).with_name("approval.py"),
        ),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


def provider():
    from vecgrep.mcp.server import _shared_provider

    return _shared_provider()


def test_registration_change_invalidates_a_visible_form_and_a_previous_approval(
    client,
) -> None:
    from pydantic import AnyUrl

    cid = register(client, ["https://client.example/a"])
    request = target(cid)
    _, fields = page(client, request)
    info = provider().store.get_client(cid)
    info.redirect_uris = [AnyUrl("https://client.example/b")]
    provider().store.save_client(info)
    response = client.post(
        "/oauth/unlock", data={**fields, "token": APPROVAL}, follow_redirects=False
    )
    assert response.status_code == 400
    assert approve(client, request).status_code == 303
    info.redirect_uris = [AnyUrl("https://client.example/c")]
    provider().store.save_client(info)
    response = client.get(local(request), follow_redirects=False)
    assert response.status_code == 303
    assert "/oauth/unlock?" in response.headers["location"]


def test_existing_unsafe_registration_is_rejected_before_consent_or_issuance(
    client,
) -> None:
    from mcp.shared.auth import OAuthClientInformationFull

    provider().store.save_client(
        OAuthClientInformationFull(
            client_id="legacy", redirect_uris=["http://remote.example/callback"]
        )
    )
    request = target("legacy")
    assert client.get("/oauth/unlock", params={"next": request}).status_code == 400
    assert client.get(local(request), follow_redirects=False).status_code == 400


def test_remembered_owner_cannot_submit_a_new_destination_without_its_form_proof(
    client,
) -> None:
    cid = register(client, ["https://client.example/a", "https://client.example/b"])
    assert approve(client, target(cid, "https://client.example/a")).status_code == 303
    response = client.post(
        "/oauth/unlock",
        data={"next": target(cid, "https://client.example/b")},
        follow_redirects=False,
    )
    assert response.status_code == 400


@pytest.mark.parametrize("redirects", [None, []])
def test_registration_without_a_destination_is_rejected(client, redirects) -> None:
    response = client.post(
        "/register",
        json={"redirect_uris": redirects, "token_endpoint_auth_method": "none"},
    )
    assert response.status_code == 400


def test_mounted_authorize_alias_requires_its_own_matching_consent(client) -> None:
    cid = register(client, ["https://client.example/callback"])
    request = target(cid).replace("/authorize?", "/mcp/authorize?", 1)
    response = client.get(request, follow_redirects=False)
    assert response.status_code == 303
    assert approve(client, request).status_code == 303
    response = client.get(request, follow_redirects=False)
    assert response.status_code == 302
    assert "code" in parse_qs(urlsplit(response.headers["location"]).query)


def test_authorization_response_preserves_the_registered_callback_query_exactly(
    client,
) -> None:
    redirect = (
        "https://client.example/callback?tenant=&value=one&value=two&path=%2f%20%26"
    )
    cid = register(client, [redirect])
    request = target(cid, redirect)
    assert approve(client, request).status_code == 303
    response = client.get(local(request), follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"].startswith(redirect + "&code=")
    values = parse_qs(
        urlsplit(response.headers["location"]).query, keep_blank_values=True
    )
    assert values["tenant"] == [""]
    assert values["value"] == ["one", "two"]
    assert values["state"] == ["state+&value"]


@pytest.mark.parametrize(
    "redirect,separator",
    [
        ("https://client.example/callback&", "?"),
        ("https://client.example/callback?", ""),
        ("https://client.example/callback?tenant=&", ""),
    ],
)
def test_authorization_response_retains_empty_query_and_path_delimiters(
    client, redirect, separator
) -> None:
    cid = register(client, [redirect])
    request = target(cid, redirect)
    assert approve(client, request).status_code == 303
    response = client.get(local(request), follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"].startswith(redirect + separator + "code=")
