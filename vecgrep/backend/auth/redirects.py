"""Resolve and bind the OAuth destination once at every consent boundary."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import re
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode, urlsplit

from pydantic import AnyUrl, ValidationError
from starlette.exceptions import HTTPException
from starlette.types import Receive, Scope

if TYPE_CHECKING:
    from mcp.server.auth.provider import OAuthAuthorizationServerProvider
    from mcp.shared.auth import OAuthClientInformationFull


def invalid_request() -> HTTPException:
    return HTTPException(
        400,
        "Invalid authorization request. Please restart the connection.",
        headers={"Cache-Control": "no-store"},
    )


def validate_redirect_uri(value: str) -> str:
    """Keep SDK URL normalization, but exclude unsafe callback forms.

    Native apps retain reverse-domain private-use schemes and exact registered
    loopback ports. No DNS resolution or new dynamic-port matching is involved.
    """
    try:
        if not value or any(ord(c) < 33 or ord(c) == 127 or c == "\\" for c in value):
            raise ValueError("invalid callback")
        parsed = urlsplit(value)
        _ = parsed.port  # Force validation of a present port.
        if (
            parsed.fragment
            or "#" in value
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("invalid callback")
        if parsed.scheme in {"https", "http"}:
            if not parsed.hostname or any(c in parsed.netloc for c in "*;'\"<>%"):
                raise ValueError("invalid callback")
            if parsed.scheme == "http":
                host = parsed.hostname.lower()
                if host != "localhost" and not ipaddress.ip_address(host).is_loopback:
                    raise ValueError("HTTP callback must be loopback")
        elif (
            not re.fullmatch(r"[a-z][a-z0-9+.-]*\.[a-z0-9+.-]+", parsed.scheme)
            or parsed.netloc
            or not parsed.path.startswith("/")
            or parsed.path.startswith("//")
        ):
            raise ValueError("invalid native callback")
        return str(AnyUrl(value))
    except (ValueError, TypeError, ValidationError) as exc:
        raise ValueError("Invalid redirect URI") from exc


def resolve_redirect(client: OAuthClientInformationFull, value: str | None) -> str:
    from mcp.shared.auth import InvalidRedirectUriError

    try:
        requested = AnyUrl(validate_redirect_uri(value)) if value is not None else None
        resolved = client.validate_redirect_uri(requested)
        return validate_redirect_uri(str(resolved))
    except (ValueError, InvalidRedirectUriError) as exc:
        raise ValueError("Invalid registered redirect URI") from exc


def authorization_query(target: str) -> dict[str, str]:
    try:
        pairs = parse_qsl(
            urlsplit(target).query,
            keep_blank_values=True,
            encoding="utf-8",
            errors="strict",
        )
        values: dict[str, str] = {}
        for key, value in pairs:
            # RFC 6749 parameters cannot appear more than once. Reject instead
            # of letting consent and the SDK choose different occurrences.
            if key in values:
                raise ValueError("duplicate parameter")
            values[key] = value
        if not values.get("client_id"):
            raise ValueError("missing client")
        return values
    except (ValueError, UnicodeError) as exc:
        raise invalid_request() from exc


async def resolve_authorization(
    target: str, provider: OAuthAuthorizationServerProvider
) -> dict:
    values = authorization_query(target)
    client = await provider.get_client(values["client_id"])
    if client is None:
        raise invalid_request()
    try:
        redirect = resolve_redirect(client, values.get("redirect_uri"))
    except ValueError as exc:
        raise invalid_request() from exc
    return {
        "client": client.client_name or "",
        "redirect_uri": redirect,
        "query": values,
    }


def consent_proof(secret: str, target: str, redirect: str, nonce: str = "") -> str:
    # Include the effective URI even when omitted in the request, so a changed
    # registration cannot reuse an old form or request approval.
    message = json.dumps(
        ["oauth-consent-v1", target, redirect, nonce], ensure_ascii=True
    ).encode()
    return hmac.new(secret.strip().encode(), message, hashlib.sha256).hexdigest()


def check_consent(
    secret: str, target: str, redirect: str, cookie: str, proof: str | None = None
) -> bool:
    if not secret or not cookie:
        return False
    expected = consent_proof(
        secret, target, redirect, cookie if proof is not None else ""
    )
    candidates = [proof] if proof is not None else cookie.split(".")
    return any(
        hmac.compare_digest(value.encode(), expected.encode())
        for value in candidates[:8]
    )


def remember_consent(existing: str, proof: str) -> str:
    """Keep a bounded set of approvals so parallel browser tabs can finish."""
    previous = [
        value
        for value in existing.split(".")
        if re.fullmatch(r"[0-9a-f]{64}", value) and value != proof
    ]
    return ".".join([proof, *previous[:7]])


def unique_form(body: bytes) -> dict[str, str]:
    try:
        pairs = parse_qsl(body.decode("utf-8"), keep_blank_values=True, errors="strict")
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate form parameter")
            result[key] = value
        return result
    except (ValueError, UnicodeError) as exc:
        raise invalid_request() from exc


async def post_authorize_target(scope: Scope, receive: Receive, path: str) -> str:
    from starlette.datastructures import Headers

    if (
        Headers(scope=scope).get("content-type", "").split(";")[0]
        != "application/x-www-form-urlencoded"
    ):
        raise invalid_request()
    body = bytearray()
    while True:
        chunk = await receive()
        if chunk["type"] != "http.request":
            raise invalid_request()
        body.extend(chunk.get("body", b""))
        if len(body) > 8192:
            raise invalid_request()
        if not chunk.get("more_body"):
            break
    # Merge through the same duplicate check; query/form conflicts never select
    # different clients or callbacks at the two ends of consent.
    query = scope.get("query_string", b"").decode("ascii")
    target = (
        path
        + "?"
        + query
        + ("&" if query else "")
        + urlencode(unique_form(bytes(body)))
    )
    authorization_query(target)
    return target


def authorization_response_uri(
    redirect_uri: str, *, code: str, state: str | None
) -> str:
    """Append response fields without changing the validated callback query."""
    params = {"code": code}
    if state is not None:
        params["state"] = state
    # Rebuilding the callback query would drop blank values and change its
    # escaping. The resolver already rejected fragments and unsafe URI forms.
    separator = "?"
    if "?" in redirect_uri:
        separator = "" if redirect_uri.endswith(("?", "&")) else "&"
    return redirect_uri + separator + urlencode(params)
