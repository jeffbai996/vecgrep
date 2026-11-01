"""One OAuth scope policy for registration, consent, and token issuance."""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.auth.provider import OAuthAuthorizationServerProvider

VALID_SCOPES = ("read", "propose")
DEFAULT_SCOPES = ("read",)


class UnsupportedScope(ValueError):
    """The client requested a permission this server does not grant."""


def effective_scopes(
    requested: Iterable[str] | None,
    *,
    valid_scopes: Iterable[str] = VALID_SCOPES,
    default_scopes: Iterable[str] = DEFAULT_SCOPES,
) -> list[str]:
    """Return the exact scopes consented to and issued.

    OAuth leaves an omitted scope to server policy. Vecgrep's policy is
    deliberately least-privilege: omission means read-only, never every scope
    the client happened to register. Unknown scopes fail closed instead of
    being silently discarded.
    """
    valid = tuple(valid_scopes)
    selected = list(requested or default_scopes)
    selected = list(dict.fromkeys(scope for scope in selected if scope))
    unsupported = [scope for scope in selected if scope not in valid]
    if unsupported:
        raise UnsupportedScope(
            f"Unsupported OAuth scope(s): {', '.join(unsupported)}"
        )
    return selected


async def approve_write_request(target: str, provider: OAuthAuthorizationServerProvider) -> str:
    """Expand one request only after the browser's explicit owner approval.

    Registration is updated as well: older clients registered only read and
    the SDK would otherwise reject the owner-selected scope before consent.
    Existing access and refresh tokens keep their original grants.
    """
    from urllib.parse import urlencode, urlsplit
    from .redirects import authorization_query, invalid_request

    values = authorization_query(target)
    client = await provider.get_client(values["client_id"])
    if client is None or effective_scopes(values.get("scope", "").split()) != ["read"]:
        raise invalid_request()
    registered = effective_scopes((client.scope or "").split())
    selected = list(dict.fromkeys([*registered, "read", "propose"]))
    await provider.register_client(client.model_copy(update={"scope": " ".join(selected)}))
    values["scope"] = "read propose"
    return urlsplit(target).path + "?" + urlencode(values)
