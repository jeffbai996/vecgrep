"""One OAuth scope policy for registration, consent, and token issuance."""
from __future__ import annotations

from collections.abc import Iterable


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
