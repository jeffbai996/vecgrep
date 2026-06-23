"""Bearer-token gate — trailing-whitespace tolerance + constant-time compare.

Regression guard: the gate stripped the client's token but not the configured
one, so a trailing newline in VECGREP_API_TOKEN (common from env/file reads)
rejected every valid request with a 403 — an auth DoS against yourself.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from vecgrep.backend.api import routes


class _Settings:
    def __init__(self, token):
        self.api_token = token


def test_trailing_newline_in_configured_token_still_authenticates(monkeypatch):
    monkeypatch.setattr(routes, "get_settings", lambda: _Settings("s3cr3t-token\n"))
    # Must NOT raise — the configured token's trailing newline is stripped.
    routes.require_token("Bearer s3cr3t-token")


def test_wrong_token_is_rejected(monkeypatch):
    monkeypatch.setattr(routes, "get_settings", lambda: _Settings("s3cr3t-token"))
    with pytest.raises(HTTPException) as ei:
        routes.require_token("Bearer not-the-token")
    assert ei.value.status_code == 403


def test_missing_token_is_401(monkeypatch):
    monkeypatch.setattr(routes, "get_settings", lambda: _Settings("s3cr3t-token"))
    with pytest.raises(HTTPException) as ei:
        routes.require_token(None)
    assert ei.value.status_code == 401


def test_no_configured_token_is_noop(monkeypatch):
    monkeypatch.setattr(routes, "get_settings", lambda: _Settings(None))
    routes.require_token(None)  # gate disabled → no-op
