"""Embedded OAuth 2.1 authorization server for the vecgrep MCP endpoint.

This package makes vecgrep its own OAuth "front desk" so an MCP client that
speaks OAuth (e.g. claude.ai) can authenticate — replacing the secret-capability
URL with short-lived, scoped, revocable tokens.

Division of labor with the MCP SDK (mcp.server.auth):
  - The SDK provides the HTTP routes (/authorize, /token, the
    /.well-known/oauth-authorization-server metadata), PKCE verification, and
    the bearer middleware that gates /mcp.
  - WE provide the OAuthAuthorizationServerProvider: the storage + lifecycle for
    clients, authorization codes, access tokens, and refresh tokens (issue,
    load, exchange, revoke, expiry). That's this package.

Scope model: tokens carry scopes. `read` = search/list/get; `propose` = the
write tool's propose step. There is NO `confirm` scope — confirmation stays a
human action off the OAuth protocol entirely (the wall).
"""
