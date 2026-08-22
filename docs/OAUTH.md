# OAuth on the MCP endpoint — setup guide

vecgrep can gate untrusted `/mcp` traffic with **OAuth 2.1**, using an embedded
authorization server. You want this in exactly one situation: **a remote MCP
client that speaks OAuth needs to reach your vecgrep over the public
internet** — the main example being claude.ai's custom connectors. If your
only clients are local (Claude Code over stdio, tools on your tailnet), you
can leave it off and keep using network trust. When OAuth is enabled for a
public connector, direct loopback clients and authenticated Tailscale Serve
users retain network trust by default; anonymous Funnel clients do not.

## TL;DR — enabling it

```bash
# 1. Three env vars on the serve process (systemd unit, EnvironmentFile, shell):
VECGREP_OAUTH_ENABLED=1
VECGREP_OAUTH_ISSUER_URL=https://<your-public-host>/mcp   # the URL clients dial
VECGREP_OAUTH_APPROVAL_TOKEN=<strong random owner approval code>
# Optional master rollback: false requires OAuth for every MCP request.
VECGREP_OAUTH_LOOPBACK_BYPASS=true
# Optional: false also requires OAuth for authenticated Tailscale Serve users.
VECGREP_OAUTH_TAILSCALE_IDENTITY_BYPASS=true

# 2. Restart `vecgrep serve`.

# 3. Point the client at https://<your-public-host>/mcp and connect.
#    claude.ai: Settings → Connectors → Add custom connector → paste the URL.
#    It discovers the auth server, registers itself, and opens a browser prompt
#    for the owner approval code. There is no client ID to pre-provision.
```

The TLS proxy must expose the configured MCP path plus `/authorize`, `/token`,
`/register`, `/oauth/unlock`, `/.well-known/oauth-authorization-server`, and
the protected-resource metadata path advertised by discovery. Keep `/api/*`
private. Secret/path-prefixed MCP URLs are supported: the advertised resource
preserves the complete configured public path.

That's the whole setup. Everything below is what's happening and why.

## What actually runs

You did not just configure a client — vecgrep **is the authorization
server**. The MCP Python SDK ships the whole OAuth machinery; vecgrep
implements one interface (`OAuthAuthorizationServerProvider`, 9 methods in
`vecgrep/backend/auth/provider.py`) backed by a token store
(`auth/store.py`). The SDK owns:

- the HTTP routes: `/mcp/authorize`, `/mcp/token`, discovery under
  `/.well-known/*` (served at the **origin root**, deliberately — see
  gotcha 2),
- **PKCE** verification (the client proves it started the flow it's
  finishing),
- **Dynamic Client Registration** — the client registers itself at connect
  time, which is why there are no client IDs to configure,
- bearer-token gating of every `/mcp` request outside the trusted-network
  bypasses.

Dynamic registration does **not** prove that the person connecting is the
vecgrep owner. Before `/authorize` can mint a code, vecgrep therefore requires
the separate `VECGREP_OAUTH_APPROVAL_TOKEN` in a no-store browser form. The
resulting cookie is HttpOnly, Secure, SameSite=Strict, short-lived, and contains
only an HMAC verifier—not the approval token.

Token lifecycle (the store's invariants):

| Artifact | Lifetime | Notes |
|---|---|---|
| authorization code | 5 min | **single-use** — a replayed code cannot mint a second token |
| access token | 1 hour | checked on every request; expired never loads |
| refresh token | 30 days | bound to the client_id that minted it; revocable |

Tokens live **in memory**. A server restart wipes them and the client
silently re-runs the flow (it holds a refresh token → gets a clean 401 →
re-auths). For a single-user deployment this is a feature: revoking
everything is `systemctl restart`.

Scopes are `read` and `propose` only, and write-shaped tools enforce `propose`
at execution. The write path stays
propose → **human confirm**; confirm is never grantable via OAuth — an
authenticated client still cannot commit writes by itself.

## What it does NOT change

- OAuth applies only to `/mcp`. Keep the service loopback-bound and expose only
  MCP and OAuth routes through the public proxy. A non-loopback vecgrep bind
  separately requires a strong `VECGREP_API_TOKEN` and fails closed without it.
- Local stdio MCP (`vecgrep mcp`) is untouched — no network, no auth.
- Direct HTTP from a loopback peer bypasses OAuth only when both
  `X-Forwarded-For` and `X-Forwarded-Proto` are absent. Any proxy marker makes
  the request untrusted unless the verified Tailscale Serve rule below applies.
- Tailscale Serve strips client-supplied identity headers and adds the logged-in
  tailnet user's identity. That verified header plus proxy transit permits the
  default tailnet bypass. Funnel supplies no identity header and remains OAuth
  protected. Tagged nodes receive no user identity header and therefore need
  OAuth or an SSH tunnel to loopback.
- `VECGREP_OAUTH_LOOPBACK_BYPASS=false` is the master kill switch: it restores
  OAuth on every MCP request, including direct loopback and Tailscale Serve.
- With `VECGREP_OAUTH_ENABLED` unset, nothing about your deployment moves.

## The three gotchas (why the wiring looks the way it does)

If you ever rewire this — or port the pattern to another MCP server — these
are the three failure modes that ate the first attempt:

1. **Mount the MCP sub-app; don't path-rewrite.** The SDK's auth routes
   (`/authorize`, `/token`) live *inside* the MCP sub-app. A wrapper that
   rewrites every `/mcp/*` request to `/` (the old single-handler trick)
   makes them unreachable — the flow 404s at step one. `Mount("/mcp", app)`
   preserves sub-paths.

2. **Discovery lives at the origin root.** Clients fetch
   `/.well-known/oauth-authorization-server` from the *root* of your host,
   not under `/mcp`. If a SPA catch-all answers root paths first, the client
   receives your HTML homepage where it expected JSON and gives up.
   vecgrep registers the discovery routes ahead of the SPA fallback.

3. **Bare `/mcp` must not redirect.** Behind a path-prefixing proxy
   (Tailscale Funnel and friends), a 307 from `/mcp` → `/mcp/` is emitted
   absolute-from-root, drops the proxy's prefix, and the client loops on
   redirects forever. vecgrep delegates bare `/mcp` into the sub-app
   in-process — no redirect ever leaves the server.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Client says "couldn't fetch metadata" | Discovery shadowed — check `curl https://host/.well-known/oauth-authorization-server` returns JSON, not HTML |
| Redirect loop on connect | Something upstream is redirecting bare `/mcp`; dial the URL exactly as configured |
| Client re-auths after every deploy | Expected — in-memory token store; restart = clean slate |
| 401 on `/mcp` but flow completed | Access token expired (1 h) and the client isn't using its refresh token — reconnect the client |
| Local client gets 401 | Confirm it dials the direct loopback listener and no local proxy injects forwarding headers; check `VECGREP_OAUTH_LOOPBACK_BYPASS` |
| Tailnet client gets 401 | Use Tailscale Serve from a user-owned node; tagged nodes have no user identity header and need OAuth or a loopback SSH tunnel |
| Flow works locally, fails via funnel | `VECGREP_OAUTH_ISSUER_URL` must be the PUBLIC url, not localhost — issuer mismatch fails validation |

## Porting the pattern

The `auth/` module has no vecgrep-specific logic — token store + provider
lift onto any MCP server built on the Python SDK. Implement nothing yourself:
subclass the provider, hand the SDK `issuer_url`, `resource_server_url` and
client-registration options, and respect the three gotchas above.
