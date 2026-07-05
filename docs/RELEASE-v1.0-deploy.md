# v1.0.0 — merge & deploy runbook

**Status: STAGED — awaiting owner confirmation. Nothing below has been run.**

## What ships

`feat/memory-v1`, 10 commits over `v0.7.0` (`508249f`), tagged `v1.0.0`:

| Commit | What |
|---|---|
| `47c35a8` | eval harness + transcript fixtures + v0.7.0 baseline |
| `d0c1557` | source-span dedup + MMR diversity selection |
| `840eeb9` | result budget + stub tier (8 full + token-capped stubs) |
| `6348630` | date/path/time filters as hard constraints |
| `94ed2bc` | relevance labels + de-clustered display pct |
| `7bc4ae7` | timeline mode — chronological event reconstruction |
| `b300541` | config-driven alias/entity expansion |
| `cfa76de` | OAuth 2.1 on /mcp (port of feat/mcp-oauth) |
| `69841b2` | line anchors + incident object (stretch 8a/8b) |
| (release) | v1.0.0 — version, CHANGELOG, README, this runbook |

Suite: **350 pass**, 3 pre-existing environment failures (identical on the
live checkout: no local Ollama at localhost, one qdrant-lock test quirk).
Measured wins (committed baseline → now): dup redundancy 0.089→0.022,
keyword-probe redundancy 0.111→0 with recall 0.67→1.0, dated-question
out-of-window leakage 4→0.

## Compatibility

- **API/CLI/MCP: strictly additive.** Old request shapes and response fields
  all still work; `similarity_pct` kept as alias of `relevance_pct`.
- **Behavior changes** (intentional, display/assembly only):
  1. Search results are deduped harder + MMR-diversified (repeated bot spam
     collapses). Ranking of distinct content unchanged.
  2. Rerank display % is flatter (strong hits read 84–96% instead of 99.x).
     Order unchanged.
  3. Web UI default hit count 5 → 20.
- **No index/schema migration.** Existing corpora work as-is.
- OAuth is **off by default** — no change to the running deployment's auth.

## Merge + deploy steps (when confirmed)

```bash
# 1. merge (in the LIVE checkout ~/repos/vecgrep)
cd ~/repos/vecgrep
git fetch ~/repos/vecgrep-dev feat/memory-v1
git checkout main
git merge --ff-only FETCH_HEAD        # fast-forward; no merge commit needed
git tag -f v1.0.0 && git push origin main v1.0.0   # owner pushes

# 2. deps unchanged (no new required packages; mcp extra already installed)

# 3. restart the live service
systemctl --user restart vecgrep-serve
sleep 3 && systemctl --user is-active vecgrep-serve
curl -sf http://127.0.0.1:8765/api/health

# 4. smoke (against the live chats corpus)
vecgrep search "test" --corpus chats --budget --top 5 | head
vecgrep timeline "any recent incident" --corpus chats | head
```

Rollback: `git reset --hard v0.7.0 && systemctl --user restart vecgrep-serve`
(no data migration to unwind).

## Deferred / follow-ups

- Query-intent hints (stretch 8c) — deliberately skipped, most speculative.
- `service.py` god-file split, operator vision wiring — out of scope per brief.
- The 3 pre-existing env test failures (need a local Ollama or fixture rework).
