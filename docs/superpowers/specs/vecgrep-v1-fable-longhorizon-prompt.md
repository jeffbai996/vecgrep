# Fable long-horizon job — vecgrep v1 (memory-retrieval quality release)

> Paste this whole document as the job prompt for a Fable 5 long-horizon
> autonomous run. It is self-contained: context, scope, constraints, phase
> plan, verification gates, and definition of done. Work through the phases in
> order; each phase has an explicit "done when" gate you must satisfy before
> moving on.

---

## Who you are and what you're doing

You are an autonomous engineer taking `vecgrep` from **v0.7.0 → v1.0**. vecgrep is
a local-first semantic search engine ("grep for meaning, not keywords") — hybrid
vector + BM25 retrieval with a cross-encoder rerank, a CLI, an HTTP/MCP API, and a
React web UI. Its highest-value real-world use has turned out to be a **personal
memory / chat-transcript search layer**: an AI assistant querying messy,
multilingual, multi-channel conversation logs in informal language.

This job implements the improvements from a real-world-usage review plus a couple
of owner requests. The full design is committed at
`docs/superpowers/specs/2026-07-04-vecgrep-v1-memory-retrieval-design.md` — read
it first; this prompt is the executable version of it.

## Starting state (already done — do NOT redo)

- You are on branch **`feat/memory-v1`** in an isolated git worktree at
  `~/repos/vecgrep-dev`, with its own venv. **The live production service runs
  from a DIFFERENT checkout (`~/repos/vecgrep`) — never touch it.** All your work
  happens here; it merges to `main` only at the very end.
- Version is already SSOT'd: `vecgrep/__init__.py:__version__` is the one source,
  `pyproject.toml` reads it dynamically. Baseline is tagged `v0.7.0`.
- The retrieval **core already has hooks you will EXTEND, not rebuild**:
  `service.py` has `_dedup_overlapping` (overlap dedup), `_recency_factor`
  (recency decay), `_rerank_to_pct` + `_calibration_for` (score calibration);
  `SearchResult` carries `context_before/after`; `ingestion/dates.py` extracts
  per-chunk `doc_timestamp`. Build on these.

## Absolute constraints (violating any of these fails the job)

1. **Do not touch the live service checkout** (`~/repos/vecgrep`) or restart
   `vecgrep-serve` until the final merge step, and only then if instructed.
2. **Preserve the working core.** Do NOT rewrite the hybrid-search / rerank
   pipeline. This job improves *result assembly* — dedup, budget, filters,
   chronology, scores, aliases — on top of the existing stack. (This is the
   review's hard guardrail, point 10.)
3. **TDD.** Every feature gets failing tests FIRST, then implementation to pass
   them. No production code without a failing test that describes it.
4. **Public-mirror safe.** This repo has a public GitHub mirror. NEVER commit
   real personal data: no real nicknames, bot handles, usernames, household
   names, tickers, or private project names in code, tests, fixtures, or docs.
   Use generic placeholders (Alice/Bob/alias-1, `channel-a`, `2026-01-01`). A
   pre-commit PII hook will block you if you slip — treat a block as a correct
   signal to genericize, not something to override.
5. **Keep the CLI/API simple** enough that an AI assistant uses it naturally
   (the review's closing note). Prefer explicit params over magic.
6. **Every phase ends green**: full test suite passes, no regressions, and you've
   updated `CHANGELOG.md [Unreleased]` with what landed.

## Scope — build in this order

### Phase 0 — Test harness + eval fixtures (foundation)
Everything else tests against this. Create a small evaluation set of
**synthetic, PII-free** chat-transcript fixtures that mimic the messy real-world
shape: bilingual text, aliases/nicknames (generic), multiple channels, a
multi-message incident spanning files, duplicate/overlapping chunks, exact-keyword
cases, semantic-vague cases, code/project terms in casual chat. Add golden
benchmark queries (genericized versions of the review's: "why did <system> fail
today?", "what was X arguing with Y about this morning?", "show the greatest hits
of Z this morning"). **Done when:** fixtures + a runnable eval harness exist and
current-behavior baselines are captured (so improvements are measurable).

### Phase 1 — Source-span dedup / MMR
Extend `_dedup_overlapping` into diversity-aware selection (MMR). Dedup on: same
`source_id` + overlapping message/line range + high text similarity + nearby
timestamps. Goal: return one chunk each for distinct sub-events (escalation / main
thread / postmortem / resolution) instead of five near-identical slices of one
exchange. **Done when:** on a fixture with 5 overlapping slices of one incident,
results collapse to diverse representatives; eval harness shows less redundancy,
no loss of distinct evidence.

### Phase 2 — Result budget + stub tier
Surface far more results without blowing the token budget: **top 8 full**
(context windows, all scores) then **up to 72 stubs** (source path + timestamp +
one-line snippet + score, NO context windows). **Token-driven**: emit stubs until
a token ceiling, so it auto-scales (8 full + ≤72 stubs ≤ 80 total). Add a way for
the assistant to re-query full context for any stub. **Also:** web UI default hit
count **5 → 20** (plain default change; UI renders full cards, no stub tier).
Sequenced after dedup on purpose — expanding the count is only worth it once dups
are gone. **Done when:** a query returns 8 full + a token-bounded stub tail; UI
defaults to 20; token ceiling is respected and configurable.

### Phase 3 — Date / path / time filters
Parse assistant-supplied filters and push them as HARD constraints:
`date:2026-01-01`, `after:<iso>`, `before:<iso>`, `channel:<name>`,
`source_path:<glob>`. Fixes over-eagerness (dragging in old lore for a "this
morning" query). vecgrep does NOT infer intent from natural language — the caller
passes explicit filters. **Done when:** filtered queries exclude
out-of-window/out-of-path evidence on the fixtures; filters are documented and
easy to pass via CLI + API + MCP.

### Phase 4 — Clearer score output
Expose raw vector / BM25 / rerank / fused scores separately. Add qualitative
labels (exact / strong / related / weak). Make the displayed percent less
clustered at 99.x (calibrate so 99.6 vs 99.2 is meaningful) — consider renaming
`similarity_pct` → `relevance_pct` (keep the old field as an alias for
compatibility). **Done when:** results carry the component scores + a label; the
displayed percent spreads meaningfully across a mixed result set.

### Phase 5 — Timeline mode
`search(mode="timeline")` or a `timeline()` helper: find the likely source files /
time window, pull a **contiguous timestamp-ordered slice** around the incident,
return the event sequence in chronological order with speakers + timestamps
preserved, **grouped by source file** (not ranked chunks). **Done when:** a
"what happened" fixture query returns an ordered, readable event sequence the
assistant doesn't have to re-sort from overlapping chunks.

### Phase 6 — Alias / entity expansion
A config-file alias map (one entity, many surface forms). Query mentions one form
→ BM25/vector query expands to the set. Config-driven, not learned. The map is
**user-supplied and lives OUTSIDE the repo** (it's personal data); vecgrep only
consumes it, ships with an empty/example generic map. **Done when:** with a test
alias map, a query for one alias surfaces evidence written under any of its forms.

### Phase 7 (parallel track) — MCP OAuth re-integration
Re-integrate the OAuth work parked on branch `feat/mcp-oauth` (6 commits: a
`vecgrep/backend/auth/` module — token store, auth-code + refresh provider — plus
wiring into `main.py` / `config.py` / `mcp/server.py`). The auth module ports
mostly clean; the wiring needs a careful merge because `mcp/server.py` advanced 6
commits on main since the branch. Port it onto current `feat/memory-v1`, resolve
conflicts, get its tests green. **Done when:** OAuth flow works end-to-end against
the MCP server with tests, no regression to the existing gate.

### Phase 8 (stretch — only if phases 0–7 are solid) — the review's deferred items
These are higher-risk; do them ONLY after the core v1 is green, each behind its
own tests:
- **Precise source anchors** (review #7): include line ranges / message indices /
  timestamps / thread + channel names / stable span IDs on results, so the
  assistant can cite/re-open exact regions.
- **Incident reconstruction format** (review #5): an `incident` result object —
  title, sources, participants, time range, ordered primary timeline, related
  context (separated from primary), confidence — assembled from the timeline +
  dedup + filter primitives already built.
- **Query-intent helper** (review #8): optional hints (fact-lookup / timeline /
  greatest-hits / debugging / recap / entity-history) that tune retrieval, either
  explicit modes or returned hints. Keep it optional and additive.

### Phase 9 — Cut v1.0
When phases 0–7 are green (8 is a bonus): bump `__version__` to `1.0.0`, write the
`1.0.0` CHANGELOG entry, update the README (version badge + roadmap + document the
new modes/filters/config), tag `v1.0.0`. Prepare the merge to `main` but **do not
merge or restart the live service without explicit owner confirmation** — leave it
staged with a clear summary of what changed and what the merge/deploy steps are.

## How to work

- Read the design spec first, then the relevant source before each phase.
- One phase at a time; keep the suite green between phases; commit per logical
  unit with clear messages (conventional commits, no AI attribution).
- Update `CHANGELOG.md [Unreleased]` as you go.
- If a phase reveals the design was wrong, stop and write down the conflict +
  your proposed change rather than silently diverging.
- Prefer small, focused modules (one job per file). If `service.py` grows
  unwieldy, factor the new assembly logic into its own module rather than piling
  on.
- Leave the hybrid-search/rerank core alone unless a change is provably necessary
  and covered by tests.

## Definition of done

- Phases 0–7 complete, each with passing tests; phase 8 attempted if time allows.
- Full suite green, no regressions against the v0.7.0 baseline behaviors.
- CLI + API + MCP + web UI all reflect the new capabilities, documented.
- No personal data anywhere in the repo (PII hook clean).
- `v1.0.0` tagged on `feat/memory-v1`, CHANGELOG + README updated, merge to main
  staged with a written deploy summary — awaiting owner confirmation to merge and
  restart the live service.
