# vecgrep v1 — memory-retrieval quality design

**Status:** draft for review · **Target:** v1.0 (stable, usable) · **Branch:** `feat/memory-v1`
**Baseline:** v0.7.0 (`508249f`) · **Source review:** ChatGPT real-world-usage review (10 points), 2026-07-04

## Goal

vecgrep already works well as a personal-memory / Discord-transcript search layer.
The v1 push is **not** a rewrite of the retrieval core — it's better **result
assembly** on top of the existing hybrid (vector + BM25) + rerank pipeline, so an
AI assistant asking messy conversational questions ("what was X arguing with Y
about this morning?") gets clean, chronological, de-duplicated, well-attributed
evidence instead of five overlapping slices of the same exchange.

**Hard guardrail (from the review, point 10):** do not over-engineer away current
strengths. Hybrid search + rerank + context-window design stays. We improve
assembly, dedup, chronology, filters, and source precision.

## What already exists (so we build on, not rebuild)

The current pipeline (`vecgrep/backend/service.py` orchestrates) already has the
hooks several review items want:

| Capability | Current state (v0.7.0) | Location |
|---|---|---|
| Hybrid retrieval + fusion | ✅ vector + BM25, fused | `service.py`, `store/` |
| Rerank → calibrated `similarity_pct` | ✅ `_rerank_to_pct`, `_calibration_for` | `service.py:1226,1282` |
| Chunk dedup (overlap) | ✅ `_dedup_overlapping(min_overlap=0.5)` | `service.py:1292` |
| Recency decay | ✅ `_recency_factor`, per-corpus half-life | `service.py:106,620` |
| Per-chunk `doc_timestamp` | ✅ extracted from source dates | `ingestion/dates.py` |
| Context windows | ✅ `context_before/after` on `SearchResult` | `service.py:128` |

So the review's asks are mostly **extend existing mechanisms**, not new subsystems.

## Scope — two independent tracks

### Track A — retrieval quality (the review's top 5)
Prioritized exactly as the review's deliverables list:

1. **Source-span dedup / MMR.** Extend `_dedup_overlapping` from pure overlap into
   diversity-aware selection (MMR): prefer one chunk each for distinct
   sub-events (escalation / main thread / postmortem / resolution) over five
   near-identical slices of the same timestamp span. Dedup keys: same `source_id`
   + overlapping message/line range + high text similarity + nearby timestamps.
2. **Date / path / time filters.** Parse assistant-supplied filters —
   `date:2026-07-04`, `after:<iso>`, `before:<iso>`, `channel:yaya`,
   `source_path:*yaya/2026-07-04.md` — and push them into the query as hard
   constraints. Fixes the "over-eager, drags in July-3 lore for a 'this morning'
   query" problem. Filters are explicit params the caller passes; vecgrep does
   NOT guess intent from natural language here (that's the assistant's job).
3. **Timeline mode.** `search(mode="timeline")` (or a `timeline()` helper): find
   the likely source files / time window, pull a **contiguous timestamp-ordered
   slice** around the incident, return event sequence in chronological order with
   speakers + timestamps preserved — grouped by source file, not ranked chunks.
4. **Clearer score output.** Expose raw vector / BM25 / rerank / fused scores
   separately; add qualitative labels (exact / strong / related / weak); make
   `similarity_pct` (or a renamed `relevance_pct`) less clustered at 99.x so
   99.6 vs 99.2 means something. Calibration hook already exists.
5. **Alias / entity expansion.** A config-file alias map — one entity, many
   surface forms (a nickname ⇄ a bot handle ⇄ a username ⇄ a name in another
   language). Query mentions one form → BM25/vector query expands to the whole
   set. Config-driven, not learned (v1). The map itself is user-supplied and
   lives outside the repo (it's personal data); vecgrep just consumes it.

6. **Result budget + stub tier.** Surface far more results by default without
   blowing the token budget: **top 8 full** (context windows, all scores) then
   **up to 72 stubs** (source path + timestamp + a one-line snippet + score, NO
   context windows). Token-driven: emit stubs until a token ceiling is hit, so
   it auto-scales rather than a fixed count (8 full + ≤72 stubs = ≤80 total).
   The assistant gets breadth to spot patterns, then can re-query for full
   context on any stub. Pairs with dedup/MMR (item 1): kill the duplicate slices
   first, so the expanded budget spends on *distinct* evidence.
   - **Web UI:** default hit count **5 → 20** (a plain default change; the UI
     doesn't need the stub tier — it renders full cards, just more of them).

Deferred to post-v1 (review points 5, 8): full "incident reconstruction" object
format and automatic query-intent detection. They're valuable but higher-risk;
v1 gives the assistant the primitives (timeline slice + filters + aliases) to
assemble incidents itself. Revisit once the primitives are proven.

### Track B — MCP OAuth (from `feat/mcp-oauth`)
Re-integrate the orphaned OAuth work onto current main: the self-contained
`vecgrep/backend/auth/` module (token store, auth-code + refresh provider) ports
mostly clean; the wiring into `main.py` / `config.py` / `mcp/server.py` needs a
careful merge (main advanced `mcp/server.py` by 6 commits since the branch). This
track is independent of Track A and can land in parallel or after.

## Testing (review point 9)

An evaluation set of messy Discord-style transcript fixtures + benchmark queries
(bilingual EN/中, aliases, same-day filtering, multi-channel incident, dup
suppression, exact-keyword, semantic-vague). Each v1 feature ships with fixtures
that prove the improvement. TDD: fixtures + failing assertions before each
feature's implementation. Golden queries from the review (e.g. "why did the world
cup watcher fail today?") become regression cases.

## Sequencing

Highest-leverage first, each isolated + tested in the `feat/memory-v1` worktree,
merged to main only when green + proven, **never touching the live service checkout**:

1. Test harness + transcript fixtures (the eval set — everything else tests against it)
2. Source-span dedup / MMR
3. Result budget + stub tier (8 full + ≤72 stubs, token-driven) + web UI 5→20
4. Date / path / time filters
5. Clearer score output
6. Timeline mode
7. Alias expansion
8. (parallel track) OAuth re-integration
9. Cut **v1.0**, update CHANGELOG, tag, merge to main, restart live service

Dedup (2) lands before the stub tier (3) on purpose: expanding the result count
is only worth it once duplicates are gone, else the extra budget fills with noise.

## Non-goals for v1

- Rewriting the hybrid-search / rerank core.
- Natural-language intent parsing inside vecgrep (assistant's job; vecgrep takes
  explicit filters/modes).
- Learned alias discovery (config-driven only in v1).
- Full incident-reconstruction object (primitives only; assistant assembles).
