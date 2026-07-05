# Changelog

All notable changes to vecgrep are documented here. Versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0: minor = features,
patch = fixes). The version is a single source of truth in
`vecgrep/__init__.py`; `pyproject.toml` reads it dynamically.

## [Unreleased]

## [1.1.0] — 2026-07-04

Tool-surface follow-up to v1.0, from a post-release audit.

### Added
- **`browse`** (service + MCP both transports): location-first reading — the
  full event sequence for a channel and/or UTC day and/or path glob, no
  query, no ranking. Complements query-first search/timeline ("show me
  channel X on day Y"). Requires at least one selector — a bare corpus dump
  is refused.
- **`get_source`** (service + MCP): whole source document by source_id (raw
  text + parsed transcript events + metadata) — the by-source complement of
  get_chunk's by-chunk_id window.
- **`list_aliases`** (MCP): read-only view of the active alias-expansion map,
  so a caller can see why a query matched terms it didn't type.

### Changed
- MCP tool handlers now reuse one `VecgrepService` per settings generation
  (previously a fresh service per tool call — registry/store/cache handles
  rebuilt every invocation); stale services are closed on settings change
  and at exit.
- The stdio MCP transport now exposes the same `propose_write` /
  `propose_edit` / `propose_delete` write tools as the HTTP transport
  (parity was silently missing).

## [1.0.0] — 2026-07-04

**The memory-retrieval quality release** — v1.0, stable surface. Driven by
real-world use as a personal-memory / chat-transcript search layer: better
RESULT ASSEMBLY over the unchanged hybrid-search + rerank core — dedup/MMR,
result budget, hard time/path filters, timeline + incident reconstruction,
clearer scores, alias expansion, precise anchors — plus OAuth on /mcp.
Measured end to end by a committed eval harness: duplicate-slice redundancy
0.089 → 0.022 (0.111 → 0 on the keyword probe, with recall 0.67 → 1.0), and
out-of-window leakage on dated questions 4 hits → 0. See
`docs/superpowers/specs/2026-07-04-vecgrep-v1-memory-retrieval-design.md`.

### Added
- **Precise source anchors**: every result carries a 1-based inclusive
  `line_start`/`line_end` range and a compact `anchor` citation
  (`path#L12-L24`) alongside the stable `chunk_id`, on the service, API and
  MCP surfaces — an assistant can cite and re-open the exact region.
- **Incident object**: `service.incident()` + MCP `incident` tool — one
  structured answer for an incident question (title, sources, participants,
  time range, primary chronological timeline with related context kept
  separate, confidence label), assembled purely from the search + timeline
  primitives. Returns nothing rather than inventing structure when no
  evidence matches.
- **MCP OAuth** (re-integrated from the parked `feat/mcp-oauth` branch):
  optional OAuth 2.1 on the `/mcp` endpoint (`VECGREP_OAUTH_ENABLED=1` +
  `VECGREP_OAUTH_ISSUER_URL=<public url>`). Embedded authorization server
  (`vecgrep/backend/auth/`: token store + auth-code/refresh provider), MCP
  sub-app mounted so the SDK's /authorize, /token and /.well-known routes
  resolve, discovery served at the origin root ahead of the SPA catch-all,
  funnel-safe bare `/mcp` delegate (no 307 loop), dynamic client
  registration advertised. Off by default; the localhost/tailnet /api
  surface is unchanged. The no-frontend fallback now answers all unknown
  paths with the API-running message (uniform with the SPA catch-all).
- **Alias / entity expansion** (`vecgrep/backend/aliases.py`): one entity,
  many surface forms (nickname ⇄ handle ⇄ another language). A query naming
  any form expands with the entity's other forms before retrieval, so
  evidence written under a different form still surfaces. The map is
  user-supplied personal data OUTSIDE the repo (`$VECGREP_HOME/aliases.json`
  or `VECGREP_ALIASES_FILE`; generic example at `docs/aliases.example.json`);
  empty/missing map is an exact no-op. ASCII forms match on word boundaries
  ("ali" never fires inside "alignment"), CJK forms by containment.
  Config-driven, not learned. Opt out per call with `expand_aliases=False`.
- **Timeline mode** (`vecgrep/backend/timeline.py`): "what happened?"
  questions get an ordered event sequence instead of ranked chunks. A normal
  search (dedup/MMR/filters apply) finds anchor chunks; the best source
  files each contribute ONE contiguous slice spanning their anchors, parsed
  into (speaker, time, text) events via the transcript format — chronological
  within a file, files oldest → newest, non-transcript sources degrade to
  the raw slice. Surfaces: `service.timeline()`, `POST /api/timeline`, MCP
  `timeline` tool (stdio + HTTP), CLI `vecgrep timeline "<query>"`.
- **Clearer score output**: results now carry `relevance_pct` (going-forward
  name; `similarity_pct` stays as a compatibility alias) and a qualitative
  `relevance_label` (exact ≥95 / strong ≥75 / related ≥40 / weak) on the
  service, API, MCP and CLI surfaces. Raw component scores (vector cosine,
  BM25, RRF, decay, rerank) were already attached to every result. The
  rerank display sigmoid is flattened (slope 35 → 18) so strong hits stop
  bunching at 99.x — 0.66 vs 0.75 rerank scores now read ~84% vs ~96%
  instead of 96/99.8. Ranking order is unchanged (monotonic remap).
- **Date / path / time filters as hard constraints**: `date:YYYY-MM-DD`,
  `after:<iso>`, `before:<iso>`, `channel:<name>` (quote-tolerant against
  archiver frontmatter), `source_path:<glob>` (alias of `source:`) join the
  filter grammar on every surface (CLI `--filter`, API `filters`, MCP
  `search.filters`). Time filters are hard: undated chunks fail them, and an
  unparseable value matches NOTHING (failing closed makes a typo'd date
  visible as zero results instead of silently leaking old lore). Fixes the
  over-eager "this morning" query dragging in week-old evidence — eval:
  the dated incident query goes from 4 out-of-window hits to 0 with
  `date:` passed, recall unchanged. `/api/corpora/{name}/filters` now
  advertises the new forms (time filters only when the corpus has dated
  chunks).
- **Result budget + stub tier**: breadth without a blown context. Budgeted
  search returns the top `full_k` (default 8) hits with context windows plus
  a one-line stub tail (source + timestamp + snippet + score + chunk_id, NO
  context) emitted in rank order until a configurable token ceiling
  (default ~4000), capped at 80 total. Every stub expands to full context
  via its chunk_id. Surfaces: `service.search_budgeted()`,
  `POST /api/search {"budget": true}` (+ `stubs` in the response), MCP
  `search(budget=true)` + new `get_chunk` tool (stdio + HTTP), CLI
  `vecgrep search --budget` + new `vecgrep chunk <corpus> <chunk_id>`.
  Runs after dedup/MMR by design — the widened budget is spent on distinct
  evidence, not duplicate slices. `SearchResult`/API hits now carry
  `doc_timestamp`. Web UI default hit count 5 → 20.
- **Source-span dedup / MMR** (`vecgrep/backend/assembly.py`): result
  selection is now diversity-aware. The hard dedup extends beyond span
  overlap to same-source text clones at distant spans (repeated messages /
  bot alert spam), and top_k selection uses greedy MMR (relevance-dominant,
  λ=0.7) with near-clone exclusion, on both the fused-score and rerank
  paths. Corpora without near-dups degrade to plain score order. Eval:
  dup-probe redundancy 0.089 → 0.022; exact-keyword redundancy 0.111 → 0.0
  with expected-source recall 0.67 → 1.0 (freed slots went to distinct
  evidence).
- **Eval harness** (`tests/eval_harness.py`): synthetic PII-free Discord-style
  transcript fixtures (bilingual, alias-bearing, multi-channel incident, bot
  alert spam, exact-keyword + semantic-vague cases) + 7 golden benchmark
  queries with assembly-quality metrics (redundancy, expected-source recall,
  forbidden-source leakage). v0.7.0 baseline captured at
  `tests/fixtures/baselines/v0_7_0.json`; regenerate with
  `python -m tests.eval_harness`. The baseline documents the problems this
  release fixes: duplicate slices survive span-only dedup (redundancy 0.09 on
  the dup probe) and date-less queries leak week-old lore (4 forbidden hits on
  the "today" incident query).

## [0.7.0] — 2026-07-04

The write + retrieval-quality release. 91 commits since v0.6.0; retroactively
versioned as part of adopting a real release cadence toward v1.

### Added
- **Write path**: `vecgrep write` tool with a propose → human-confirm → commit
  flow; per-corpus write-through, `propose_delete` (inert, confirmed removal),
  discard command, memory/todo source kinds, auto-derived titles, timestamped
  doc ids. Default-deny corpus allowlist + size cap on the propose tool. (#11,
  #13, #20)
- **Retrieval quality**: dedup of overlapping chunks from the same source;
  per-corpus tunable recency decay; per-chunk `doc_timestamp` extracted from
  source dates; gentle vector noise floor before fusion; BM25 partial-coverage
  penalty as default; markdown-section chunker respecting heading boundaries.
  (#21, #23, #25)
- **Scoring**: `similarity_pct` unified on the calibrated rerank score.
- **Indexing**: `--include` glob to filter directory indexing; filterable field
  schema exposed in `get_corpus`.
- **Embeddings**: optional second Ollama fallback before OpenAI; embed cache
  capped at 50k rows with FIFO eviction. (#24)
- **Web UI**: "how search works" explainer; skeleton search loader; mobile text
  wrap; clickable tuning toggle; per-corpus calibration seeded from server. (#22)

### Changed
- Dead-code cleanup: removed unused imports, deduped export-adapter JSON
  loading and CLI HTTP error handling.

## [0.6.0] — 2026-05-02

- pytest suite + 3 bug fixes the tests caught.

## [0.5.0] — 2026-05-02

- Early iteration (same-day cut with 0.3.0–0.6.0 during initial versioning).

## [0.4.0] — 2026-05-02

- Early iteration.

## [0.3.0] — 2026-05-02

- First tagged release.

[Unreleased]: https://github.com/jeffbai996/vecgrep/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/jeffbai996/vecgrep/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/jeffbai996/vecgrep/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/jeffbai996/vecgrep/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/jeffbai996/vecgrep/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jeffbai996/vecgrep/releases/tag/v0.3.0
