# Changelog

All notable changes to vecgrep are documented here. Versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0: minor = features,
patch = fixes). The version is a single source of truth in
`vecgrep/__init__.py`; `pyproject.toml` reads it dynamically.

## [Unreleased]

### Fixed
- Same-machine trust now requires the client peer address to be exactly
  `127.0.0.1` or `::1`. It was any address in `127.0.0.0/8`, so a local TCP
  forwarder dialing from `127.0.0.2` was treated as a direct local MCP client,
  and as a verified tailnet user when it carried Tailscale Serve headers. Such
  peers now go through OAuth like any other remote caller.
- OAuth access tokens live one hour, as the documentation always said; they had
  been issued for a year. The approval still lasts a year through the refresh
  token, which clients rotate silently. Access and refresh tokens already on
  disk are clamped to the current lifetimes when the store loads, so a restart
  retires the year-long access tokens within the hour.
- The OAuth unlock page names the host the connection returns to, next to the
  client's own name, so an owner can tell a client that registered a callback
  elsewhere.
- The registration ingress hands the application the client's own `receive()`
  after replaying the buffered body, instead of synthetic empty bodies.
- Replaying an OAuth refresh token that was already rotated revokes every
  access and refresh token descended from the same approval, instead of only
  refusing the retired token. Revoking a refresh token revokes its family's
  access tokens too. State files written before families existed load with one
  family per refresh token, so connected clients stay connected.

### Added
- Parser-backed `code_symbol` chunks for Python (standard library `ast`) and
  TypeScript/TSX (tree-sitter, the optional `vecgrep[code]` extra). Each class
  member and nested class gets its own chunk with `qualname`, `kind`, `parent`
  and `lines` metadata; functions stay whole and list what they define in
  `nested`. Decorators and a comment block directly above a definition belong
  to it. A file that fails to parse, or nests deeper than the recursion limit,
  is chunked by the old heuristic and marked `parse_error`, so no content is
  lost. Line spans follow each parser's own line breaks, so a form feed,
  U+2028 or bare carriage return no longer cuts a symbol short, and a Python
  file that starts with a byte order mark parses. Other languages are
  unchanged.
- Chunks may carry a `search_text` (embedded and BM25-indexed in place of the
  chunk text): code chunks put their qualified name and the imports they use in
  front of the code. The stored `text` and its offsets still address the source
  exactly. BM25 rebuilds and mutation-journal recovery index `search_text` read
  back from the payload, and a cache sweep keeps the vectors embedded from it.
- An identifier-shaped query (`run`, `Importer.run`, `Stats.run`) ranks and pins
  chunks whose metadata defines that symbol above chunks that only mention it.
  It reorders candidates already retrieved and is a no-op for prose queries and
  for chunks without symbol metadata. `scripts/bench_code_symbols.py` compares
  the heuristic with the parser on fixtures in `tests/fixtures/code_symbols`.
- Bulk embedding batches record their route in the local receipt database's
  `embed_route` table: `primary` or `overflow`, `ok` or `fallback`, and a
  short reason code for a fallback (`refused`, `timeout`, `unreachable`,
  `error`, `http_<code>`, `bad_response`, `bad_vectors`). An overflow
  fallback sends the batch to the primary; a primary fallback sends it to
  per-chunk requests on the primary. `ok` is written only after the answer
  validates. `error` is a connection that broke mid-answer. Queries are not
  recorded there. The shared `usage` table keeps its six columns, and no URL,
  host or text is stored.
- `VECGREP_OLLAMA_OVERFLOW_URL` / `ollama_overflow_url`: an optional second
  Ollama endpoint for bulk embedding. Multi-text batches go there first;
  queries stay on the primary, and any unclean overflow answer sends the
  batch to the primary and benches the overflow for two minutes. It must
  serve the same model as the primary.
- Offline fixed-candidate reranker benchmark with committed-artifact and model
  fingerprints, separate uncached/cache-hit trials, and bounded CPU-only execution.
  Planning reports `NOT_RUN`; see [benchmark methodology](docs/RERANK_BENCHMARK.md).
- MCP search intent presets, explicit ignored-parameter reporting, and complete
  JSON response budgets measured with cl100k_base. See [agent search](docs/AGENT_SEARCH.md).
- Push/PR validation reuses the release gate: Python tests, frontend build,
  package build, and clean-environment wheel install/import smoke.
- SQLite/FTS5 BM25 store, persisted backend selection, and explicit pickle
  conversion utility; compatibility default remains pickle. Existing corpora
  require parity verification before switching.
- Corpus routing descriptions and use/avoid guidance, plus search operations
  and corpus exploration views.

### Fixed
- The embedding cache row cap is a setting, `embed_cache_max_rows` in config.json,
  with `VECGREP_EMBED_CACHE_MAX_ROWS` still overriding it. It was read only
  from the environment, so a deployment that raised it in its service unit
  still had CLI runs, confirms and timers evict the shared cache back to the
  500,000-row default on their next write. The README listed a stale default.
- A confirmed edit, merge or delete reaches a document under whichever path
  spelling the corpus registry uses. A file registered before its directory
  became a symlink (for example a home directory moved to another disk) was
  indexed a second time on edit, leaving the old chunks searchable, and a
  confirmed delete removed the file but none of its chunks. Indexing a single
  file now retires any other registered spelling of that file, and confirmed
  deletes and merges de-index every spelling (`VecgrepService.delete_file`).
  `delete_source` still removes exactly the id it is given.
- The post-confirm retrievability check queried the frontmatter fence instead
  of the document body, so every confirm reported "post-write verify did not
  retrieve it" and a confirmed merge never deleted the absorbed documents.
- An edit carries the document's existing frontmatter forward (`created_at`,
  `tier`, `status`, `tags`, `source_kind`, an explicitly chosen title) instead
  of rebuilding it from proposal defaults. A title derived from the old heading
  is re-derived from the new one. `origin` records who produced the new text
  and confirm stamps `edited_at`.
- A document the chunker raises on is skipped and logged instead of ending the
  index run. `/api/index` lists it in `failed`, `vecgrep index` prints it, and
  the next run tries it again.
- Length-aware reranker batches separate long outliers without shortening inputs.
  Scores are restored to candidate order; performance gains require uncached
  measurements rather than HTTP samples containing score-cache hits.
- Corpus imports reject invalid names before storage writes, including hostile
  path forms.
- Embedding-cache flush timing tests use a controlled clock.
- Final search ordering uses the decayed score rather than display percentage;
  lexical-only confidence remains below semantic confidence bands.
- `doctor` reports duplicate source path aliases after directory moves or
  symlink changes, with exact groups in JSON and a failing health gate. Alias
  repair remains manual so canonical coverage is verified before deletion.
- MCP tool calls wait a bounded `VECGREP_MCP_GATE_WAIT_S` for the single tool
  slot and report `ToolBusyError` instead of queueing behind a stuck call.
  One tool body blocked on a saturated disk used to hold that slot with no
  ceiling, so every later MCP call hung while REST kept answering.
- Corpus search admission is bounded by `search_lock_timeout_s`. A corpus that
  cannot admit a search in time is reported rather than waited on, so an
  unscoped search answers from the corpora it can reach and a scoped one
  raises. A thread that already holds a corpus read lock now re-enters it
  instead of queueing behind a waiting writer, which never resolved.
- Unchanged-source indexing checks its content hash before chunk construction;
  recovery still verifies chunk counts and force-indexing still rebuilds.
- MCP response trimming protects top passages from preview-heavy tails and
  preserves fitting passages when a higher-ranked chunk exceeds the budget.
- Test settings are isolated by default; production paths and symlink escapes
  fail before settings can open persistent state.
- Reranker failures preserve hybrid ranking, model warming stays outside search,
  and bounded batches/token lengths limit prediction memory.
- Active BM25 bulk indexes cannot be evicted and reloaded from stale sidecars;
  interrupted bulk operations are repairable from Qdrant.
- OAuth registration, public-client metadata, callback CSP, proxy admission,
  and REST/MCP DNS-rebinding boundaries are covered by defensive tests.
- Protected write confirmation and metadata validation reject invalid changes.
- SQLite hybrid search rejects missing/inconsistent sidecars and propagates
  storage failures. Source mutation intents retire only after lexical commits;
  embedding migration rebuilds the selected backend from canonical payloads.
- Refresh-token scope requests fail closed; narrowed grants stay narrowed on
  successor refresh tokens, guarded by mirrored OAuth contract fixtures.

## [1.2.0] — 2026-07-09

Operational safety and self-hosting release. Search and ranking behavior are
unchanged.

### Added
- Verified whole-instance backups for embedded and server-mode Qdrant, with
  schema-versioned manifests, SHA-256 checksums, non-secret config, aliases,
  write documents, corpus metadata, and per-collection snapshots.
- Restore preflight, exact-ID confirmation, a cross-process mutation lock,
  automatic pre-restore safety backups, rollback on failure, BM25 rebuild, and
  post-restore registry/vector diagnostics.
- `vecgrep backup create|list|verify|restore`, `vecgrep init`, and
  `vecgrep serve --open`.
- Admin API and web controls for backup creation, verification, download,
  restore, and a persisted disabled-by-default daily/weekly scheduler.
  Scheduled retention never deletes manual or pre-restore backups.
- Tag-triggered GitHub release automation that tests Python, builds the React
  UI, produces wheel/sdist artifacts, and smoke-installs the wheel in a clean
  environment.

## [1.1.2] — 2026-07-09

### Added
- A loopback-safe administration API and web UI for non-secret runtime
  configuration. Values show default/file/environment provenance; environment
  overrides are read-only and secrets expose presence only.
- Atomic, validation-first `config.json` updates that preserve unknown keys,
  require typed confirmation before switching Qdrant, and report whether a
  service restart is required.
- Hot reload for embedding endpoints/models and default result count, including
  invalidation of API and MCP service caches. Existing corpus model pins are
  preserved and called out explicitly.

## [1.1.1] — 2026-07-09

Forward-only maintenance release collecting the work shipped after v1.1.0.

### Added
- **Retrieval controls**: relative time filters, negated filters, enriched
  speaker/bot/content filters, and a wider default budget of 10 full results
  plus up to 90 stubs. The enrichment and reindex runbooks document how to
  apply the new metadata to existing corpora.
- **Insight and OAuth administration tools**: related-item discovery,
  comparison, corpus statistics, extractive summaries, and an OAuth status /
  client-revocation surface.
- **`propose_edit` patch mode** (both MCP transports): pass `old_str` + `new_str`
  instead of `content` for a surgical str-replace of one *unique* span, instead
  of re-sending the whole body to fix one line. Non-unique or missing `old_str`
  is a hard error (never a silent mis-edit); body-only, frontmatter preserved.
  Reuses the exact proposal/confirm/corpus-gate path — no forked approval logic.
  `content` and `old_str` are mutually exclusive.
- **Integrity diagnostics**: `vecgrep doctor` reconciles the corpus registry
  with the active vector store and reports embedded/server split-brain
  configurations.

### Fixed
- Corpus imports restore all data required for an immediately searchable
  corpus instead of leaving gaps in the registry or BM25 state.
- Corpus chunk counts are read from Qdrant rather than maintained as a
  potentially drifting delta.

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

[Unreleased]: https://github.com/jeffbai996/vecgrep/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/jeffbai996/vecgrep/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/jeffbai996/vecgrep/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/jeffbai996/vecgrep/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/jeffbai996/vecgrep/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/jeffbai996/vecgrep/compare/v0.7.0...v1.0.0
[0.7.0]: https://github.com/jeffbai996/vecgrep/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/jeffbai996/vecgrep/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/jeffbai996/vecgrep/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/jeffbai996/vecgrep/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jeffbai996/vecgrep/releases/tag/v0.3.0
