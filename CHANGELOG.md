# Changelog

All notable changes to vecgrep are documented here. Versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0: minor = features,
patch = fixes). The version is a single source of truth in
`vecgrep/__init__.py`; `pyproject.toml` reads it dynamically.

## [Unreleased]

Work toward **v1.0** — a stable, usable release. Focus (from real-world
personal-memory / Discord-transcript usage): better result assembly over the
existing hybrid-search core — source-span dedup/MMR, date & path & time
filters, timeline reconstruction, clearer score output, and alias/entity
expansion. See `docs/superpowers/specs/` for the design.

### Added
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
