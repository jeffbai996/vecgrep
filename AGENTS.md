# Agent guidance and scaling roadmap

## Scope and authority

This file governs work in this repository. The scaling roadmap below records
the agreed direction as of 2026-09-11; unchecked work is proposed, not shipped.
It supersedes the obsolete v0.2/v0.3 build order in CLAUDE.md, not the project's
privacy, compatibility, or test requirements. Consult README.md, CHANGELOG.md,
CONTRIBUTING.md, docs/DEVELOPING.md, and the implementation for current behavior.
Other candidates remain in docs/IDEAS.md.

Keep this public repository generic: no private transcripts, evaluation gold,
credentials, internal hostnames, or deployment-specific paths in committed files.
Operational observations belong in private deployment records. Preserve the
local-first default and stable CLI, HTTP, and MCP contracts.

## Objective

Retain an expanding archive while controlling the resident working set and the
cost of ordinary queries. Growth should primarily consume disk; a new message
should require work proportional to the changed content. Broad historical
investigations may spend more time and I/O through an explicit expansion path.
Do not promise unlimited data, fixed resources, perfect recall, and fixed latency.

Separate three responsibilities behind the existing vecgrep interface:

1. Evidence: canonical permitted content, versions, source references, and history.
2. Search: rebuildable lexical, metadata, and semantic indexes with bounded caches.
3. Working knowledge: current facts, decisions, and summaries linked to evidence.

The evidence layer must respect source privacy rules; it does not authorize
centralizing raw agent logs. Summaries are derived records, never substitutes for
original evidence or the sole route to historical details.

## Existing work to reuse

Read docs/STORAGE_RETRIEVAL_2026-08.md before proposing a storage or ranking change.
It documents payload carriers, disk-backed indexes, embedding cache improvements,
retrieval evaluations, and the distinction between quantization and disk savings.
Inspect current code and deployed settings because those measurements are dated.
Reuse vecgrep.eval and existing source-level quality metrics. Do not recreate the
evaluation harness or assume a database replacement is necessary.

## Ordered milestones and acceptance gates

### Execution checkpoint: 2026-09-14

Recheck deployed roots and current commits before repeating earlier experiments.
An upstream opt-in stable-segment normalizer and vecgrep multi-root watching now
exist; Desktop producer rollout is separate from local-recorder rollout. Source
hash comparisons must audit the actual watched roots, not retired output trees.
The September 11 append timings predate later index/journal optimizations.

Before running an expensive retrieval suite, validate that positive gold cases
still resolve to registered evidence. Missing expectations are an evaluation
health problem, not automatically ranking failure. Never repair gold by accepting
whatever search currently returns. Keep accepted-baseline and previous-run
comparisons distinct, version/fingerprint evaluation inputs, and detect cumulative
degradation. Keep private expectations and operational reports outside this repo.

Use bounded diagnostics first: source/hash preflight, saved-report alert tests,
focused append cases. Run a broad retrieval or scale experiment only with a stated
hypothesis, a bounded work budget, and a decision its result will change.

### 0. Restore reliable operation and establish the baseline

- [ ] Verify source sync through indexing and retrieval; a running watcher or
  copied file does not establish freshness. Repair recurring failures and
  reconcile missed sources with bounded retries and checkpoints.
- [ ] Verify the configured host-pressure producer, consumer, freshness, and
  missing-signal behavior. Document the actual protection rather than relying
  on comments or an absent signal file.
- [ ] Record per-corpus sources, unique chunks, vectors, database/WAL bytes,
  ingest backlog, embedding work per update, and query latency p50/p95.
- [ ] Account separately for API/worker RSS, anonymous memory, file cache,
  vector storage, embedding services, and host headroom.

Gate: demonstrate successful new-content retrieval and recovery of missed work;
save a reproducible private capacity baseline before changing storage defaults.

### 1. Stable incremental evidence and ingestion

Start with the measured append amplification in
docs/APPEND_COST_2026-09.md. Existing embedding-cache reuse does not eliminate
whole-source replacement. Keep completed segment metadata immutable: a changing
total-part count must not invalidate earlier segments. Treat source-path changes
as a migration with stale-source cleanup, not just a new naming convention.

- [ ] Define stable source, segment, version, and occurrence identifiers. Separate
  identity from paths, chunk positions, content hashes, and embedding identity.
- [ ] Seal completed conversation segments and retain a small mutable tail.
  Use bounded segments that do not shift when a new turn arrives. Determine
  adapter versus upstream-normalizer ownership before changing the contract.
- [ ] Reuse embeddings for identical content under the same model identity;
  retain source occurrence, timestamps, permissions, and provenance separately.
  Do not merge merely similar discussions or expose data across access boundaries.
- [ ] Handle edits, archival moves, deletions, crashes, and retries explicitly.
  Delete shared content only after its permitted references are gone; propagate
  invalidation to all derived indexes and summaries.

Gate: appending a turn to a long synthetic session leaves sealed segment IDs and
embeddings unchanged. Cost scales with changed content; retry is idempotent;
edits/deletions remove stale results without breaking surviving references.

### 2. Bounded retrieval over a disk-backed archive

- [ ] Audit actual lexical/vector storage and query access patterns, including
  duplicated text, cache budgets, full-corpus materialization, and write amplification.
- [ ] Budget per-corpus retrieval, merged candidates, reranker input tokens,
  query concurrency, ingestion concurrency, and queue lengths. Define visible
  timeout/degradation behavior and a separate budget for deeper searches.
- [ ] Evaluate existing on-disk storage, supported datatypes, and quantization on
  isolated collections before selecting changes. Quantization may add a second
  vector representation; do not advertise it as automatic disk reduction.
- [ ] Preserve lexical access to old exact strings, names, paths, and multilingual
  text. Expose search scope and expansion; do not silently exclude old corpora.

Gate: before/after quality and resource results, including negative queries and
concurrent ingestion. No storage or ranking change ships without an evaluation row.

### 3. Semantic directory and evidence expansion

- [ ] Add versioned session/segment descriptions, topics, entities, and evidence
  links as a derived index. Update only affected descriptions.
- [ ] Combine directory retrieval with lexical/metadata retrieval and detailed
  semantic fallback. Routing is an optimization, not evidence of irrelevance.
- [ ] Expand selected hits to original passages under a text budget. Preserve
  citations and disclose scope, truncation, and incomplete searches.

Gate: test obscure old facts absent from summaries, ambiguous project names,
cross-corpus questions, and routing misses against the unrestricted baseline.

### 4. Evidence-backed current knowledge

- [ ] Represent current state, decisions, preferences, and unresolved questions
  with evidence references, verification status, time validity, and supersession.
- [ ] Preserve contradictions and change history. Recompute or invalidate claims
  when evidence changes; never promote an unsupported summary into a fact.
- [ ] Reuse an existing knowledge-store owner where appropriate; avoid creating
  a competing source of truth inside the search engine.

Gate: a changed decision updates the current answer while retaining historical
retrieval; conflicting and deleted evidence are handled explicitly.

### 5. Capacity growth and partitioning

- [ ] Extend the existing evaluation harness with representative 1x/3x/10x data,
  cold/warm-cache runs, constrained memory, concurrent search and ingestion,
  update amplification, and restart/recovery measurements.
- [ ] Use diverse synthetic or permitted private material, not identical copies
  that inflate deduplication gains or make retrieval artificially easy.
- [ ] Set numeric RAM, latency, freshness, and recall budgets from the baseline.
  A 10x growth test must retain those agreed ordinary-query budgets; document
  any degradation or extra work required for exhaustive historical searches.
- [ ] Add partitions, storage tiers, or nodes only after a measured bottleneck.
  Keep routing and global candidate budgets bounded; do not fan out unlimited
  work as partition count rises. Preserve the external vecgrep interface.

Gate: publish generic results and keep private fixtures outside the repository.
Select an engine migration only if the existing backend fails the agreed targets.

## Execution rules

Work in small independently reviewable changes. Treat this roadmap as direction,
not permission for an unrequested migration or broad service changes. Keep
deployment repairs distinct from architecture work and preserve unrelated edits.
Use isolated VECGREP_HOME/evaluation collections for tests; never drop production
caches or rebuild live corpora to simulate scale. Establish rollback and recovery
for migrations before deploying them. Mark milestones complete only with evidence.
