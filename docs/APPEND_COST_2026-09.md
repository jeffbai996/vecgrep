# Transcript append cost: structural baseline

Measured 2026-09-11 against vecgrep commit f9892b2 and upstream transcript
normalizer version 7. This is an isolated synthetic experiment, not a production
latency benchmark, retrieval-quality evaluation, or capacity estimate.

## Method

Use a temporary VECGREP_HOME, the real VecgrepService index path, persistent local
Qdrant and SQLite BM25, sentence_window chunking, and CachedBackend wrapping a
deterministic 16-dimensional fake embedding backend. No embedding server or
production collection is used. Normalize synthetic Codex event_msg records,
index once, append one 573-character message, normalize and index again, then
index unchanged input a third time. Count embedding-cache misses at the wrapped
backend. The synthetic messages repeat sentences with unique turn numbers;
their high cache reuse must not be generalized to arbitrary real conversations.

Each scenario is measured once. Index times include local transaction and
journaling work; they exclude real model cost and use local embedded Qdrant,
not a production server configuration. Chunk counts are returned by the real
index path, which upserts all chunks of each changed source; they are not disk
bytes or physical storage write counts.

## Results

| Prior turns | Parts before -> after | Changed parts | Markdown characters in changed parts | Chunks upserted | Embedding cache misses | Index seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 10 | 1 -> 1 | 1 | 6,819 | 51 | 2 | 0.357 |
| 150 | 1 -> 1 | 1 | 91,939 | 681 | 1 | 5.696 |
| 492 | 3 -> 4 | 4 | 300,336 | 2,223 | 5 | 16.683 |

All three unchanged follow-up index calls returned zero documents and zero
chunks written. The respective skipped-source counts were 1, 1, and 4.

## Interpretation

The embedding cache already prevents much repeated model work. However,
VecgrepService._index_locked deletes and replaces all chunks for a changed
source, journals the prior points, and updates the lexical index and registry.
The source hash check also occurs after chunking; unchanged documents still
incur parsing/chunking work before they are skipped.

The upstream normalizer uses bounded parts, but each carries the total part
count. Crossing a boundary changes that metadata in all old parts. Moving from
one part to multiple parts also changes the first part's filename. These are
separate causes of amplification from embedding computation.

## Next implementation slice

1. In the upstream normalizer, introduce stable segment identifiers from the
   first segment and immutable metadata for sealed segments. Keep mutable total
   counts in a manifest excluded from the search corpus. Bound the active tail.
2. Test ordinary append, first split, later splits, oversized turns, edits,
   truncation, source moves/deletion, retry, and interrupted publication. Prove
   unchanged sealed segment bytes and IDs, not merely identical embeddings.
3. Provide a versioned migration and explicit removal of superseded indexed
   source IDs. Source-file deletion alone is insufficient proof of index cleanup.
4. In vecgrep, investigate moving the unchanged-source hash check ahead of
   chunking without breaking recovery/count validation. Benchmark independently.
5. Repeat this structural baseline with varied synthetic text and representative
   permitted private sessions; add actual write metrics and before/after retrieval
   quality using the existing evaluation harness before changing storage defaults.

This baseline supports prioritizing incremental ingestion. It does not establish
fixed-memory operation, unlimited scaling, or a need to replace the database.
