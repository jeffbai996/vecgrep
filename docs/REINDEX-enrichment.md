# Re-index for chunk enrichment (speaker:/bot:/has: filters)

The v1.2 filters `speaker:`, `author:`, `bot:` and `has:code|table|link` read
per-chunk metadata (`speakers`, `has_bot_speaker`, `has_code`, `has_table`,
`has_link`) that is attached at **index time** (`ingestion/enrich.py`, wired
into the payload build in `service.py`). Existing corpora indexed before v1.2
don't carry it — on those chunks the new filters **fail closed** (a `speaker:`
query returns nothing rather than everything), which is correct but means the
filters only become useful after a re-index.

## What a re-index costs

Enrichment changes only payload metadata, not chunk text — so every embedding
is unchanged and the **embed cache** absorbs the vector cost. A forced
re-index of the `chats` corpus (~21k chunks) is therefore mostly re-chunk +
cache-hit + upsert, not 21k fresh embeddings. Still: it rewrites every point
in Qdrant and the BM25 store for the corpus.

## Staged procedure (do NOT run against the live corpus unprompted)

1. **Smoke on a scratch corpus first** (any box, new code):

   ```bash
   vecgrep index tests/fixtures/transcripts --corpus enrich-smoke --include '*.md'
   vecgrep search "relay-service" --corpus enrich-smoke --filter "speaker:alice"
   vecgrep search "relay-service" --corpus enrich-smoke --filter "bot:true"
   vecgrep corpora delete enrich-smoke   # clean up
   ```

   Non-empty filtered results = enrichment landing end-to-end.

2. **Live re-index** (owner's go required — vecgrep-serve keeps running; the
   indexer's idempotent per-source wipe handles the swap):

   ```bash
   time vecgrep index <your-transcripts-dir> \
       --corpus chats --include '*.md' --force
   ```

   `--force` bypasses the source-hash skip so every doc re-chunks and its
   payloads rebuild with enrichment. Watch the embed cache hit-rate in the
   log — it should be ~100%.

3. **Verify** with a real query:

   ```bash
   vecgrep search "portfolio margin" --corpus chats --filter "speaker:<owner-handle>" --filter "after:7d"
   ```

4. Other corpora can re-index the
   same way whenever convenient — the filters just fail closed on them until
   then. Non-transcript corpora gain only `has:` (no speakers to extract),
   which is expected.

## Rollback

Enrichment is additive payload metadata. Old code reading new payloads
ignores the extra keys; new code reading old payloads fails the new filters
closed. No schema migration, no rollback step — re-indexing again with any
version simply rewrites the payloads that version produces.
