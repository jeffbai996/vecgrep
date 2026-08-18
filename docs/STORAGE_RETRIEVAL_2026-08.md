# Storage and retrieval at scale — findings, 2026-08

Round 2 of the storage/retrieval work. Round 1 fixed payload duplication
(the carrier layout, `2dade45` / `e8b38ea`) and connection pooling. This
round built the instrument that was missing — a side-by-side eval harness —
and used it to price every remaining trade. Numbers below are from a real
deployment (a ~120k-point transcript corpus, a ~30k-point code corpus and a
~2k-point notes corpus, all `bge-m3` 1024-dim), measured on eval copies,
never on the live corpora.

## The harness (`vecgrep.eval`)

    python -m vecgrep.eval build --from <live> --to eval-<name> [--chunker X] [--datatype float16]
    python -m vecgrep.eval run   --gold $VECGREP_HOME/eval/gold.json --config configs.json
    python -m vecgrep.eval report reports/*.json --cases
    python -m vecgrep.eval drop

One row per configuration: hit@1/3/5/10, MRR, P@3, spread(1v3),
negative-query false-positive rate, latency p50/p95, points, qdrant MB,
BM25 MB. Scoring is source-level (a document shattered into 400 chunks
gets one shot). Gold lives outside the repo; the format is in
`vecgrep/eval/gold.py`. **No storage or ranking change ships without a
before/after row.**

## Results — retrieval at current scale (M2)

Gold: 101 cases (44 transcript, 41 notes, 8 code, 8 negatives). Source-level
hit@k. "Round 1" is the state of the world before this work; "round 2" is
after two fixes the harness itself exposed on day one:

1. **Gold rot.** 12 of the notes cases pointed at memory ids whose content
   had since been promoted into other documents; the system was returning
   the right document and the gold called it a miss. Re-keyed by content.
   (This is the second time this corpus's gold rotted; the harness now
   makes it visible in the per-case dump instead of in a summary number.)
2. **CJK was invisible to BM25.** The tokenizer was `[A-Za-z]+|\d+`. A
   Chinese phrase present verbatim in exactly one transcript could not be
   found lexically at all. Now character bigrams (`9c6047d`); existing
   sidecars need `vecgrep bm25 rebuild <corpus>` to pick it up.

| config (round 2) | hit@1 | hit@3 | hit@5 | hit@10 | MRR | neg FP | p50 ms |
|---|---|---|---|---|---|---|---|
| hybrid (defaults: BM25 w1.5, RRF k60, decay 45d) | 48.4 | 78.5 | 83.9 | — | .635 | 87.5% | 464 |
| vector only | 54.8 | 76.3 | 78.5 | — | .664 | 0% | 44 |
| bm25 only | 46.2 | 64.5 | 76.3 | — | .581 | 100% | 331 |
| hybrid, BM25 w0.5 | 49.5 | 78.5 | 87.1 | — | .655 | 62.5% | 369 |
| hybrid, decay floor 0.5 | 49.5 | 79.6 | 84.9 | — | .648 | 87.5% | 358 |
| hybrid, w0.5 + floor 0.5 | 50.5 | 79.6 | 87.1 | — | .659 | 62.5% | 356 |
| hybrid + rerank | 52.1 | 66.0 | 71.3 | 78.7 | .607 | 0% | 662 |
| hybrid + rerank, floor 0.5 | 55.9 | 76.3 | 87.1 | — | .681 | 0% | 631 |

Per corpus (hit@1 / hit@3 / hit@5, MRR):

| config | transcripts (44) | notes (41) | code (8) |
|---|---|---|---|
| hybrid defaults | 59 / 82 / 86, .71 | 37 / 76 / 80, .56 | 50 / 75 / 88, .63 |
| hybrid, w0.5 | 59 / 73 / 84, .70 | 39 / **85 / 90**, .61 | 50 / 75 / 88 |
| hybrid, floor 0.5 | 61 / **84 / 89**, .74 | 37 / 76 / 80, .56 | 50 / 75 / 88 |
| hybrid + rerank | 59 / 75 / 82, .70 | **56** / 78 / 88, .68 | 62 / 75 / 88, .70 |

What the table says:

- **The #200 fear ("hit@3 collapses as the corpus grows") did not
  materialise at 122k points.** Transcript hit@3 is 82-84% under hybrid at
  full scale; notes hit@3 is 76-85%. The earlier 44% number was measured
  against a rotted gold on a stub-embedded harness.
- **The right BM25 weight is per corpus, not per install.** Notes (short
  semantic memories) peak at 0.5; transcripts (names, dates, exact phrases,
  CJK) need 1.5. `vecgrep corpora bm25-weight <corpus> --factor` now pins
  it per corpus (`d8fd7f4`); the global env stays as the default.
- **Recency decay must have a floor.** Unfloored, a 100-day-old exact match
  scores 0.21x and loses to a vague recent one (a Chinese-language query
  went from rank 1 vector-only to unranked under decay). `DECAY_FLOOR=0.5`
  (`99e4595`) lifts transcript hit@3 82 -> 84 and hit@5 86 -> 89 at no cost
  elsewhere. 14-day half-life is catastrophic (hit@3 38%); 45 vs none is a
  wash on this recency-skewed gold; 45 stays.
- **The reranker is not a "quality wash" at this scale — but it is not a
  free win either.** It roughly doubles notes hit@1 (37 -> 56) and is the
  ONLY configuration whose negatives calibrate: without it 6-8 of 8
  should-match-nothing queries return a top hit above the 60% recall floor
  (BM25-only hits are rank-scaled to ~90% for display, so any lexical
  overlap reads as confident), with it every negative sits at 22-27%. It
  costs +250 ms and it demotes a few right answers the pool already held
  (notes hit@3 78 vs 85 for w0.5): `bge-reranker-base` is weak on long
  technical chunks. Keep it ON for the per-turn recall hook (calibration
  is what that consumer needs); a stronger reranker is the next thing to
  measure, not more RRF tuning.
- RRF k (20/60/120) is a wash. Cosine floor untested (left at default).
- 8 negatives is a small sample; the FP rate is directional.

Adopted: `DECAY_FLOOR` defaults to 0.5; per-corpus BM25 weight pinned on
the live install (transcripts 1.5, notes 0.5); live sidecars rebuilt for
CJK. Rerank stays opt-in on the API and on for the recall hook.

## Segment overhead — decided, do not re-ask

The "21% segment overhead" in the round-1 measurement was WAL + id trackers +
the payload index, not something an optimizer setting fixes:

    vector_storage   475 MB   (122k x 1024 x float32 = 500 MB; this IS the vectors)
    payload_storage  191 MB
    payload_index     11 MB
    vector_index       7 MB   (HNSW, on disk)
    id_tracker        ~4 MB
    wal               48 MB   (fixed-size ring, ~32 MB segments)

Two segments, status green. There is nothing here to reclaim by touching
`optimizer_config`; the only levers on this collection are the vector
datatype (float16 halves the 475 MB) and the chunk count. **Answer: no.**
Qdrant RSS is page cache and `on_disk` is already true (round-1 rule, still
correct); on-disk segment overhead is a rounding error. Stop measuring it.

## Quantization is not a disk lever

Qdrant keeps the original vectors when scalar/binary quantization is on;
the quantized copy is *additional* and exists to make the search pass
cheaper in RAM/latency. `int8` therefore cannot be "a straight 4x on the
dominant term" of disk. The disk lever is `datatype=float16` (exact 2x on
`vector_storage`), which the store now supports at collection creation and
which the harness measures.

## BM25 sidecar write amplification (fixed, `214408b`)

`BM25Store.upsert` re-pickled the whole corpus index after every source. A
directory index of N sources wrote the growing pickle N times: on the
transcript corpus that is ~1,000 writes of a file growing to 250 MB — on
the order of 125 GB of writes for one rebuild — and it is why a warm-cache
side-by-side build took 45 minutes. Bulk mode persists once per index call
and leaves a `.dirty` marker while the in-memory index is ahead of the file;
a crash mid-bulk is repaired from qdrant on the next service start.

## Embed cache (`embed_cache.db`)

Rows were JSON text (~13.8 KB per 1024-dim vector). 292k rows = 4.0 GB, of
which ~96k rows referenced chunks no live corpus held. Now: float32 blobs
(4 KB), `vecgrep cache sweep` (keep-set derived from qdrant per embedding
identity), `vecgrep cache compact` (rewrite legacy rows + VACUUM).

## Round 3 (2026-08-18) — reranker, negatives, cache, float16

Negatives grown 8 -> 26 (out-of-domain EN + CJK across all three corpora),
gold now 119 cases. Same eval-* corpora as round 2, `DECAY_FLOOR` 0.5 and
per-corpus BM25 weights live. Latencies were measured on a busy box (a cache
compaction and a corpus build overlapped the run) so read them relative to
each other, not against the round-2 absolutes.

| config | hit@1 | hit@3 | hit@5 | hit@10 | MRR | neg FP (26) | p50 ms |
|---|---|---|---|---|---|---|---|
| hybrid (no rerank) | 49.5 | 79.6 | 84.9 | 92.5 | .648 | 92.3% | 1028 |
| + bge-reranker-base | 55.9 | 76.3 | 87.1 | 91.4 | .681 | 11.5% | 1282 |
| + bge-reranker-v2-m3 | 55.9 | **82.8** | **89.2** | 92.5 | **.702** | **3.8%** | 2425 |
| + bge-reranker-large | **57.0** | 79.6 | 89.2 | **93.5** | .697 | 3.8% | 4518 |

- **v2-m3 is the reranker round 2 asked for.** It keeps base's hit@1 gain
  and stops paying for it at hit@3 (76 -> 83, above the unreranked pool),
  and 1 in 26 negatives leaks vs 3 in 26 for base and 24 in 26 unreranked.
  `large` buys +1 hit@1 for double the latency; not worth it.
- **Negative FP is now measured on 26 cases, not 8.** The round-2 "directional"
  caveat is retired: unreranked hybrid really does surface a confident top
  hit for almost every off-topic query, and a reranker really does fix it.
- **Not adopted as default — blocked by the serve path, not the numbers.**
  Rerank runs synchronously in the request handler. The first v2-m3
  load+predict inside `vecgrep-serve` blocked the event loop for 5+ minutes
  (20 GB box, 5.2 GB RSS + swap already), `squad-watchdog` (strikes=1) killed
  and restarted the server mid-request, and a hook that reranks by default
  would re-trigger that on every restart. `DEFAULT_RERANKER` is now
  env-overridable (`VECGREP_RERANKER`) and stays `bge-reranker-base` until the
  model is warmed at startup or predicted off the event loop. That is the
  next reranker task; measuring is done.
- **Embed cache sweep + compact ran on live:** 335,161 -> 226,970 rows,
  `embed_cache.db` 4.45 GB -> 1.07 GB, integrity ok, search unaffected.
- **float16:** see the row below this section once `eval-chats-f16` finishes.
