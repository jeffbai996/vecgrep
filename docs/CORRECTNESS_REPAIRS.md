# Retrieval and indexing correctness repairs

Base: `68e1a93842e9fef2ab26ee863b91ca7bb211eb59`.

The 24 original reproductions produced 21 assertion failures and three invalid
fixture failures. Reproductions are hypotheses; the disposition below distinguishes
production defects from fixture mistakes. All validation uses isolated local state
and deterministic embedding doubles.

| Finding | Disposition | Regression file |
|---|---|---|
| F-CHUNK-01 | Reproduced, fixed: expand short sentence windows rather than discard their content; split CJK punctuation without requiring spaces; preserve source offsets. | `test_area1_chunking.py` |
| F-CHUNK-02 | Reproduced, fixed: keep preamble and meaningful short turns with speaker context, exclude empty/format-only windows. | `test_area1_chunking.py`, `test_correctness_repairs.py` |
| F-CHUNK-03 | Invalid fixture and unsupported claim: constructor parameters describe a different implementation. A corrected supported window configuration passes unchanged main; headings are not synthesized on every chunk. | `test_area1_chunking.py` |
| F-CHUNK-04 | Invalid fixture corrected, then reproduced and fixed: token boundaries expand to complete Unicode codepoints with exact source spans. A boundary character may overlap adjacent chunks. | `test_area1_chunking.py` |
| F-CHUNK-05 | Reproduced, fixed: code without blank lines obeys the chunk-size bound. | `test_area1_chunking.py` |
| F-INDEX-01 | Reproduced for both stores, fixed: upsert replaces chunk identity; unchanged IDs and metadata survive; incremental lexical results match a fresh fixture index. | `test_area2_indexing.py`, `test_correctness_repairs.py` |
| F-INDEX-02 | Reproduced, fixed: a successful directory ingestion removes absent local files under that root through the normal deletion journal. Existing unreadable or excluded files and sources outside the root remain. | `test_area2_indexing.py` |
| F-INDEX-03 | Reproduced, fixed: failed bulk persistence retains the recovery marker. | `test_area2_indexing.py` |
| F-INDEX-04 | Invalid fixture corrected: proper migration arguments, resolved model identity, actual temporary collection name and one-shot failure after deletion. Corrected fixture reproduced data loss. Fixed with write admission and durable, resumable promotion intent; staged replacement survives incomplete copying. | `test_area2_indexing.py` |
| F-INDEX-05 | Reproduced, fixed: cache bypass resets in `finally`, including indexing exceptions. | `test_area2_indexing.py` |
| F-RETR-01 | Reproduced, fixed: storage errors raise a typed exception; generic embedder outages and query-vector failures raise instead of yielding zeros. Dimension probing uses the same safe query path. Hybrid lexical fallback carries a warning, zero hits carry none, and recovery clears degradation. HTTP/MCP dependency failures omit exception details. | `test_area3_retrieval.py`, `test_correctness_repairs.py` |
| F-RETR-02 | Reproduced, fixed: active/legacy status eligibility reaches Qdrant before its candidate cap. Both BM25 backends apply eligibility before candidate selection. Superseded-heavy fixtures cover fewer-than-k, source/status filters and full/stub budget expansion. | `test_area3_retrieval.py`, `test_correctness_repairs.py` |
| F-RETR-03 | Reproduced, fixed: corpus rank weights apply in vector and lexical modes. | `test_area3_retrieval.py` |
| F-RETR-04 | Reproduced, fixed: incident timestamps normalize single-digit hours before range selection. | `test_area3_retrieval.py` |
| F-RETR-05 | Reproduced, fixed: related results exclude superseded and expired records. | `test_area3_retrieval.py` |
| F-API-01 | Reproduced, fixed: direct edits patch the body already read through the validated no-follow target. | `test_area4_mcp_http.py` |
| F-API-02 | Reproduced, fixed: comparison validates parameters and maps dependency errors to 503. Its outage test explicitly requests vector mode because hybrid has a supported lexical fallback. | `test_area4_mcp_http.py` |
| F-API-03 | Partly already fixed: negative full_k was already rejected. Fixed remaining top_k, group-count and padding validation. | `test_area4_mcp_http.py` |
| F-API-04 | Reproduced, fixed: batch transport failures propagate without retrying every item. Poison-input ingestion fallback remains; query embeddings never accept its zero vector. | `test_area4_mcp_http.py`, existing Ollama/receipt tests |
| F-API-05 | Reproduced, fixed: backup jobs borrow the active embedded service instead of acquiring its storage lock twice; borrowed backup handles remain open. Restore releases and invalidates borrowed runtime state before reopening. | `test_area4_mcp_http.py`, existing backup tests |

## Deterministic retrieval comparison

`tests/test_retrieval_fixture.py` reuses `vecgrep.eval` scoring with four synthetic
sources (prose, a technical identifier and CJK) plus a negative query. This checks
routing, indexing and ranking contracts, not real embedding-model quality.

| Store/mode | Baseline hit@1 / MRR / negative FP | Repaired hit@1 / MRR / negative FP |
|---|---|---|
| Pickle / lexical | 100% / 1.0 / 0% | 100% / 1.0 / 0% |
| Pickle / vector | 100% / 1.0 / 0% | 100% / 1.0 / 0% |
| Pickle / hybrid | 100% / 1.0 / 0% | 100% / 1.0 / 0% |
| SQLite / lexical | 100% / 1.0 / 0% | 100% / 1.0 / 0% |
| SQLite / vector | 100% / 1.0 / 0% | 100% / 1.0 / 0% |
| SQLite / hybrid | 100% / 1.0 / 0% | 100% / 1.0 / 0% |

The context-expansion test now selects its known target with lexical search. Its
old prefix-only embedding double collapsed many repeated sentences to the same
vector, making the target assertion depend on vector tie ordering. The test still
asserts target content, wider context and exact source length. Transport receipt
tests now require propagated failures while retaining their reason-code assertions.

Validation: full suite 1,537 passed, 10 skipped (host-side weekly-evaluation script
absent), two dependency warnings. Existing lint debt prevents a clean lint gate:
416 findings on baseline, 414 after repair, none introduced. Package builds and
isolated wheel imports passed. No live corpus, reindex, migration, model inference,
service restart or deployment was used to validate these source changes.
