# Deferred behavior changes

**Unresolved lint diagnostics: none.** No source fixes are deferred from this task branch because the upstream cleanup was already present.

The constructs below were deliberately retained by upstream. They remain possible hardening work, not a claim that blanket lint suppression proves each failure path safe. Narrowing an exception boundary, changing an exception type, replacing a framework default, or changing reentrant resource ownership requires separate contract-specific tests. This verification adds no suppression.

- `tests/fixtures/code_symbols/broken.py:5`: two syntax diagnostics remain in intentional malformed input. The upstream Ruff exclusion keeps parser-failure coverage intact.

## Targeted upstream annotations (85 locations)

Each entry below preserves the upstream one-line reason. Existing full-suite and retrieval comparisons passed; they do not establish exhaustive coverage of every exceptional condition.

| Location | Rules | Reason |
| --- | --- | --- |
| `scripts/rerank_benchmark.py:87` | TRY004 | Malformed benchmark content uses one ValueError contract. |
| `scripts/rerank_benchmark.py:97` | TRY004 | Malformed benchmark content uses one ValueError contract. |
| `tests/conftest.py:111` | BLE001, S110 | Teardown must not mask the test result if a backend close fails. |
| `tests/eval_harness.py:244` | BLE001, S110 | Evaluation teardown must not replace its collected result. |
| `tests/test_bm25_bulk_persist.py:99` | BLE001, S110 | Cleanup of a deliberately interrupted store must not mask the assertion. |
| `tests/test_search_text_write_paths.py:70` | BLE001, S110 | The following assertions verify recovery after the injected indexing failure. |
| `tests/test_write_source_alias.py:79` | BLE001, S110 | Test cleanup must not mask the alias assertions. |
| `vecgrep/backend/api/admin.py:128` | BLE001, S110 | Reset must release stale service references even if a backend cannot close. |
| `vecgrep/backend/api/routes.py:402` | B008 | FastAPI reads Query metadata at route construction. |
| `vecgrep/backend/auth/google_bearers.py:118` | BLE001 | Registry failures retain the logged auth-boundary error contract. |
| `vecgrep/backend/auth/google_bearers.py:127` | BLE001 | Registry failures retain the logged auth-boundary error contract. |
| `vecgrep/backend/auth/google_bearers.py:139` | BLE001 | Registry failures retain the logged auth-boundary error contract. |
| `vecgrep/backend/auth/google_bearers.py:169` | BLE001 | Registry failures retain the logged auth-boundary error contract. |
| `vecgrep/backend/auth/store.py:166` | BLE001 | A corrupt state snapshot falls back to owner reapproval rather than crashing startup. |
| `vecgrep/backend/backup.py:151` | BLE001 | Backup rollback and cleanup must contain arbitrary storage-provider failures. |
| `vecgrep/backend/backup.py:310` | BLE001, S110 | Backup rollback and cleanup must contain arbitrary storage-provider failures. |
| `vecgrep/backend/backup.py:373` | BLE001 | Backup rollback and cleanup must contain arbitrary storage-provider failures. |
| `vecgrep/backend/backup.py:400` | BLE001, S110 | Backup rollback and cleanup must contain arbitrary storage-provider failures. |
| `vecgrep/backend/backup.py:449` | BLE001, S110 | Backup rollback and cleanup must contain arbitrary storage-provider failures. |
| `vecgrep/backend/backup.py:505` | BLE001, S112 | Backup rollback and cleanup must contain arbitrary storage-provider failures. |
| `vecgrep/backend/ingestion/adapters/pdf.py:40` | BLE001 | One unreadable PDF page must not discard text from the other pages. |
| `vecgrep/backend/ingestion/visual_worker.py:34` | BLE001 | An individual enrichment failure must be recorded without stopping the consumer. |
| `vecgrep/backend/main.py:63` | BLE001 | Optional transport, warm-up and background cleanup failures are reported without stopping the app. pragma: no cover - defensive |
| `vecgrep/backend/main.py:144` | BLE001 | Optional transport, warm-up and background cleanup failures are reported without stopping the app. |
| `vecgrep/backend/main.py:165` | BLE001 | Optional transport, warm-up and background cleanup failures are reported without stopping the app. |
| `vecgrep/backend/main.py:186` | BLE001 | Optional transport, warm-up and background cleanup failures are reported without stopping the app. |
| `vecgrep/backend/main.py:235` | BLE001 | Optional transport, warm-up and background cleanup failures are reported without stopping the app. pragma: no cover - defensive |
| `vecgrep/backend/mutation.py:204` | SIM115 | Reentrant lock state owns the handle until the existing finally closes it. |
| `vecgrep/backend/rerank.py:221` | BLE001, S110 | External model and process failures cross this worker boundary as explicit error results. |
| `vecgrep/backend/rerank.py:334` | BLE001 | External model and process failures cross this worker boundary as explicit error results. |
| `vecgrep/backend/rerank.py:353` | BLE001 | External model and process failures cross this worker boundary as explicit error results. |
| `vecgrep/backend/rerank.py:563` | BLE001 | External model and process failures cross this worker boundary as explicit error results. |
| `vecgrep/backend/rerank.py:728` | BLE001 | External model and process failures cross this worker boundary as explicit error results. |
| `vecgrep/backend/service.py:226` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:312` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:635` | TRY004 | Malformed persisted chunk data retains the RuntimeError contract. |
| `vecgrep/backend/service.py:1033` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:1090` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:1336` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:1515` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:1525` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:1596` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:1991` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:2012` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:2039` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:2461` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. never let warming break a search |
| `vecgrep/backend/service.py:2984` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. bookkeeping only -- never block shutdown |
| `vecgrep/backend/service.py:3189` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:3210` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:3941` | BLE001, S110 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:4251` | BLE001, S112 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:4255` | BLE001, S112 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/store/bm25_store.py:274` | BLE001 | A corrupt derived index is rebuilt from canonical source data. |
| `vecgrep/backend/store/qdrant_store.py:166` | BLE001, S110 | The storage adapter retains its existing missing-item and optional-index error contract. |
| `vecgrep/backend/store/qdrant_store.py:254` | BLE001 | The storage adapter retains its existing missing-item and optional-index error contract. |
| `vecgrep/backend/store/qdrant_store.py:354` | TRY004 | Unsupported backend snapshot shape retains the RuntimeError contract. |
| `vecgrep/backend/store/qdrant_store.py:468` | BLE001 | The storage adapter retains its existing missing-item and optional-index error contract. |
| `vecgrep/backend/store/qdrant_store.py:487` | BLE001 | The storage adapter retains its existing missing-item and optional-index error contract. |
| `vecgrep/backend/store/qdrant_store.py:510` | BLE001 | The storage adapter retains its existing missing-item and optional-index error contract. |
| `vecgrep/backend/write/confirm.py:67` | BLE001 | Write failures become explicit confirmation results instead of losing the user operation. |
| `vecgrep/backend/write/confirm.py:143` | BLE001 | Write failures become explicit confirmation results instead of losing the user operation. |
| `vecgrep/backend/write/confirm.py:282` | BLE001 | Write failures become explicit confirmation results instead of losing the user operation. |
| `vecgrep/backend/write/confirm.py:349` | BLE001 | Write failures become explicit confirmation results instead of losing the user operation. |
| `vecgrep/backend/write/confirm.py:373` | BLE001 | Write failures become explicit confirmation results instead of losing the user operation. |
| `vecgrep/cli/main.py:68` | BLE001 | Adapter failures become CLI errors while independent watched sources remain active. |
| `vecgrep/cli/main.py:1155` | BLE001 | Adapter failures become CLI errors while independent watched sources remain active. |
| `vecgrep/cli/main.py:1194` | BLE001 | Adapter failures become CLI errors while independent watched sources remain active. |
| `vecgrep/cli/main.py:1271` | BLE001 | Adapter failures become CLI errors while independent watched sources remain active. |
| `vecgrep/cli/main.py:1571` | BLE001 | Adapter failures become CLI errors while independent watched sources remain active. |
| `vecgrep/cli/main.py:1845` | BLE001, S112 | Adapter failures become CLI errors while independent watched sources remain active. |
| `vecgrep/eval/harness.py:22` | PLC0414 | Preserve the existing public re-export. |
| `vecgrep/eval/harness.py:185` | BLE001 | Failed readiness and warm-up attempts remain isolated from measured evaluation cases. |
| `vecgrep/eval/harness.py:306` | BLE001, S110 | Failed readiness and warm-up attempts remain isolated from measured evaluation cases. |
| `vecgrep/mcp/server.py:122` | BLE001, S110 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:294` | BLE001 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:324` | BLE001 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:491` | BLE001 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:511` | BLE001 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:524` | BLE001 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:541` | BLE001 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:666` | BLE001, S110 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:1446` | BLE001, S110 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:2040` | BLE001 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:2541` | BLE001, S110 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:2575` | BLE001, S110 | Tool and optional metadata failures are converted to their established protocol results. |
