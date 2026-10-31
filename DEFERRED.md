# Deferred behavior changes

**Unresolved lint diagnostics: none.**
**Unresolved Pyright (`standard` mode) diagnostics: none.**

The constructs below were deliberately retained. Narrowing an exception boundary, changing an exception type, replacing a framework default, or changing reentrant resource ownership requires separate contract-specific tests.

- `tests/fixtures/code_symbols/broken.py:5`: syntax diagnostics remain in intentional malformed input. Both `[tool.ruff]` and `[tool.pyright]` exclude this fixture to keep parser-failure coverage intact.

## Pyright annotations

None. The optional `rerank` extra (`torch`, `sentence-transformers`) resolves
through the partial stubs in `typings/`. `reportMissingModuleSource` is off
because the dev environment does not install that extra.

## Targeted Ruff annotations (96 locations)

Each entry below preserves the upstream one-line reason. Existing full-suite and retrieval comparisons passed; they do not establish exhaustive coverage of every exceptional condition.

| Location | Rules | Reason |
| --- | --- | --- |
| `scripts/rerank_benchmark.py:89` | TRY004 | Malformed benchmark content uses one ValueError contract. |
| `scripts/rerank_benchmark.py:99` | TRY004 | Malformed benchmark content uses one ValueError contract. |
| `tests/conftest.py:112` | BLE001, S110 | Teardown must not mask the test result if a backend close fails. |
| `tests/eval_harness.py:244` | BLE001, S110 | Evaluation teardown must not replace its collected result. |
| `tests/test_bm25_bulk_persist.py:100` | BLE001, S110 | Cleanup of a deliberately interrupted store must not mask the assertion. |
| `tests/test_oauth_token_store.py:430` | BLE001 | The test reports any failure below. |
| `tests/test_search_stage_timings.py:92` | F401 | Importing applies the level. |
| `tests/test_search_text_write_paths.py:70` | BLE001, S110 | The following assertions verify recovery after the injected indexing failure. |
| `tests/test_write_source_alias.py:80` | BLE001, S110 | Test cleanup must not mask the alias assertions. |
| `vecgrep/backend/api/admin.py:128` | BLE001, S110 | Reset must release stale service references even if a backend cannot close. |
| `vecgrep/backend/api/routes.py:655` | BLE001 | No reason recorded at the site. |
| `vecgrep/backend/auth/google_bearers.py:118` | BLE001 | Registry failures retain the logged auth-boundary error contract. |
| `vecgrep/backend/auth/google_bearers.py:127` | BLE001 | Registry failures retain the logged auth-boundary error contract. |
| `vecgrep/backend/auth/google_bearers.py:139` | BLE001 | Registry failures retain the logged auth-boundary error contract. |
| `vecgrep/backend/auth/google_bearers.py:169` | BLE001 | Registry failures retain the logged auth-boundary error contract. |
| `vecgrep/backend/auth/store.py:176` | BLE001 | A corrupt state snapshot falls back to owner reapproval rather than crashing startup. |
| `vecgrep/backend/backup.py:94` | BLE001 | Cleanup cannot invalidate a download. |
| `vecgrep/backend/backup.py:153` | BLE001 | Backup rollback and cleanup must contain arbitrary storage-provider failures. |
| `vecgrep/backend/backup.py:316` | BLE001, S110 | Backup rollback and cleanup must contain arbitrary storage-provider failures. |
| `vecgrep/backend/backup.py:391` | BLE001 | Backup rollback and cleanup must contain arbitrary storage-provider failures. |
| `vecgrep/backend/backup.py:462` | BLE001 | Backup rollback and cleanup must contain arbitrary storage-provider failures. |
| `vecgrep/backend/backup.py:468` | BLE001, S110 | Backup rollback and cleanup must contain arbitrary storage-provider failures. |
| `vecgrep/backend/backup.py:524` | BLE001, S112 | Backup rollback and cleanup must contain arbitrary storage-provider failures. |
| `vecgrep/backend/ingestion/adapters/pdf.py:40` | BLE001 | One unreadable PDF page must not discard text from the other pages. |
| `vecgrep/backend/ingestion/chunkers/code_symbol.py:102` | F401 | No reason recorded at the site. |
| `vecgrep/backend/ingestion/chunkers/code_symbol.py:103` | F401 | No reason recorded at the site. |
| `vecgrep/backend/ingestion/visual_worker.py:34` | BLE001 | An individual enrichment failure must be recorded without stopping the consumer. |
| `vecgrep/backend/main.py:63` | BLE001 | Optional transport, warm-up and background cleanup failures are reported without stopping the app. pragma: no cover - defensive. |
| `vecgrep/backend/main.py:144` | BLE001 | Optional transport, warm-up and background cleanup failures are reported without stopping the app. |
| `vecgrep/backend/main.py:165` | BLE001 | Optional transport, warm-up and background cleanup failures are reported without stopping the app. |
| `vecgrep/backend/main.py:186` | BLE001 | Optional transport, warm-up and background cleanup failures are reported without stopping the app. |
| `vecgrep/backend/main.py:235` | BLE001 | Optional transport, warm-up and background cleanup failures are reported without stopping the app. pragma: no cover - defensive. |
| `vecgrep/backend/mutation.py:204` | SIM115 | Reentrant lock state owns the handle until the existing finally closes it. |
| `vecgrep/backend/rerank.py:221` | BLE001, S110 | External model and process failures cross this worker boundary as explicit error results. |
| `vecgrep/backend/rerank.py:337` | BLE001 | External model and process failures cross this worker boundary as explicit error results. |
| `vecgrep/backend/rerank.py:356` | BLE001 | External model and process failures cross this worker boundary as explicit error results. |
| `vecgrep/backend/rerank.py:566` | BLE001 | External model and process failures cross this worker boundary as explicit error results. |
| `vecgrep/backend/rerank.py:731` | BLE001 | External model and process failures cross this worker boundary as explicit error results. |
| `vecgrep/backend/service.py:231` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:317` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:642` | TRY004 | Malformed persisted chunk data retains the RuntimeError contract. |
| `vecgrep/backend/service.py:919` | BLE001 | Prewarm is best-effort; canonical errors surface inside _index_locked. |
| `vecgrep/backend/service.py:1107` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:1168` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:1438` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:1625` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:1635` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:1706` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:2118` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:2139` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:2167` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:2621` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. never let warming break a search. |
| `vecgrep/backend/service.py:2721` | BLE001 | An unknown corpus gets no prior. |
| `vecgrep/backend/service.py:3175` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. bookkeeping only -- never block shutdown. |
| `vecgrep/backend/service.py:3389` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:3411` | BLE001 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:4151` | BLE001, S110 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:4488` | BLE001, S112 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/service.py:4492` | BLE001, S112 | Provider failures are propagated or reported here without stranding concurrent corpus work. |
| `vecgrep/backend/store/bm25_store.py:274` | BLE001 | A corrupt derived index is rebuilt from canonical source data. |
| `vecgrep/backend/store/qdrant_store.py:171` | BLE001, S110 | The storage adapter retains its existing missing-item and optional-index error contract. |
| `vecgrep/backend/store/qdrant_store.py:282` | BLE001 | The storage adapter retains its existing missing-item and optional-index error contract. |
| `vecgrep/backend/store/qdrant_store.py:382` | TRY004 | Unsupported backend snapshot shape retains the RuntimeError contract. |
| `vecgrep/backend/store/qdrant_store.py:496` | BLE001 | The storage adapter retains its existing missing-item and optional-index error contract. |
| `vecgrep/backend/store/qdrant_store.py:517` | BLE001 | The storage adapter retains its existing missing-item and optional-index error contract. |
| `vecgrep/backend/store/qdrant_store.py:540` | BLE001 | The storage adapter retains its existing missing-item and optional-index error contract. |
| `vecgrep/backend/write/confirm.py:67` | BLE001 | Write failures become explicit confirmation results instead of losing the user operation. |
| `vecgrep/backend/write/confirm.py:143` | BLE001 | Write failures become explicit confirmation results instead of losing the user operation. |
| `vecgrep/backend/write/confirm.py:282` | BLE001 | Write failures become explicit confirmation results instead of losing the user operation. |
| `vecgrep/backend/write/confirm.py:349` | BLE001 | Write failures become explicit confirmation results instead of losing the user operation. |
| `vecgrep/backend/write/confirm.py:373` | BLE001 | Write failures become explicit confirmation results instead of losing the user operation. |
| `vecgrep/cli/main.py:1156` | BLE001 | Adapter failures become CLI errors while independent watched sources remain active. |
| `vecgrep/cli/main.py:1195` | BLE001 | Adapter failures become CLI errors while independent watched sources remain active. |
| `vecgrep/cli/main.py:1272` | BLE001 | Adapter failures become CLI errors while independent watched sources remain active. |
| `vecgrep/cli/main.py:1572` | BLE001 | Adapter failures become CLI errors while independent watched sources remain active. |
| `vecgrep/eval/harness.py:23` | PLC0414 | Preserve the existing public re-export. |
| `vecgrep/eval/harness.py:192` | BLE001 | Failed readiness and warm-up attempts remain isolated from measured evaluation cases. |
| `vecgrep/eval/harness.py:316` | BLE001, S110 | Failed readiness and warm-up attempts remain isolated from measured evaluation cases. |
| `vecgrep/mcp/server.py:87` | F401 | No reason recorded at the site. |
| `vecgrep/mcp/server.py:124` | BLE001, S110 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:242` | BLE001, S110 | No reason recorded at the site. |
| `vecgrep/mcp/server.py:328` | BLE001 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:361` | BLE001 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:380` | BLE001 | Fallback inside _should_* handles missing counts. |
| `vecgrep/mcp/server.py:534` | BLE001 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:554` | BLE001 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:567` | BLE001 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:584` | BLE001 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:715` | BLE001, S110 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:1138` | BLE001 | Never lose the write over an embed. |
| `vecgrep/mcp/server.py:1261` | BLE001 | The file is the source of truth. |
| `vecgrep/mcp/server.py:1406` | BLE001 | The file on disk is the truth. |
| `vecgrep/mcp/server.py:1501` | BLE001, S110 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:2095` | BLE001 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:2636` | BLE001, S110 | Tool and optional metadata failures are converted to their established protocol results. |
| `vecgrep/mcp/server.py:2670` | BLE001, S110 | Tool and optional metadata failures are converted to their established protocol results. |
