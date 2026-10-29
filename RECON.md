# Lint burndown reconnaissance

## Scope and base

The requested branch was created from fetched `origin/main` at `4fda7b5d975c63cacdd55f1c498b1c1829ff9944`. That revision already contains the lint cleanup, in `style: make the repository lint gate reproducible`. Its parent is the supplied integration baseline `8b61a556d6a4f893484387405508aace7cc42299`.

The initial zero count triggered the requested reconnaissance stop. After authorization to finish, acceptance verification resumed. The pre-cleanup comparison uses an archived copy of the parent commit; the task branch never moved backward and no upstream commit was rewritten.

## Tool and configuration

`docs/DEVELOPING.md` specifies `ruff check .`. The current dev extra and `tool.ruff.required-version` pin Ruff to `>=0.17.3,<0.18`; the executable used here is Ruff 0.17.3. There is no `.cc-land.json` or separate repository Ruff configuration. The effective enabled rule lists were captured for both revisions and are identical.

The upstream cleanup added one exclusion: `tests/fixtures/code_symbols/broken.py`, which deliberately contains invalid Python for parser-failure coverage. This accounts for two historical syntax diagnostics. It is an intentional exclusion, not a repaired syntax error. This task made no lint configuration changes.

The comparison passed each revision's `pyproject.toml` explicitly and used the complete tracked Python file list with `--no-respect-gitignore` and `--force-exclude`. This prevents the archived copy's scratch-directory ignore rule or its parent configuration from changing the historical result. The ordinary current-tree `ruff check . --no-cache` also passes.

## Findings by rule

Fixability below is Ruff's pre-cleanup classification. It is not an assertion that all suggested unsafe fixes were applied. `Other` includes manual and display-only suggestions. Behavior-sensitive retained constructs are listed in [DEFERRED.md](DEFERRED.md).

| Rule | Before | After | Safe fix | Unsafe fix | Other |
| --- | ---: | ---: | ---: | ---: | ---: |
| B008 | 1 | 0 | 0 | 0 | 1 |
| B018 | 2 | 0 | 0 | 0 | 2 |
| B023 | 6 | 0 | 0 | 0 | 6 |
| BLE001 | 77 | 0 | 0 | 0 | 77 |
| C408 | 8 | 0 | 0 | 8 | 0 |
| F401 | 33 | 0 | 33 | 0 | 0 |
| F541 | 2 | 0 | 2 | 0 | 0 |
| F841 | 2 | 0 | 0 | 2 | 0 |
| FLY002 | 1 | 0 | 0 | 1 | 0 |
| FURB188 | 6 | 0 | 6 | 0 | 0 |
| FURB192 | 2 | 0 | 0 | 2 | 0 |
| I001 | 98 | 0 | 98 | 0 | 0 |
| ISC004 | 1 | 0 | 0 | 1 | 0 |
| PERF102 | 1 | 0 | 0 | 1 | 0 |
| PIE790 | 1 | 0 | 1 | 0 | 0 |
| PIE810 | 4 | 0 | 0 | 4 | 0 |
| PLR1730 | 1 | 0 | 1 | 0 | 0 |
| PLW1510 | 5 | 0 | 0 | 0 | 5 |
| RUF010 | 1 | 0 | 1 | 0 | 0 |
| RUF012 | 2 | 0 | 0 | 0 | 2 |
| RUF015 | 2 | 0 | 0 | 2 | 0 |
| RUF022 | 4 | 0 | 4 | 0 | 0 |
| RUF046 | 1 | 0 | 0 | 1 | 0 |
| RUF059 | 10 | 0 | 0 | 10 | 0 |
| RUF100 | 11 | 0 | 11 | 0 | 0 |
| S110 | 18 | 0 | 0 | 0 | 18 |
| S112 | 4 | 0 | 0 | 0 | 4 |
| SIM102 | 3 | 0 | 0 | 1 | 2 |
| SIM103 | 1 | 0 | 0 | 1 | 0 |
| SIM114 | 1 | 0 | 1 | 0 | 0 |
| SIM115 | 24 | 0 | 0 | 0 | 24 |
| SIM117 | 9 | 0 | 6 | 0 | 3 |
| SIM905 | 1 | 0 | 1 | 0 | 0 |
| TRY004 | 4 | 0 | 0 | 0 | 4 |
| UP006 | 2 | 0 | 2 | 0 | 0 |
| UP031 | 1 | 0 | 0 | 0 | 1 |
| UP035 | 21 | 0 | 20 | 0 | 1 |
| UP037 | 28 | 0 | 28 | 0 | 0 |
| invalid-syntax | 2 | 0 | 0 | 0 | 2 |
| **Total** | **401** | **0** | **215** | **34** | **152** |

The baseline affects 128 files. Current diagnostics: none. New diagnostics: none.

## Findings by file

| File | Before | After | Safe fix | Unsafe fix | Other |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scripts/bench_code_symbols.py` | 10 | 0 | 1 | 3 | 6 |
| `scripts/rerank_benchmark.py` | 3 | 0 | 1 | 0 | 2 |
| `scripts/smoke_base_install.py` | 2 | 0 | 2 | 0 | 0 |
| `tests/conftest.py` | 3 | 0 | 1 | 0 | 2 |
| `tests/eval_harness.py` | 2 | 0 | 0 | 0 | 2 |
| `tests/fixtures/code_symbols/broken.py` | 2 | 0 | 0 | 0 | 2 |
| `tests/fixtures/code_symbols/orders.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_adapters.py` | 2 | 0 | 2 | 0 | 0 |
| `tests/test_aliases.py` | 1 | 0 | 0 | 1 | 0 |
| `tests/test_anchors_incident.py` | 2 | 0 | 1 | 1 | 0 |
| `tests/test_auto_modes.py` | 1 | 0 | 0 | 1 | 0 |
| `tests/test_backup.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_bm25_bulk_persist.py` | 4 | 0 | 1 | 1 | 2 |
| `tests/test_bm25_cache_budget.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_bm25_runtime_integrity.py` | 3 | 0 | 3 | 0 | 0 |
| `tests/test_bm25_sqlite_equivalence.py` | 3 | 0 | 3 | 0 | 0 |
| `tests/test_bm25_store.py` | 1 | 0 | 0 | 1 | 0 |
| `tests/test_bm25_wal_maintenance.py` | 1 | 0 | 0 | 0 | 1 |
| `tests/test_cli_search_watch.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_cli_status.py` | 2 | 0 | 2 | 0 | 0 |
| `tests/test_cli_watch_resilience.py` | 1 | 0 | 0 | 0 | 1 |
| `tests/test_cli_write.py` | 4 | 0 | 1 | 0 | 3 |
| `tests/test_code_symbol_parser.py` | 3 | 0 | 1 | 2 | 0 |
| `tests/test_corpora_registry_churn.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_corpus_listing_hides_eval.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_cross_corpus_scope.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_dates.py` | 1 | 0 | 0 | 0 | 1 |
| `tests/test_decay.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_embed_cache_shutdown_flush.py` | 2 | 0 | 2 | 0 | 0 |
| `tests/test_embed_factory.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_embed_local_usage.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_embed_overflow.py` | 3 | 0 | 1 | 2 | 0 |
| `tests/test_eval_real_harness.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_fixed_token_chunker.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_google_bearer_registry.py` | 3 | 0 | 0 | 0 | 3 |
| `tests/test_index_include_glob.py` | 1 | 0 | 0 | 1 | 0 |
| `tests/test_mcp_direct_edit.py` | 9 | 0 | 2 | 0 | 7 |
| `tests/test_mcp_direct_write.py` | 4 | 0 | 2 | 0 | 2 |
| `tests/test_mcp_oauth_boundary.py` | 2 | 0 | 2 | 0 | 0 |
| `tests/test_mcp_propose.py` | 10 | 0 | 6 | 0 | 4 |
| `tests/test_mcp_propose_merge.py` | 2 | 0 | 2 | 0 | 0 |
| `tests/test_mcp_propose_meta_only.py` | 1 | 0 | 0 | 0 | 1 |
| `tests/test_mcp_server.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_mcp_tools_offloaded.py` | 3 | 0 | 3 | 0 | 0 |
| `tests/test_migrate.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_mixed_model_backend.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_multi_corpus_interfaces.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_multi_corpus_search.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_mutation_locks.py` | 6 | 0 | 5 | 0 | 1 |
| `tests/test_mutation_recovery.py` | 1 | 0 | 0 | 1 | 0 |
| `tests/test_oauth_google_registry.py` | 2 | 0 | 1 | 0 | 1 |
| `tests/test_oauth_provider.py` | 3 | 0 | 3 | 0 | 0 |
| `tests/test_oauth_root_discovery.py` | 1 | 0 | 0 | 1 | 0 |
| `tests/test_ollama_connect_timeout.py` | 2 | 0 | 2 | 0 | 0 |
| `tests/test_orphaned_source.py` | 2 | 0 | 2 | 0 | 0 |
| `tests/test_payload_source_dedup.py` | 3 | 0 | 0 | 3 | 0 |
| `tests/test_propose_merge.py` | 1 | 0 | 0 | 1 | 0 |
| `tests/test_qdrant_connection_pooling.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_qdrant_upsert_batching.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_registration_ingress_replay.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_registry_race.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_release_workflow.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_rerank_benchmark.py` | 2 | 0 | 1 | 0 | 1 |
| `tests/test_rerank_failure_degrades.py` | 2 | 0 | 2 | 0 | 0 |
| `tests/test_rerank_score_cache.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_scaling_audit.py` | 2 | 0 | 2 | 0 | 0 |
| `tests/test_search_policy.py` | 4 | 0 | 4 | 0 | 0 |
| `tests/test_search_text_write_paths.py` | 2 | 0 | 0 | 0 | 2 |
| `tests/test_serve_mcp.py` | 4 | 0 | 4 | 0 | 0 |
| `tests/test_source_path_aliases.py` | 3 | 0 | 1 | 0 | 2 |
| `tests/test_temp_corpus.py` | 3 | 0 | 1 | 0 | 2 |
| `tests/test_v11_tools.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_visual_enrichment.py` | 9 | 0 | 7 | 0 | 2 |
| `tests/test_web_workspace.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_write_codex_fixes.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_write_confirm.py` | 1 | 0 | 1 | 0 | 0 |
| `tests/test_write_frontmatter_safety.py` | 2 | 0 | 1 | 1 | 0 |
| `tests/test_write_metadata_readpath.py` | 2 | 0 | 2 | 0 | 0 |
| `tests/test_write_protected.py` | 2 | 0 | 2 | 0 | 0 |
| `tests/test_write_source_alias.py` | 2 | 0 | 0 | 0 | 2 |
| `tests/test_write_wall.py` | 2 | 0 | 2 | 0 | 0 |
| `vecgrep/backend/api/admin.py` | 2 | 0 | 0 | 0 | 2 |
| `vecgrep/backend/api/routes.py` | 3 | 0 | 1 | 1 | 1 |
| `vecgrep/backend/api/schemas.py` | 2 | 0 | 2 | 0 | 0 |
| `vecgrep/backend/assembly.py` | 18 | 0 | 18 | 0 | 0 |
| `vecgrep/backend/auth/google_bearers.py` | 4 | 0 | 0 | 0 | 4 |
| `vecgrep/backend/auth/scopes.py` | 1 | 0 | 1 | 0 | 0 |
| `vecgrep/backend/auth/store.py` | 1 | 0 | 0 | 0 | 1 |
| `vecgrep/backend/auth/throttle.py` | 4 | 0 | 3 | 0 | 1 |
| `vecgrep/backend/backup.py` | 14 | 0 | 4 | 0 | 10 |
| `vecgrep/backend/config.py` | 7 | 0 | 2 | 2 | 3 |
| `vecgrep/backend/embed/local_usage.py` | 1 | 0 | 1 | 0 | 0 |
| `vecgrep/backend/embed/ollama.py` | 1 | 0 | 1 | 0 | 0 |
| `vecgrep/backend/explorer.py` | 1 | 0 | 1 | 0 | 0 |
| `vecgrep/backend/explorer_store.py` | 1 | 0 | 1 | 0 | 0 |
| `vecgrep/backend/ingestion/adapters/__init__.py` | 2 | 0 | 2 | 0 | 0 |
| `vecgrep/backend/ingestion/adapters/base.py` | 1 | 0 | 1 | 0 | 0 |
| `vecgrep/backend/ingestion/adapters/chatgpt_export.py` | 1 | 0 | 1 | 0 | 0 |
| `vecgrep/backend/ingestion/adapters/claude_export.py` | 1 | 0 | 1 | 0 | 0 |
| `vecgrep/backend/ingestion/adapters/discord_jsonl.py` | 1 | 0 | 1 | 0 | 0 |
| `vecgrep/backend/ingestion/adapters/image.py` | 2 | 0 | 2 | 0 | 0 |
| `vecgrep/backend/ingestion/adapters/markdown.py` | 2 | 0 | 2 | 0 | 0 |
| `vecgrep/backend/ingestion/adapters/pdf.py` | 3 | 0 | 2 | 0 | 1 |
| `vecgrep/backend/ingestion/adapters/plaintext.py` | 1 | 0 | 1 | 0 | 0 |
| `vecgrep/backend/ingestion/adapters/url.py` | 1 | 0 | 1 | 0 | 0 |
| `vecgrep/backend/ingestion/chunkers/__init__.py` | 2 | 0 | 2 | 0 | 0 |
| `vecgrep/backend/ingestion/chunkers/code_parse.py` | 2 | 0 | 2 | 0 | 0 |
| `vecgrep/backend/ingestion/chunkers/code_symbol.py` | 1 | 0 | 1 | 0 | 0 |
| `vecgrep/backend/ingestion/visual_processor.py` | 1 | 0 | 1 | 0 | 0 |
| `vecgrep/backend/ingestion/visual_queue.py` | 2 | 0 | 2 | 0 | 0 |
| `vecgrep/backend/ingestion/visual_worker.py` | 2 | 0 | 1 | 0 | 1 |
| `vecgrep/backend/main.py` | 5 | 0 | 0 | 0 | 5 |
| `vecgrep/backend/migrate_payload.py` | 1 | 0 | 0 | 1 | 0 |
| `vecgrep/backend/mutation.py` | 4 | 0 | 3 | 0 | 1 |
| `vecgrep/backend/rerank.py` | 7 | 0 | 2 | 0 | 5 |
| `vecgrep/backend/service.py` | 40 | 0 | 13 | 5 | 22 |
| `vecgrep/backend/store/__init__.py` | 2 | 0 | 2 | 0 | 0 |
| `vecgrep/backend/store/bm25_store.py` | 3 | 0 | 2 | 0 | 1 |
| `vecgrep/backend/store/qdrant_store.py` | 7 | 0 | 0 | 0 | 7 |
| `vecgrep/backend/timeline.py` | 2 | 0 | 2 | 0 | 0 |
| `vecgrep/backend/write/__init__.py` | 2 | 0 | 2 | 0 | 0 |
| `vecgrep/backend/write/confirm.py` | 6 | 0 | 0 | 0 | 6 |
| `vecgrep/cli/main.py` | 9 | 0 | 2 | 0 | 7 |
| `vecgrep/eval/gold.py` | 2 | 0 | 2 | 0 | 0 |
| `vecgrep/eval/harness.py` | 4 | 0 | 1 | 0 | 3 |
| `vecgrep/eval/metrics.py` | 2 | 0 | 1 | 1 | 0 |
| `vecgrep/mcp/search_policy.py` | 4 | 0 | 1 | 3 | 0 |
| `vecgrep/mcp/server.py` | 36 | 0 | 16 | 1 | 19 |

## Acceptance evidence

See [LINT_VERIFICATION.md](LINT_VERIFICATION.md) for test counts, deterministic evaluation, package build and isolated wheel checks.
