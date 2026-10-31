# Phase 0 Pyright Reconnaissance (`vecgrep`)

## 1. Environment & Baseline Snapshot

- **Branch**: `chore/pyright-basic` (isolated git worktree).
- **Base Commit (`origin/main`)**: `9bd41709fe72fa4ccff8d5b4fd888da2e4ca47ae` (`2026-10-30`).
  - Note: Between the prompt snapshot (`defc349`) and worktree creation, 7 commits landed on `origin/main` (`defc349..9bd4170`: `31d44f0`, `90a0134`, `9574eb3`, `a278bf7`, `c9f52b1`, `33e97d6`, `9bd4170`). Numbers below are recorded at `9bd4170` with the `defc349..9bd4170` delta noted where relevant.
- **Throwaway Virtualenv**: `./venv` (gitignored)
  - Python: `3.13.16`
  - Pinned Pyright: `pyright[nodejs]==1.1.408` (`pyright 1.1.408`, bundled `nodejs-wheel-binaries==24.19.0`)
  - Ruff: `0.17.3`
  - Pytest: `9.1.1`
- **Packaging Note for Reproducibility**:
  1. Installing bare `pyright==1.1.408` on hosts without a global `node` binary fails at runtime in `nodeenv` (`ValueError: invalid literal for int() with base 10: '0-alpha'` when parsing `nodejs.org` release tags). Pinning `pyright[nodejs]==1.1.408` in `[project.optional-dependencies].dev` makes the gate self-contained.
  2. Running `./venv/bin/pyright` outside an activated shell defaults to `/usr/bin/python3` unless `--pythonpath ./venv/bin/python` or `venvPath`/`venv` is set.
  3. `tests/fixtures/code_symbols/broken.py` is an intentionally malformed Python fixture for AST fallback tests (already in `[tool.ruff] extend-exclude`) and emits 7 syntax diagnostics unless excluded in `[tool.pyright].exclude`.

---

## 2. Baseline Verification Metrics

### 2.1 Pytest Full Suite
Command: `./venv/bin/pytest -q`

- **Result**: `1625 passed, 10 skipped in 64.80s` (at `defc349`: `1594 passed`; the 7 commits in `defc349..9bd4170` added 31 passing tests and 10 host-script tests in `tests/test_weekly_eval_alert.py` that skip when the optional external eval script is absent).

### 2.2 Ruff Check
Command: `./venv/bin/ruff check .`

- **Result at `9bd4170`**: **14 errors** across 6 files:
  - Present at `defc349` (2 errors):
    - `scripts/rerank_benchmark.py:1` — `EXE001` (shebang present, file not executable)
    - `vecgrep/backend/store/bm25_sqlite.py:109` — `SIM102` (collapsible nested `if`)
  - Added in `defc349..9bd4170` (12 errors):
    - `vecgrep/backend/service.py:3227, 3255` — `BLE001` (2), `S110` (1)
    - `tests/test_bm25_bulk_persist.py:3` — `I001` (1), `F401` (1)
    - `tests/test_qdrant_connection_pooling.py:13` — `RUF100` (1)
    - `tests/test_round2_audit_fixes.py:3, 212, 238, 305, 312, 319` — `I001` (1), `F401` (1), `SIM117` (2), `F821` (2: undefined `VecgrepService`)

### 2.3 Deterministic Eval Baseline

#### `python -m tests.test_retrieval_fixture` (both stores, all 3 modes)
| Store | Mode | `hit@1` | `hit@3` | `mrr` | `forbid_leaks` | `neg_fp_rate` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `pickle` | `bm25` | `100.0` | `100.0` | `1.0` | `0` | `0.0` |
| `pickle` | `vector` | `100.0` | `100.0` | `1.0` | `0` | `0.0` |
| `pickle` | `hybrid` | `100.0` | `100.0` | `1.0` | `0` | `0.0` |
| `sqlite` | `bm25` | `100.0` | `100.0` | `1.0` | `0` | `0.0` |
| `sqlite` | `vector` | `100.0` | `100.0` | `1.0` | `0` | `0.0` |
| `sqlite` | `hybrid` | `100.0` | `100.0` | `1.0` | `0` | `0.0` |

#### `python -m tests.eval_harness`
| Query | `expected_recall` | `forbidden_hits` | `redundancy` | `top_sources` |
| --- | ---: | ---: | ---: | --- |
| `alias_cn` | `1.0` | `0` | `0.0000` | `notes:17, code:1` |
| `argument` | `1.0` | `3` | `0.0000` | `chat:21` |
| `dup_stress` | `1.0` | `0` | `0.0222` | `notes:21` |
| `exact_keyword` | `1.0` | `0` | `0.0000` | `code:4, notes:2, chat:1` |
| `greatest_hits` | `1.0` | `0` | `0.0000` | `chat:9, notes:8, docs:2, code:2` |
| `incident_today` | `1.0` | `4` | `0.0000` | `notes:17, docs:2, code:2` |
| `incident_today_filtered` | `1.0` | `0` | `0.0000` | `notes:5, code:2, docs:2` |
| `semantic_vague` | `1.0` | `0` | `0.0000` | `chat:9, notes:9, docs:2, code:1` |

---

## 3. Pyright 1.1.408 Error Counts by Rule

Analyzed across 243 Python files (`vecgrep/`, `tests/`, `scripts/`), excluding `venv/` and `tests/fixtures/code_symbols/broken.py` (+7 syntax errors if `broken.py` is not excluded).

### 3.1 Rule Breakdown Across Configurations

| Rule | `.[dev]` only (`standard` + `broken.py`) | Full extras (`basic`, excl. `broken.py`) | Full extras (`standard`, excl. `broken.py`) | Full extras (`strict`, excl. `broken.py`) | Primary Classification |
| --- | ---: | ---: | ---: | ---: | --- |
| `reportArgumentType` | 106 | **145** | **145** | 158 | Mostly Optional/Literal/TypedDict widening; **4 real bugs** (`bm25_sqlite.py`, `eval/harness.py`, `visual_processor.py`, `test_round2_audit_fixes.py`) |
| `reportAttributeAccessIssue` | 55 | **54** | **54** | 54 | **1 real crash** (`auth/provider.py:140`); rest are test fake/mock attributes and union narrowing (`Cursor` vs `BatchCursor`) |
| `reportOptionalMemberAccess` | 48 | **50** | **50** | 50 | Unchecked `None` returns in `qdrant_store.py`, `mcp/server.py`, `spec_from_file_location`, and test assertions |
| `reportOptionalSubscript` | 32 | **35** | **35** | 35 | Optional dict/row indexing in `auth/store.py:216` and test assertions |
| `reportCallIssue` | 12 | **12** | **12** | 12 | Pydantic v2 `Field(default, ...)` positional stub mismatch in `search_policy.py`, `auth/store.py`, and test `SearchRequest` calls |
| `reportGeneralTypeIssues` | 10 | **10** | **10** | 10 | **2 real adapter bugs** (`chatgpt_export.py`, `claude_export.py`), **2 bad annotations** (`timeline.py`), 3 MCP tool decorator return mismatches, 3 test `list` mutations on `Sequence[str]` |
| `reportOptionalOperand` | 0 | **8** | **8** | 8 | Optional arithmetic/comparison in `tests/test_oauth_token_store.py` and `tests/test_oauth_google_registry.py` |
| `reportIndexIssue` | 8 | **8** | **8** | 8 | Typing `res["results"]` as `dict[str, object]` in `tests/test_round2_audit_fixes.py` |
| `reportOperatorIssue` | 6 | **7** | **7** | 7 | `object` arithmetic in `assembly.py` (`chunk.get("tokens")`), `routes.py`, and test assertions |
| `reportMissingImports` | 57 | **5** | **5** | 5 | Optional `torch` and `sentence_transformers` (`rerank` extra) in `vecgrep/backend/rerank.py` and `scripts/rerank_benchmark.py` (with `.[dev]` only, also 48 `mcp.*` and 4 `tree_sitter*`) |
| `reportUndefinedVariable` | 2 | **2** | **2** | 2 | **Real bug**: unimported `VecgrepService` in `tests/test_round2_audit_fixes.py:238, 312` |
| `reportReturnType` | 2 | **2** | **2** | 2 | Narrowing/guard gaps in `qdrant_store.py:513` and `local_usage.py:67` |
| `reportPrivateImportUsage` | 0 | **1** | **1** | 1 | **Fragile import**: `TransportSecuritySettings` imported from private re-export in `vecgrep/mcp/server.py:2286` |
| `reportPossiblyUnboundVariable` | 1 | 0 | **1** | 1 | **Standard-only hazard**: `intent` across correlated `if tailnet_login:` blocks in `vecgrep/backend/main.py:488` |
| `reportFunctionMemberAccess` | 2 | 0 | **2** | 2 | **Standard-only**: monkeypatched function attributes in `tests/test_ollama_backend.py:350` |
| `<syntax-error>` (`broken.py`) | 7 | (7 if incl.) | (7 if incl.) | (7 if incl.) | Deliberately broken AST test fixture `tests/fixtures/code_symbols/broken.py` |
| Strict-only `reportUnknown*` / `reportMissing*` | — | 0 | 0 | 16,229 | Untyped JSON dicts, pytest fixtures, third-party `Any` |
| **Total** | **348** | **339** | **342** | **16,574** | **180 in production (`vecgrep/` + `scripts/`), 162 in `tests/` (`standard`)** |

*Note on optional extras*: Installing `.[dev]` alone leaves `mcp`, `tree-sitter`, and `openai` uninstalled, turning 48 `mcp` type checks and 4 `tree-sitter` type checks into blind `reportMissingImports`. Installing `.[dev,mcp,code,openai,watch]` eliminates those missing imports and exposes 39 additional real type checks inside `vecgrep/mcp/server.py`, `vecgrep/backend/auth/*`, and `tests/test_oauth_*`.

---

## 4. Error Breakdown by Module Cluster & File

### 4.1 Production Code (`vecgrep/` and `scripts/`) — 179 `basic` / 180 `standard` errors across 21 files

| Module Cluster / File | `basic` | `standard` | Rules | Nature of Findings (Real Bug vs Noise) |
| --- | ---: | ---: | --- | --- |
| **Core Backend & Search** | **42** | **43** | | |
| `vecgrep/backend/service.py` | 34 | 34 | `reportArgumentType:28`, `reportAttributeAccessIssue:6` | **Noise / Narrowing**: (1) `Corpus(**{...})` at L3350 where dict comprehension widens value type to `object` (24 errors); (2) `BatchCursor` vs `Cursor` union attribute access on `.fetchall()`/`.fetchone()` at L3295-3306 (6 errors); (3) `dict[str, object]` unpack into `Chunk` / `extract_rerank_window` (4 errors). |
| `vecgrep/backend/assembly.py` | 5 | 5 | `reportArgumentType:3`, `reportOperatorIssue:2` | **Noise / Narrowing**: `dict.get("tokens", 0)` on `dict[str, object]` returns `object`, causing `+` and `int(...)` complaints. |
| `vecgrep/backend/timeline.py` | 2 | 2 | `reportGeneralTypeIssues:2` | **Bug (#7)**: Uses builtin function `callable` instead of `collections.abc.Callable` in type annotations at L84, L87. |
| `vecgrep/backend/explorer.py` | 1 | 1 | `reportArgumentType:1` | **Noise**: `Counter.update` / dict value narrowing at L181. |
| `vecgrep/backend/main.py` | 0 | 1 | `reportPossiblyUnboundVariable:1` | **Hazard (#6, `standard` mode)**: `intent` bound inside `if tailnet_login:` at L338 and used inside a second `if tailnet_login:` 150 lines later at L488. |
| **Auth & OAuth (`vecgrep/backend/auth/`)** | **21** | **21** | | |
| `vecgrep/backend/auth/store.py` | 12 | 12 | `reportArgumentType:9`, `reportCallIssue:2`, `reportOptionalSubscript:1` | **Narrowing**: `OAuthClientInformationFull.client_id` is typed `str \| None` in `mcp` stubs even when validated (9); `AnyUrl(str)` positional stub issue (2); unchecked `row[0]` on `fetchone()` at L216 (1). |
| `vecgrep/backend/auth/provider.py` | 8 | 8 | `reportArgumentType:7`, `reportAttributeAccessIssue:1` | **Real Bug (#1)**: L140 `from mcp.server.auth.errors import TokenError` raises `ImportError` at runtime (`TokenError` lives in `mcp.server.auth.provider`). Remaining 7 are `client_id: str \| None` narrowing. |
| `vecgrep/backend/auth/approval.py` | 1 | 1 | `reportArgumentType:1` | **Narrowing**: `hmac.compare_digest` with `str \| None` when `expected` is not narrowed. |
| **MCP Server (`vecgrep/mcp/`)** | **19** | **19** | | |
| `vecgrep/mcp/server.py` | 17 | 17 | `reportArgumentType:12`, `reportGeneralTypeIssues:3`, `reportOptionalMemberAccess:1`, `reportPrivateImportUsage:1` | **Bug/Hazard (#4)**: L2286 imports `TransportSecuritySettings` from private re-export `mcp.server.fastmcp.server` instead of `mcp.server.transport_security`. L1432 accesses `self._WriteProposal.validate_target_path` without asserting `self._WriteProposal is not None`. Remaining 15 are MCP tool callback signature variance and `str \| None` defaults passed to `str` params. |
| `vecgrep/mcp/search_policy.py` | 2 | 2 | `reportArgumentType:1`, `reportCallIssue:1` | **Noise**: Pydantic `Field(8, ge=1, le=25)` positional default stub mismatch and `SearchMode` literal assignment. |
| **Stores (`vecgrep/backend/store/`)** | **12** | **12** | | |
| `vecgrep/backend/store/bm25_sqlite.py` | 9 | 9 | `reportArgumentType:9` | **Real Bug (#2)**: L141, 144, 148, 151 in `_sql_filter_clauses` call `float(val)` on date strings / `None` and emit SQL referencing non-existent column `c.doc_timestamp`. Remaining 5 are `sqlite3.Row` / `Cursor` typing. |
| `vecgrep/backend/store/qdrant_store.py` | 3 | 3 | `reportOptionalMemberAccess:2`, `reportReturnType:1` | **Narrowing**: `self._client` optional access before guard (2) and `scroll` return tuple narrowing at L513 (1). |
| **Rerank & Benchmark (`rerank.py`, `scripts/`)** | **14** | **14** | | |
| `vecgrep/backend/rerank.py` | 5 | 5 | `reportMissingImports:3`, `reportArgumentType:2` | **Noise**: Optional `torch` / `sentence_transformers` imports (3) and ` dict[str, object]` worker response extraction (2). |
| `scripts/rerank_benchmark.py` | 9 | 9 | `reportAttributeAccessIssue:4`, `reportOptionalMemberAccess:2`, `reportMissingImports:2`, `reportArgumentType:1` | **Hazard (#10)**: Unchecked `spec` and `spec.loader` in `importlib.util.module_from_spec(spec)` at L47-49 (3); dynamic module attrs on loaded script (4); optional `sentence_transformers` import (2). |
| **Ingestion & Embeddings** | **6** | **6** | | |
| `vecgrep/backend/ingestion/adapters/chatgpt_export.py` | 1 | 1 | `reportGeneralTypeIssues:1` | **Real Bug (#3)**: L89 iterates `data = load_json_file(...)` (`object`) without verifying `isinstance(data, list)`. |
| `vecgrep/backend/ingestion/adapters/claude_export.py` | 1 | 1 | `reportGeneralTypeIssues:1` | **Real Bug (#3)**: L71 iterates `data = load_json_file(...)` (`object`) without verifying `isinstance(data, list)`. |
| `vecgrep/backend/ingestion/visual_processor.py` | 1 | 1 | `reportArgumentType:1` | **Hazard (#9)**: L45 passes `scale: float` (`min(2.0, 1800 / max(width, height))`) to `page.render(scale=...)` typed as `int`. |
| `vecgrep/backend/ingestion/chunkers/code_parse.py` | 1 | 1 | `reportArgumentType:1` | **Noise**: `tree_sitter.Language(lang_fn())` ctypes pointer annotation mismatch (`int` vs ` object`). |
| `vecgrep/backend/embed/local_usage.py` | 1 | 1 | `reportReturnType:1` | **Noise**: JSON payload dict return narrowing at L67. |
| `vecgrep/backend/embed/ollama.py` | 1 | 1 | `reportArgumentType:1` | **Noise**: `httpx.Timeout` / payload dict typing. |
| **API, CLI & Eval** | **8** | **8** | | |
| `vecgrep/backend/api/routes.py` | 3 | 3 | `reportArgumentType:2`, `reportOperatorIssue:1` | **Noise**: `seen = {cid: None}` inferred as `dict[str, None]`, then assigned `seen[cid] = True` at L594. |
| `vecgrep/eval/harness.py` | 3 | 3 | `reportArgumentType:3` | **Bug (#8)**: `decay_half_life_days: float \| None \| str = "inherit"` only checks `== "inherit"`, allowing arbitrary `str` into `Corpus.decay_half_life_days: float \| None`. |
| `vecgrep/cli/main.py` | 2 | 2 | `reportArgumentType:2` | **Noise**: `typer` / `uvicorn` optional arg narrowing. |

### 4.2 Test Suite (`tests/`) — 160 `basic` / 162 `standard` errors across 48 files (excl. `broken.py`)

| Test Cluster | Files | `basic` | `standard` | Dominant Rules & Causes |
| --- | ---: | ---: | ---: | --- |
| OAuth & Auth tests (`test_oauth_*`, `test_mcp_oauth_boundary.py`, `test_google_bearer_registry.py`, `oauth_flow.py`) | 9 | 67 | 67 | `reportArgumentType:30`, `reportOptionalMemberAccess:18`, `reportOptionalOperand:8`, `reportOptionalSubscript:5`, `reportGeneralTypeIssues:3`: `OAuthClientInformationFull.client_id` (`str \| None`), optional token fields (`expires_at`, `refresh_token`) asserted in tests without `assert x is not None` first, and `list(scopes).append(...)` on `Sequence[str]`. |
| Code symbol & chunker tests (`test_code_symbol_parser.py`, `test_code_symbol.py`, `test_turn_window_chunker.py`, `test_area2_indexing.py`) | 4 | 38 | 38 | `reportOptionalSubscript:23`, `reportOptionalMemberAccess:11`, `reportOperatorIssue:3`: indexing `find_by_name(...)` (`dict \| None`) in test assertions before `assert sym is not None`. |
| Write-path tests (`test_write_confirm.py`, `test_write_protected.py`, `test_write_wall.py`, `test_write_metadata_readpath.py`, `test_write_source_alias.py`, `test_cli_write.py`) | 6 | 25 | 25 | `reportAttributeAccessIssue:23`: `_FakeService` test helper sets dynamic attributes (`svc.confirm_calls = ...`) not declared on class body. |
| Rerank & Embedding tests (`test_visual_enrichment.py`, `test_rerank_*`, `test_embed_*`, `test_ollama_backend.py`, `test_score_calibration.py`) | 12 | 45 | 47 | `reportArgumentType:17`, `reportAttributeAccessIssue:16`, `reportOptionalMemberAccess:6`, `reportOptionalSubscript:4`, `reportFunctionMemberAccess:2`: mock/fake classes missing dummy attributes in annotations; `Path` vs `str` in visual enrichment fixtures. |
| Round-2 audit & store/search tests (`test_round2_audit_fixes.py`, `test_web_workspace.py`, `test_scaling_audit.py`, `test_multi_corpus_interfaces.py`, `test_bm25_*`, `test_qdrant_*`, `test_backup.py`, `test_mcp_tools_offloaded.py`, `test_correctness_repairs.py`, `test_search_stage_timings.py`, `test_weekly_eval_alert.py`, `conftest.py`) | 17 | 45 | 45 | **Real test bug (#5)** in `test_round2_audit_fixes.py` (unimported `VecgrepService`, invalid `_score_cache` tuple key, `list` for `GoldCase.want: tuple`); `SearchRequest` positional/keyword call mismatches in `test_web_workspace.py` and `test_multi_corpus_interfaces.py`; `spec_from_file_location` optional checks. |

---

## 5. Top 10 Findings That Are Actual Bugs or Runtime Hazards

### 1. `vecgrep/backend/auth/provider.py:140` — `ImportError` crash on invalid/replayed OAuth authorization code
- **Rule**: `reportAttributeAccessIssue` (`"TokenError" is unknown import symbol` from `mcp.server.auth.errors`)
- **What happens**: Inside `VecgrepOAuthProvider.exchange_authorization_code`, if `self.store.consume_authorization_code(...)` returns `None`, line 140 executes:
  ```python
  from mcp.server.auth.errors import TokenError
  raise TokenError("invalid_grant", "Authorization code was already used or expired")
  ```
  `mcp.server.auth.errors` does not define or export `TokenError` (`TokenError` is in `mcp.server.auth.provider`, which lines 162, 173, and 181 of the same file import correctly).
- **Triggering input**: Any call to `exchange_authorization_code` where the code is missing, expired, or already consumed (for example, two concurrent `/token` requests racing with the same `authorization_code` after `load_authorization_code` succeeds). Instead of returning an OAuth `invalid_grant` response, it raises an unhandled `ImportError: cannot import name 'TokenError' from 'mcp.server.auth.errors'`.

### 2. `vecgrep/backend/store/bm25_sqlite.py:141, 144, 148, 151` — Broken `after:`/`before:` SQL filter pushdown (`ValueError`/`TypeError` + missing SQLite column `c.doc_timestamp`)
- **Rule**: `reportArgumentType` (`Argument of type "str | Any | None" cannot be assigned to parameter "x" of type "ConvertibleToFloat" in function "float"`)
- **What happens**: `_sql_filter_clauses` (introduced in `31d44f0`) parses `after:` and `before:` clauses from `filter_clauses`:
  1. When `filter_clauses` contains a string filter like `"after:2026-01-15"` or `"-before:2026-01-15"`, `val` is a raw date string (`"2026-01-15"`) and `float(val)` immediately raises `ValueError: could not convert string to float: '2026-01-15'`. When `filter_clauses` contains an object/dict filter where `val` is `None`, `float(val)` raises `TypeError`.
  2. Even when `val` is a numeric epoch string like `"after:1700000000"`, `_sql_filter_clauses` emits `(c.doc_timestamp >= ?)` against the `chunks c` table. However, the `chunks` table schema (`bm25_sqlite.py:64-70`) only has columns `(pos, cid, source_id, tokens, payload)` — `doc_timestamp` is stored inside the JSON `payload`, not as a column! Executing the query raises `sqlite3.OperationalError: no such column: c.doc_timestamp`.
- **Triggering input**: Calling `BM25SqliteStore.search("query", top_k=5, filter_clauses=["after:2026-01-15"])` (raises `ValueError`) or `BM25SqliteStore.search("query", top_k=5, filter_clauses=["after:1700000000"])` (raises `sqlite3.OperationalError: no such column: c.doc_timestamp`).

### 3. `vecgrep/backend/ingestion/adapters/chatgpt_export.py:89` & `claude_export.py:71` — Uncaught `TypeError` or key-iteration when top-level JSON is not a `list`
- **Rule**: `reportGeneralTypeIssues` (`"object" is not iterable`)
- **What happens**: `load_json_file(p, source)` parses any valid JSON value (`dict`, `list`, `int`, `bool`, `None`) and returns `object`. While `matches()` checks `_looks_like_*_export(obj)` (which requires a `list`), `load()` does not verify `isinstance(data, list)` before `for conv in data:`.
- **Triggering input**: Calling `ChatGPTExportAdapter().load(source)` or `ClaudeExportAdapter().load(source)` on a `.json` file whose root is `null`, `123`, or `true` raises an uncaught `TypeError: 'NoneType' object is not iterable` (bypassing `AdapterError`); if the root is a JSON object `{"a": 1}`, `for conv in data:` iterates over string keys `"a"`, silently skipping `if not isinstance(conv, dict): continue` and returning 0 documents without raising `AdapterError`.

### 4. `vecgrep/mcp/server.py:2286` — Private re-export import of `TransportSecuritySettings`
- **Rule**: `reportPrivateImportUsage` (`"TransportSecuritySettings" is not exported from module "mcp.server.fastmcp.server"`)
- **What happens**: `_LoopbackFastMCP.__init__` executes `from mcp.server.fastmcp.server import TransportSecuritySettings`. `mcp.server.fastmcp.server` imports `TransportSecuritySettings` from `mcp.server.transport_security` without re-exporting it in `__all__`.
- **Triggering input**: Instantiating `_LoopbackFastMCP(name="vecgrep")` with `transport_security=None` under any `mcp` release that drops the unused import in `mcp.server.fastmcp.server`. Importing from `mcp.server.transport_security` uses the public export.

### 5. `tests/test_round2_audit_fixes.py:238, 312, 320, 324` — Unimported `VecgrepService` + wrong `_score_cache` key type in test
- **Rules**: `reportUndefinedVariable` (L238, L312), `reportArgumentType` (L320, L324)
- **What happens**:
  1. Lines 238 and 312 annotate `svc: VecgrepService = ...` without importing `VecgrepService` (also flagged by Ruff `F821`).
  2. Line 320 seeds `rerank_mod._score_cache[("model", "q", "chunk")] = 0.99` using a 3-tuple key, whereas `vecgrep/backend/rerank.py:39` defines `_score_cache: OrderedDict[str, float]` keyed by SHA-256 hex strings from `_score_key(model_name, query, text)`.
  3. Line 324 constructs `GoldCase(..., want=[str(p)])` with a `list[str]` instead of `tuple[str, ...]`.
- **Triggering input**: Calling `typing.get_type_hints()` on `test_round2_audit_fixes` raises `NameError: name 'VecgrepService' is not defined`; and any code inspecting `_score_cache` keys as `str` encounters a `tuple[str, str, str]`.

### 6. `vecgrep/backend/main.py:488` — `intent` bound and consumed across separate `if tailnet_login:` blocks 150 lines apart
- **Rule**: `reportPossiblyUnboundVariable` (`"intent" is possibly unbound` in `standard` mode)
- **What happens**: In `oauth_unlock_page` (`vecgrep/backend/main.py:338`), `intent` is assigned only inside `if tailnet_login:`. After 150 lines of HTML construction, line 487 checks `if tailnet_login:` again and passes `intent` into `_set_login_intent_cookie(resp, intent)`.
- **Triggering input**: Currently `tailnet_login` is not mutated between lines 338 and 487, so it does not crash today, but `intent` is never initialized before the branch. Initializing `intent: dict[str, Any] | None = None` (and guarding `if tailnet_login and intent is not None:`) eliminates the correlated-branch trap.

### 7. `vecgrep/backend/timeline.py:84, 87` — Builtin function `callable` used as a type annotation (`callable | None`)
- **Rule**: `reportGeneralTypeIssues` (`Expected class but received "(obj: object, /) -> TypeIs[(...) -> object]"`)
- **What happens**: `build_timeline` declares `payload_for: callable` and `source_text_for: callable | None = None`. `callable` is a builtin function, not `collections.abc.Callable`.
- **Triggering input**: Evaluating runtime type hints via `typing.get_type_hints(build_timeline)` (used by FastAPI, Pydantic, or doc/schema generators) raises `TypeError: unsupported operand type(s) for |: 'builtin_function_or_method' and 'NoneType'`.

### 8. `vecgrep/eval/harness.py:68, 109` — Unvalidated string sentinel leaks into `Corpus.decay_half_life_days`
- **Rule**: `reportArgumentType` (`Argument of type "float | str | None" cannot be assigned to parameter "decay_half_life_days" of type "float | None" in function "__init__"`)
- **What happens**: `build_side_by_side` accepts `decay_half_life_days: float | None | str = "inherit"`. Line 110 only checks `if decay_half_life_days == "inherit":`; any other string passed by a caller falls through the `else` branch and is assigned directly to `Corpus(..., decay_half_life_days=decay_half_life_days)`.
- **Triggering input**: Calling `build_side_by_side(..., decay_half_life_days="none")` or `"30"` writes a raw `str` into `Corpus.decay_half_life_days`, which subsequently crashes search decay math (`age_days / half_life`) with `TypeError: unsupported operand type(s) for /: 'float' and 'str'`. Narrowing the signature to `float | None | Literal["inherit"] = "inherit"` (and raising `ValueError` on any unexpected string) prevents this.

### 9. `vecgrep/backend/ingestion/visual_processor.py:45` — Float `scale` passed to `pypdfium2` `page.render(scale: int)`
- **Rule**: `reportArgumentType` (`Argument of type "float" cannot be assigned to parameter "scale" of type "int" in function "render"`)
- **What happens**: Line 43 computes `scale = min(2.0, 1800 / max(width, height))` (`float`) and passes it to `page.render(scale=scale)`. While `pypdfium2`'s runtime Ctypes bindings accept a float today, its type stub declares `scale: int = 1` (and `float` annotation is missing from the stub).
- **Triggering input**: Rendering any PDF page via `extract_pdf_page_images(file_path)`. Verifying `scale > 0` and typing/suppressing the stub mismatch cleanly documents the `pypdfium2` float scale contract.

### 10. `scripts/rerank_benchmark.py:47-49` (and 3 test files) — Unchecked `None` from `importlib.util.spec_from_file_location`
- **Rules**: `reportArgumentType` (`spec` may be `None`), `reportOptionalMemberAccess` (`spec.loader` may be `None`)
- **What happens**: `spec = importlib.util.spec_from_file_location(...)` returns `ModuleSpec | None`, and `spec.loader` is `Loader | None`. Calling `importlib.util.module_from_spec(spec)` and `spec.loader.exec_module(mod)` without asserting `spec is not None and spec.loader is not None` raises an unhelpful `AttributeError: 'NoneType' object has no attribute 'name'` if the path is missing or invalid.
- **Triggering input**: Running `rerank_benchmark.py` (or the test loaders in `tests/test_weekly_eval_alert.py:24`, `tests/test_oauth_consent_redirects.py:373`, `tests/test_rerank_benchmark.py:13`) when the target script path does not resolve to a loadable `.py` file.

---

## 6. Target Mode Assessment: Why `"standard"` Beats `"basic"` Here

I disagree with targeting `"basic"` and recommend **`"standard"`** instead:

1. **Cost is virtually identical (342 vs 339 errors — a delta of 3 lines across 243 files / 49k LOC)**:
   - Switching `typeCheckingMode` from `"basic"` to `"standard"` adds **only 3 diagnostics** across the entire repository:
     - `vecgrep/backend/main.py:488` (`reportPossiblyUnboundVariable` on `intent` — Finding #6 above, a genuine 150-line correlated-branch hazard)
     - `tests/test_ollama_backend.py:350` (`reportFunctionMemberAccess` x2 on a monkeypatched test function attribute)
2. **What `"standard"` turns on that `"basic"` leaves off**:
   - `reportPossiblyUnboundVariable` (catches variables used after conditional initialization — probably the single highest-signal check for Python refactoring regressions)
   - `reportFunctionMemberAccess`
   - `reportInvalidTypeVarUse`
   - `reportUnsafeMultipleInheritance`
3. **`"standard"` is what bare `pyright` runs by default**:
   - In Pyright 1.1.408, omitting `typeCheckingMode` runs `"standard"` (including in the initial 483-error probe). Choosing `"basic"` explicitly steps down from Pyright's default to save 3 lines of fixes while blinding the gate to `reportPossiblyUnboundVariable`.
4. **Why `"strict"` is the wrong target**:
   - `"strict"` explodes to **16,574 errors**, ~98% of which are `reportUnknownVariableType`, `reportUnknownMemberType`, `reportUnknownArgumentType`, and `reportMissingParameterType` on untyped JSON dicts and pytest fixtures. Cleaning that up would require massive churn with near-zero bug yield.

**Recommendation**: Pin `pyright[nodejs]==1.1.408`, install full extras in the gate environment (`pip install -e ".[dev,mcp,code,openai,watch]"` or include the type-checked runtime dependencies in `dev`), exclude `tests/fixtures/code_symbols/broken.py`, and set `typeCheckingMode = "standard"` (or `"basic"` with `reportPossiblyUnboundVariable = "error"` if you want the configuration label to say `"basic"`).

---

## 7. Proposed Phase 1 Execution Plan (Awaiting Go-Ahead)

1. **Config & Ruff Cleanup**:
   - Add `[tool.pyright]` in `pyproject.toml` (or `pyrightconfig.json`), pin `pyright[nodejs]==1.1.408` in `dev` extra, exclude `tests/fixtures/code_symbols/broken.py`.
   - Fix the 14 `ruff check .` errors back to 0.
2. **Bug Fixes (each in its own commit with a failing test first)**:
   - Commit A: Fix `TokenError` import in `vecgrep/backend/auth/provider.py:140` + unit test for `exchange_authorization_code` when `consume_authorization_code` returns `None`.
   - Commit B: Fix `BM25SqliteStore._sql_filter_clauses` in `vecgrep/backend/store/bm25_sqlite.py:135-153` (handle non-float date strings / JSON payload `doc_timestamp` instead of non-existent column `c.doc_timestamp`) + unit test on `BM25SqliteStore.search(..., filter_clauses=["after:..."])`.
   - Commit C: Guard non-list top-level JSON in `chatgpt_export.py:89` and `claude_export.py:71` (raise `AdapterError` when root JSON is not a `list`) + unit tests.
   - Commit D: Validate `decay_half_life_days` in `vecgrep/eval/harness.py` (`Literal["inherit"]` + runtime check) and fix `callable` -> `Callable` in `vecgrep/backend/timeline.py` + unit tests.
3. **Annotation & Narrowing Commits by Module Cluster (no behavior change)**:
   - Cluster 1: `vecgrep/backend/service.py`, `assembly.py`, `explorer.py`, `main.py`, `api/routes.py`, `cli/main.py`
   - Cluster 2: `vecgrep/backend/auth/*` and `vecgrep/mcp/*`
   - Cluster 3: `vecgrep/backend/store/*`, `vecgrep/backend/rerank.py`, `vecgrep/backend/embed/*`, `vecgrep/backend/ingestion/*`, `scripts/rerank_benchmark.py`
   - Cluster 4: `tests/*` (`test_oauth_*`, `test_code_symbol*`, `test_write_*`, `test_rerank_*`, `test_round2_audit_fixes.py`, etc.)
4. **Documentation & Gate Verification**:
   - Document the exact `pyright` command in `docs/DEVELOPING.md` next to `ruff check .`.
   - Run full verification pass (pyright 0 errors, pytest full suite, deterministic eval across both stores and all 3 modes, wheel build + isolated import, parent-fail/child-pass proof for each bug commit).
