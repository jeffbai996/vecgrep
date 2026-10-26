# vecgrep

Local-first semantic search over text, Markdown, PDFs, URLs and chat exports: hybrid BM25 + vector retrieval through a CLI, HTTP API, web UI and MCP server.

![vecgrep web UI](docs/assets/web-ui.png)

## Install

The `vecgrep` project on PyPI is unrelated. Install from GitHub with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install "vecgrep[mcp] @ git+https://github.com/jeffbai996/vecgrep"
```

One-shot isolated run without installing:

```bash
uvx --from "vecgrep[mcp] @ git+https://github.com/jeffbai996/vecgrep" vecgrep --help
```

Extras (combine in the brackets): `openai`, `rerank`, `watch`, `mcp`, `code`, `visual`, `dev`. They are defined in `pyproject.toml`.

The default embedding backend is [Ollama](https://ollama.com) with `bge-m3`:

```bash
ollama pull bge-m3
ollama serve
```

With the `openai` extra and `OPENAI_API_KEY` set, vecgrep falls back to `text-embedding-3-small` when Ollama is unreachable.

## Usage

```bash
vecgrep index ./my-docs --corpus papers            # incremental; --force re-embeds, --ephemeral keeps nothing on disk
vecgrep index https://example.com/article --corpus web
vecgrep watch ./my-docs --corpus papers            # re-index on change
vecgrep watch ./a --also-watch ./b --corpus cli    # several producers, one corpus

vecgrep search "rate hikes" --corpus papers --top 10
vecgrep search "rate hikes" --mode vector          # or --mode bm25; default is hybrid
vecgrep search "rate hikes" --rerank               # cross-encoder rescoring
vecgrep search "rate hikes" --explain              # score decomposition
vecgrep search "rate hikes" --budget               # top 10 in full + one-line stubs
vecgrep search "rate hikes" --watch --interval 5   # re-run and diff source IDs
vecgrep search "rate hikes" --filter "source:*2026*.md" --filter after:7d   # repeatable, ANDed

vecgrep timeline "what happened during the deploy" --corpus chatlogs --filter after:7d
vecgrep related <chunk-id> --corpus chatlogs
vecgrep compare "deployment failures" --corpus chatlogs --a-after 60d --a-before 30d --b-after 30d
vecgrep stats chatlogs
vecgrep summarize chatlogs --after 7d
vecgrep chunk <corpus> <chunk_id>

vecgrep corpora list | delete <name> | export <name> --out f.tar.gz | import f.tar.gz --rename <name>
vecgrep corpora migrate papers --to-backend openai --to-model text-embedding-3-small
vecgrep corpora decay chatlogs --half-life 14      # or --off
vecgrep cache stats | clear --identity ollama:bge-m3
vecgrep bm25 rebuild <corpus>

vecgrep status [--json]       # daemon, auth, per-corpus counts
vecgrep doctor [--json]       # registry vs. vector store drift
vecgrep backup create | list | verify <file> | restore <file> --confirm <id>
vecgrep config                # effective settings with provenance
vecgrep init --yes            # create config.json
vecgrep serve                 # web UI, /api/* and /mcp on 127.0.0.1:8765
vecgrep mcp                   # MCP over stdio
```

Run `vecgrep <command> --help` for all options.

### Filters

`--filter` grammar: `date:<iso|today>`, `after:<iso|7d|24h|2w>`, `before:<iso>`, `channel:<name>`, `source:<glob>`, `source_path:<glob>`, `corpus:<name>`, `speaker:<name>`, `bot:true|false`, `has:code|table|link`, `meta.KEY:<value>`. A leading `-` inverts a filter. Recognized filters fail closed: a malformed value returns zero results.

### Adapters and chunkers

Directories are walked recursively and each file goes to the matching adapter; unrecognized files are skipped.

| Adapter | Inputs |
|---|---|
| `plaintext` | `.txt`, `.log`, `.csv`, `.tsv`, `.rst`, `.org`, `.tex`, `.srt`, extensionless |
| `markdown` | `.md`, `.markdown`, `.mdx` |
| `pdf` | `.pdf` (text layer only) |
| `url` | `http://`, `https://` |
| `discord_jsonl` | `.jsonl`, one message per line with `author` and `content` |
| `claude_export`, `chatgpt_export` | `conversations.json` from the provider's data export |
| `image` | `.png`, `.jpg`, `.jpeg`, `.webp`, `.tif`, `.tiff`, `.bmp`; needs `visual` ([docs/VISUAL_ENRICHMENT.md](docs/VISUAL_ENRICHMENT.md)) |

`--chunker` on `index` and `watch`: `sentence_window` (default), `fixed_token`, `markdown_section`, `code_symbol` (TypeScript needs `code`), `turn_window`. `--include '*.md'` limits a directory to matching names.

## Retrieval

- Hybrid search fuses the top 50 of vector and BM25 results with Reciprocal Rank Fusion, `score = Σ w / (60 + rank)`. BM25 weight is `VECGREP_BM25_WEIGHT`.
- Optional per-corpus recency decay multiplies the fused score by `0.5 ** (age_days / half_life)` before the top-k cut. The date is `doc_timestamp`, parsed at index time. Undated chunks are not penalized.
- Overlapping same-source hits are collapsed before top-k; MMR favors distinct evidence.
- Displayed `%` is a calibrated relevance estimate; ranking uses the underlying fused score. Reranker output is used directly when reranking is on.
- Reranker: `BAAI/bge-reranker-v2-m3`, override with `VECGREP_RERANKER`. While warming or after a failure, results are returned hybrid-ranked and marked not reranked.
- Each corpus pins its embedding backend, model and dimension at index time; corpora on different models coexist. Change a model with `corpora migrate`.

## Storage

Rooted at `VECGREP_HOME` (default `~/.vecgrep`):

| Path | Contents |
|---|---|
| `qdrant/` | Embedded Qdrant, one collection per corpus |
| `bm25/` | One `.pkl` or SQLite `.db` per corpus (`VECGREP_BM25_BACKEND`) |
| `locks/` | Cross-process corpus and registry leases |
| `mutations/` | Pending crash-recovery intents (normally empty) |
| `corpora.json` | Corpus registry |
| `config.json` | Optional settings; env vars override |

Search takes a shared corpus lease; index, delete and writes take an exclusive lease. A per-corpus intent journal makes Qdrant, BM25 and `corpora.json` one recoverable commit; startup rolls back or forward interrupted work.

## Configuration

Precedence: environment variables, then `config.json`, then defaults. Variable names, config keys and defaults are in `vecgrep/backend/config.py`; `vecgrep config` prints the effective values.

| Variable | Purpose |
|---|---|
| `VECGREP_HOME` | Storage root |
| `VECGREP_OLLAMA_URL`, `VECGREP_OLLAMA_FALLBACK_URL` | Ollama endpoints |
| `VECGREP_EMBED_MODEL`, `VECGREP_OPENAI_EMBED_MODEL`, `OPENAI_API_KEY` | Embedding models and OpenAI fallback |
| `VECGREP_API_HOST`, `VECGREP_API_PORT` | API bind (default `127.0.0.1:8765`) |
| `VECGREP_API_TOKEN` | Bearer token for `/api/*`; required (32+ characters) for a non-loopback bind |
| `VECGREP_REST_ALLOWED_HOSTS` | Exact, port-qualified Host values for a trusted REST proxy; wildcards are rejected. Prefer `VECGREP_API_TOKEN` unless the proxy supplies a separate access boundary |
| `VECGREP_ADMIN_TOKEN` | Bearer token for `/api/admin/*`; without it admin needs a loopback peer and Host |
| `VECGREP_QDRANT_URL` | Qdrant server mode |
| `VECGREP_BM25_BACKEND`, `VECGREP_BM25_WEIGHT`, `VECGREP_BM25_COVERAGE_MODE` | BM25 storage, fusion weight, partial-match handling (`penalty` or `filter`) |
| `VECGREP_EMBED_CACHE_MAX_ROWS` | Embedding cache row cap (`<= 0` disables) |
| `VECGREP_ALIASES_FILE` | Entity alias map; see `docs/aliases.example.json` |
| `VECGREP_OAUTH_*`, `VECGREP_MCP_ALLOWED_HOSTS`, `VECGREP_MCP_ALLOWED_ORIGINS` | Remote MCP; see [docs/OAUTH.md](docs/OAUTH.md) |
| `VECGREP_DEFAULT_PROPOSE_CORPUS`, `VECGREP_PROPOSE_ALLOWED_CORPORA`, `VECGREP_PROPOSE_HOOK`, `VECGREP_DIRECT_WRITE_*` | Write workflows, below |

## MCP

`vecgrep mcp` serves stdio; `vecgrep serve` also serves streamable HTTP at `/mcp`. Intent presets and an evidence-expansion playbook: [docs/AGENT_SEARCH.md](docs/AGENT_SEARCH.md).

| Intent | Tools |
|---|---|
| Find evidence | `search`, `timeline`, `incident`, `browse`, `related`, `compare` |
| Expand and inspect | `get_chunk`, `get_source`, `stats`, `summarize_corpus`, `list_aliases` |
| Discover corpora | `list_corpora`, `get_corpus` |
| Direct mutation | `write`, `edit` |
| Human-confirmed mutation | `propose_write`, `propose_edit`, `propose_delete`, `propose_merge` |

`browse` requires a selector (channel, date, `since`/`until`, or source glob) and refuses whole-corpus dumps. On corpora with at least 10,000 chunks, MCP enables `rerank` and `budget` when the caller omits them; explicit values win.

```bash
claude mcp add vecgrep --scope user -- vecgrep mcp                                  # stdio
claude mcp add --scope user --transport http vecgrep https://your-server.example/mcp   # remote
```

Claude Desktop and Cursor: add `{"mcpServers": {"vecgrep": {"command": "vecgrep", "args": ["mcp"]}}}` to the client's MCP config. Pass `VECGREP_OLLAMA_URL` under `env` for a remote Ollama. Use the absolute binary path if `vecgrep` is not on `PATH`.

### Remote access

- `vecgrep serve --host 0.0.0.0` refuses to start without `VECGREP_API_TOKEN`. Without a token, `/api/*` accepts only loopback Host values on `VECGREP_API_PORT`.
- vecgrep serves HTTP only; terminate TLS in a reverse proxy.
- A reverse proxy should use `VECGREP_API_TOKEN`. List a tokenless proxy's Host in `VECGREP_REST_ALLOWED_HOSTS` only when it supplies a separate access boundary.
- Add a proxy hostname other than the OAuth issuer to `VECGREP_MCP_ALLOWED_HOSTS`, and a browser origin to `VECGREP_MCP_ALLOWED_ORIGINS`, or MCP rejects the request.
- Do not expose an OAuth-off `/mcp` publicly. For public clients set `VECGREP_OAUTH_ENABLED=1`, `VECGREP_OAUTH_ISSUER_URL` and `VECGREP_OAUTH_APPROVAL_TOKEN`, keep the default loopback bind, and route every public MCP and OAuth path (including discovery and `/oauth/unlock`) to `VECGREP_OAUTH_PUBLIC_PORT` (8766). Private clients stay on the API port. Scopes are `read` and `propose`; no grant bypasses human confirmation. Details: [docs/OAUTH.md](docs/OAUTH.md).

## Writing

Default path is propose, review, confirm:

1. `propose_write`, `propose_edit`, `propose_delete` or `propose_merge` creates an inert proposal under `$VECGREP_HOME/write/_pending`.
2. `vecgrep pending` lists proposals.
3. `vecgrep confirm <proposal-id>` commits and re-indexes; `vecgrep discard <proposal-id>` drops it.

Proposal corpora are default-deny: `claude-ai` unless `VECGREP_DEFAULT_PROPOSE_CORPUS` is set; open more with `VECGREP_PROPOSE_ALLOWED_CORPORA` (comma-separated). `VECGREP_PROPOSE_HOOK` runs a notification command; its failure does not affect the proposal. Edits are a full-body overwrite, a unique `old_str` to `new_str` patch, or a tags/source-kind change. `propose_merge` keeps the first ID, replaces its body, and deletes the others only after confirmation.

`VECGREP_DIRECT_WRITE_CORPUS=<name>` enables immediate MCP `write` and `edit` for one plain local corpus. The corpus is not in the tool schema. Writes are capped by `VECGREP_DIRECT_WRITE_MAX_BYTES` and `VECGREP_DIRECT_WRITE_MAX_PER_HOUR`; edits keep timestamped backups and refuse protected documents; there is no direct delete; a corpus with a write-through hook is refused.

`VECGREP_TEMP_CORPUS=<name>` enables MCP `write_temp` and `renew_temp` for one plain local corpus of short-lived documents. Each document needs an expiry (`3d`, `12h`, `2w` or an ISO date) no further out than `VECGREP_TEMP_MAX_TTL_DAYS` (default 90). Search hides a document once it expires and reports `expires_in` on live ones; `vecgrep temp sweep` and each temp write delete expired documents from that corpus and no other. It must differ from the direct-write corpus.

CLI without MCP:

```bash
vecgrep write notes "A durable note" --source-kind fact --tag example
vecgrep edit notes-<id> "Replacement body" --corpus notes
```

## Backup

Backups hold per-corpus Qdrant snapshots, the registry, non-secret config, aliases and write-through documents. Embedding caches and BM25 sidecars are excluded; BM25 is rebuilt from Qdrant payloads after restore. Restore verifies checksums, requires the exact backup ID, takes a pre-restore safety backup and rolls back on failure. The scheduler is off by default; retention prunes only scheduled backups.

## Notes

- **Embedded Qdrant allows one process at a time.** If `serve`, `watch` and the CLI run together, set the same `qdrant_url` in `config.json` for every process. Setting it only in one service's environment splits the processes across two stores. `vecgrep doctor` inspects only the backend its own config selects.
- **BM25 SQLite backend.** Convert and verify existing sidecars before selecting `sqlite`. Unmigrated or stale pickles block startup; missing or inconsistent SQLite indexes block lexical and hybrid search until `vecgrep bm25 rebuild <corpus>`.
- **Duplicate file IDs.** After a directory move or symlink change, `doctor` reports several IDs resolving to one path. `--fix` does not pick one: normalize the ingestion path, verify coverage, then remove legacy IDs with `VecgrepService.delete_source` so both indexes update. Indexing a single file removes its other IDs; directory indexes leave them.
- **Embedding cache cap.** Keep `VECGREP_EMBED_CACHE_MAX_ROWS` above the number of distinct chunk texts, or a repair re-embeds evicted entries. Every writing process evicts to its own cap, so set it in `config.json` for multi-process deployments.
- **Scanned PDFs** have no text layer; OCR them first or use `visual`.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests are hermetic: no Ollama, no network, a per-test `VECGREP_HOME`, a stub embedding backend. Layout and extension points: [docs/DEVELOPING.md](docs/DEVELOPING.md). Release notes: [CHANGELOG.md](CHANGELOG.md). Candidate work: [docs/IDEAS.md](docs/IDEAS.md). Contributions: [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
