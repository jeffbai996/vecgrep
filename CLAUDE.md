# vecgrep — Claude Code working notes

## What this is

A local-first semantic search tool. CLI-first, web UI on the side. OSS — no PII, no leaked context, no portfolio data, no internal hostnames in source. Generic-only.

## Architecture cheat sheet

```
vecgrep/backend/
├── main.py              FastAPI app, mounts /api and serves frontend/dist
├── config.py            settings + ~/.vecgrep paths
├── ingestion/
│   ├── adapters/        text, markdown, pdf, url -> str
│   └── chunkers/        SentenceWindowChunker (default), FixedTokenChunker
├── embed/               OllamaBackend (default), OpenAIBackend (fallback)
├── store/               QdrantStore wrapper (embedded mode)
└── api/                 routes: search, corpus, config

vecgrep/cli/             Click commands that hit the local API
vecgrep/frontend/        React + Tailwind, single page, built into dist/
```

## Design rules

- Flat over nested. No abstract base classes for things with one impl.
- Adapter and chunker interfaces are the *only* spots designed for extension — everything else is concrete.
- Config in one place (`backend/config.py`), env vars override JSON.
- `~/.vecgrep/` is the only persistence location. `--ephemeral` skips it.
- Qdrant runs in **embedded** mode (`path=...`), no server, no Docker.
- Errors that the user can fix (Ollama not running, model not pulled) get plain English messages with the exact command to fix.
- **MCP exposes no DIRECT mutation.** The stdio MCP server is read-only (`search`, `list_corpora`, `get_corpus`). The HTTP transport adds the `propose_*` family (`propose_write`, `propose_edit`, `propose_delete`) — but these write/remove **nothing**: each creates an inert pending proposal that a human turns into a real write/delete via `vecgrep confirm` (off-protocol). So even over a leaked/unauthed `/mcp` endpoint an agent can only *suggest* a change, never land one — and a `propose_delete` removes a doc + its embeddings only after that human confirm. Direct/immediate `index` and `delete` stay on the REST routes, which operators keep behind tailnet / VPN / localhost. The propose corpus set is default-deny (`VECGREP_PROPOSE_ALLOWED_CORPORA`), so a proposal can't even target a corpus the operator hasn't opened.
  - `propose_edit` has two modes: **full overwrite** (`content=` the whole new body) or **surgical patch** (`old_str=` + `new_str=` — a str-replace of one *unique* span, so you don't re-send an 8KB doc to fix one line). The patch is a hard error if `old_str` is missing or appears more than once (add surrounding context to disambiguate); it touches the body only, frontmatter is preserved, and it flows through the identical proposal/confirm/corpus-gate path. Example — fix a date without a regen: `propose_edit(doc_id="notes-007", old_str="July 25", new_str="July 17")`.

## Don'ts

- No telemetry, no phone-home, no analytics
- Don't hardcode tickers, hostnames, or anything from global CLAUDE.md memories
- No emojis in source code or commits

## Tests

- `pip install -e ".[dev]"` then `pytest`. Hermetic: no network, no Ollama, no shared `~/.vecgrep` — each test gets its own `VECGREP_HOME` under `tmp_path`.
- Stub backend is `tests.conftest.StubEmbed`. Inject via `svc._backend_cache["auto"] = CachedBackend(stub, svc._embed_cache)` (the fixture already does this).
- New features must come with at least one test against the service layer. That's where bugs hide; FastAPI / Click plumbing tests itself.
- BM25 has a known degenerate case (single-doc corpus, or all docs share the term → IDF=0 → score=0). The store falls back to token-overlap counting in that case. Don't remove the fallback — covered by `tests/test_bm25_store.py::test_persistence_round_trip`.

## What's next (committed)

Build order is fixed. See `docs/IDEAS.md` for the long unsorted list.

**v0.2 — search quality**
1. Hybrid search: BM25 + vector via Reciprocal Rank Fusion. Add `rank-bm25` (or inline ~50 LOC), per-corpus inverted index alongside Qdrant. `--mode {hybrid,vector,bm25}`, default hybrid.
2. Cross-encoder reranking, `--rerank` flag, off by default. `BAAI/bge-reranker-base`. Optional `vecgrep[rerank]` extra.

**v0.3 — connect**
3. MCP server: stdio transport, expose `search` and `list_corpora`.
4. Discord JSONL adapter: reads exported Discord chat logs (DiscordChatExporter format), channel-as-source.
5. Claude / ChatGPT export adapters (the v2 hook the spec called out).

Anything else (file-watcher, EPUB/DOCX, OCR, single-binary, multi-tenant, query-aware chunking) is **Later** — capture in `docs/IDEAS.md`, don't pull forward.

## Embedding dimensions reference

| Model | Dim |  |
|---|---|---|
| `bge-m3` (Ollama) | 1024 | **default** |
| `nomic-embed-text` (Ollama) | 768 | lighter/faster alt |
| `mxbai-embed-large` (Ollama) | 1024 | alt |
| `text-embedding-3-small` (OpenAI) | 1536 | fallback when Ollama unreachable |

Each corpus pins the embedding model + dim it was created with. Mixing models within a corpus is rejected.
