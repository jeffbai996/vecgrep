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

| Model | Dim |
|---|---|
| `nomic-embed-text` (Ollama) | 768 |
| `text-embedding-3-small` (OpenAI) | 1536 |

Each corpus pins the embedding model + dim it was created with. Mixing models within a corpus is rejected.
