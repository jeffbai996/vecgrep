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
- No tests in initial scaffold (per spec)
- No Claude/ChatGPT export adapters yet (v2) — leave the hook
- Don't hardcode tickers, hostnames, or anything from CLAUDE.md memories
- No emojis in source code or commits

## Embedding dimensions reference

| Model | Dim |
|---|---|
| `nomic-embed-text` (Ollama) | 768 |
| `text-embedding-3-small` (OpenAI) | 1536 |

Each corpus pins the embedding model + dim it was created with. Mixing models within a corpus is rejected.
