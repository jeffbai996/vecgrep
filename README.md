# vecgrep

> grep for meaning, not keywords.

`vecgrep` is a local-first semantic search engine for any corpus you throw at it. Drop in documents — text, markdown, PDFs, URLs — and search by concept instead of exact words. Runs on your machine, no cloud roundtrip required.

```
$ vecgrep search "what did we decide about rate hikes" --corpus notes
papers/2026-q1-review.md  92%   "...the committee opted to hold rates steady..."
notes/meeting-mar-12.md   88%   "...rate path discussion landed on no change..."
```

## Why

Keyword grep is brittle. You search for "rate hikes," miss the doc that says "monetary tightening." Embeddings fix that — semantically equivalent text scores high regardless of wording. `vecgrep` wraps that into a tool that feels like a CLI utility, not a research notebook.

## Status

Early. v0.1 is a working MVP: index, search, named corpora, web UI, CLI. API is unstable until v1.

## Install

```bash
pip install vecgrep
```

You also need [Ollama](https://ollama.com) running locally for the default embedding backend:

```bash
ollama pull nomic-embed-text
ollama serve
```

If you'd rather use OpenAI, set `OPENAI_API_KEY` and `vecgrep` will fall back to `text-embedding-3-small` automatically.

## Quickstart

```bash
# Index a folder
vecgrep index ./my-docs --corpus papers

# Index a URL
vecgrep index https://example.com/article --corpus web

# Search
vecgrep search "missile guidance systems" --corpus papers --top 10

# Manage corpora
vecgrep corpora list
vecgrep corpora delete papers

# Launch the web UI
vecgrep serve
```

## How it works

```
docs ──▶ adapters ──▶ chunkers ──▶ embed ──▶ qdrant
                                              │
                                          search ──▶ ranked chunks + context
```

- **Adapters** convert source formats (plaintext, markdown, PDF, URL) to text
- **Chunkers** slice text — `SentenceWindowChunker` is the default; `FixedTokenChunker` available
- **Embed backends** are pluggable: Ollama (default) or OpenAI (fallback)
- **Qdrant** stores vectors locally at `~/.vecgrep/qdrant/` — no Docker needed

## Storage layout

```
~/.vecgrep/
├── qdrant/         # vector store
└── corpora.json    # named corpus metadata
```

`--ephemeral` mode (CLI flag or UI toggle) skips disk entirely — useful for one-shot searches over a folder you don't want to persist.

## Configuration

`vecgrep` reads from `~/.vecgrep/config.json` and environment variables. Env vars win.

| Variable | Default | Notes |
|---|---|---|
| `VECGREP_HOME` | `~/.vecgrep` | Storage root |
| `VECGREP_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `VECGREP_EMBED_MODEL` | `nomic-embed-text` | Ollama model |
| `OPENAI_API_KEY` | unset | If set, used as fallback when Ollama is down |

## Roadmap

- v0.1 — MVP: text/markdown/PDF/URL adapters, CLI, web UI, named corpora
- v0.2 — Claude/ChatGPT export adapters, hybrid keyword+vector search, reranking
- v0.3 — Incremental indexing, file-watcher mode, multi-tenant corpora

## Contributing

Issues and PRs welcome. Keep it focused — `vecgrep` is a tool, not a framework.

## License

MIT.
