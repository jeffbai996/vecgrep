# vecgrep

> grep for meaning, not keywords

`vecgrep` is a local-first semantic search engine for any corpus you throw at it. Drop in documents — text, markdown, PDFs, URLs — and search by concept instead of exact words. Runs on your machine, no cloud roundtrip required.

```
$ vecgrep search "what did we decide about rate hikes" --corpus notes
[1]  92.4%  notes  papers/2026-q1-review.md
     ... the committee discussed the path forward and
     the committee opted to hold rates steady through Q2
     citing softening labor data ...

[2]  88.1%  notes  notes/meeting-mar-12.md
     ... after some back and forth
     the rate path discussion landed on no change
     for the rest of the year ...
```

## Why

Keyword `grep` is brittle. You search for `rate hikes`, you miss the doc that says `monetary tightening`. Embeddings fix that — semantically equivalent text scores high regardless of wording.

`vecgrep` wraps that into a tool that feels like a CLI utility, not a research notebook. No cloud account, no Docker, no Postgres extension, no ceremony. Index a folder, run a query, get ranked chunks back with surrounding context.

The closest equivalents — `txtai`, `chroma`, `LlamaIndex` — are libraries you import. `vecgrep` is a binary you run.

## Status

Early. v0.1 is a working MVP: index, search, named corpora, web UI, CLI. The public API (HTTP + CLI flags) is unstable until v1.0 — expect breaking changes within v0.x.

## Install

```bash
pip install vecgrep
```

You also need [Ollama](https://ollama.com) running locally for the default embedding backend:

```bash
ollama pull nomic-embed-text
ollama serve
```

If you'd rather use OpenAI, set `OPENAI_API_KEY` and `vecgrep` will fall back to `text-embedding-3-small` automatically when Ollama isn't reachable.

## Quickstart

```bash
# Index a folder (mixed file types are fine)
vecgrep index ./my-docs --corpus papers

# Index a URL — vecgrep fetches and strips boilerplate
vecgrep index https://example.com/article --corpus web

# One-shot search across a corpus, top 10
vecgrep search "missile guidance systems" --corpus papers --top 10

# Search across every corpus you have
vecgrep search "rate hikes"

# Don't persist — index, query, throw away
vecgrep index ./scratch --corpus tmp --ephemeral

# Manage corpora
vecgrep corpora list
vecgrep corpora delete papers

# Launch the web UI on http://127.0.0.1:8765
vecgrep serve
```

## What it indexes

| Adapter | Inputs |
|---|---|
| `plaintext` | `.txt`, `.log`, `.csv`, `.tsv`, `.rst`, `.org`, `.tex`, `.srt`, extensionless |
| `markdown` | `.md`, `.markdown`, `.mdx` |
| `pdf` | `.pdf` (text layer only — scanned PDFs need OCR first) |
| `url` | `http://`, `https://` — strips `<script>`, `<style>`, etc., keeps prose |

Pointing `vecgrep index` at a directory walks it recursively and dispatches each file to the matching adapter. Unrecognized files are skipped silently — no errors.

## How it works

```
docs ──▶ adapters ──▶ chunkers ──▶ embed ──▶ qdrant
                                                │
                                            search ──▶ ranked chunks + context
```

- **Adapters** convert source formats to text. They run once per source; chunkers handle slicing.
- **Chunkers** slice text into overlapping windows. `SentenceWindowChunker` is the default — 3 sentences with 1-sentence overlap. `FixedTokenChunker` (tiktoken-backed) is the alternate for code, logs, anything where sentence boundaries are noisy.
- **Embed backends** are pluggable. Ollama (`nomic-embed-text`, 768-dim) is the default. OpenAI (`text-embedding-3-small`, 1536-dim) takes over when Ollama is unreachable and `OPENAI_API_KEY` is set.
- **Qdrant** runs in embedded mode (no server, no Docker) at `~/.vecgrep/qdrant/`. Each named corpus is its own collection.

Each corpus pins the embedding backend and dimension at index time, and refuses to mix models within itself. If you change embedding model, recreate the corpus.

## Storage layout

```
~/.vecgrep/
├── qdrant/         # vector store, one collection per corpus
├── corpora.json    # named-corpus metadata
└── config.json     # optional, env vars override
```

`--ephemeral` mode (CLI flag or UI toggle) keeps everything in memory and skips both files and the vector store on disk. Useful for a one-shot grep over a directory you don't want polluting your indexed corpora.

## Configuration

`vecgrep` reads from `~/.vecgrep/config.json` and environment variables, in that order. Env vars win.

| Variable | Default | Notes |
|---|---|---|
| `VECGREP_HOME` | `~/.vecgrep` | Storage root |
| `VECGREP_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `VECGREP_EMBED_MODEL` | `nomic-embed-text` | Ollama model |
| `VECGREP_OPENAI_EMBED_MODEL` | `text-embedding-3-small` | OpenAI model |
| `OPENAI_API_KEY` | unset | If set, used as fallback when Ollama is down |
| `VECGREP_API_HOST` | `127.0.0.1` | API bind host |
| `VECGREP_API_PORT` | `8765` | API port |
| `VECGREP_TOP_K` | `5` | Default `--top` value |

## Web UI

`vecgrep serve` boots the FastAPI server and serves a single-page React UI from the same port. Index forms, corpus list with delete, search bar with top-k slider, results with surrounding context and the matched chunk highlighted. It's deliberately small — every action it supports has a CLI equivalent.

## Roadmap

**v0.2 — search quality**
- Hybrid search: BM25 keyword scoring fused with vector similarity (RRF)
- Cross-encoder reranking on top-k results (configurable, off by default)
- Per-corpus search filters: by source path glob, by metadata field
- Query rewriting: synonym expansion behind a flag

**v0.3 — ingestion breadth**
- Adapters for Claude/ChatGPT JSON exports — search your own conversations
- EPUB and DOCX adapters
- Code-aware adapter (tree-sitter) that respects function/class boundaries
- OCR fallback path for image-only PDFs (Tesseract optional dep)

**v0.4 — operate it like a tool**
- Incremental indexing: detect changed files since last index, re-embed only the diff
- File-watcher mode: `vecgrep watch ./docs --corpus live` keeps a corpus current
- `vecgrep export` / `vecgrep import` for shipping a corpus between machines
- Per-source TTL — auto-evict URLs older than N days

**v0.5 — power features**
- MCP server so Claude/Cursor/etc can query corpora as a tool
- Multi-tenant mode: namespaced corpora behind a token, for shared servers
- Faceted result clustering (group hits by source, time, or topic)
- `--explain` flag that dumps the embedding similarity decomposition

**v1.0 — stability**
- Locked HTTP API and CLI surface
- Migration tooling for embedding model upgrades
- Documented plugin API for adapters, chunkers, embed backends, rerankers

The roadmap is intentions, not commitments. PRs welcome on any of it; open an issue first if it's a big change.

## Why not just use X?

| Tool | Why vecgrep instead |
|---|---|
| `grep` / `ripgrep` | Lexical only — misses paraphrases |
| `chroma`, `txtai`, `LlamaIndex` | Libraries, not tools — you're writing Python before you can search |
| Cloud RAG SaaS | Your docs leave your machine. `vecgrep` is local-first by default. |
| `pgvector` + Postgres | Heavy. `vecgrep` is one `pip install`. |

Use the right tool for the job. `vecgrep` is for when you have a folder and a question.

## Contributing

Issues and PRs welcome. Keep it focused — `vecgrep` is a tool, not a framework.

See [docs/DEVELOPING.md](docs/DEVELOPING.md) for layout, dev loop, and extension points (adapters, chunkers, embed backends).

## License

MIT
