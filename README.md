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
pip install vecgrep                # base — Ollama embedding, hybrid search
pip install "vecgrep[openai]"      # also: OpenAI embedding fallback
pip install "vecgrep[rerank]"      # also: cross-encoder reranking (~hundreds of MB, torch)
pip install "vecgrep[watch]"       # also: file watcher for `vecgrep watch`
pip install "vecgrep[mcp]"         # also: MCP server for Claude Desktop / Cursor
```

You also need [Ollama](https://ollama.com) running locally for the default embedding backend:

```bash
ollama pull nomic-embed-text
ollama serve
```

If you'd rather use OpenAI, install with `[openai]`, set `OPENAI_API_KEY`, and `vecgrep` will fall back to `text-embedding-3-small` automatically when Ollama isn't reachable.

## Quickstart

```bash
# Index a folder (mixed file types are fine; incremental — unchanged files skipped)
vecgrep index ./my-docs --corpus papers

# Re-embed everything regardless of content hash
vecgrep index ./my-docs --corpus papers --force

# Index a URL — vecgrep fetches and strips boilerplate
vecgrep index https://example.com/article --corpus web

# Watch a folder and re-index on change
vecgrep watch ./my-docs --corpus papers

# Hybrid search (default — BM25 + vector fused via RRF), top 10
vecgrep search "missile guidance systems" --corpus papers --top 10

# Pure-vector or pure-BM25 if you want to A/B
vecgrep search "rate hikes" --mode vector
vecgrep search "FOMC" --mode bm25

# Cross-encoder reranking on the candidate pool — slower, more accurate
vecgrep search "what did we decide about rates" --rerank

# Filter results — repeatable, all ANDed
vecgrep search "rate hikes" --filter "source:*2026*.md" --filter "corpus:papers"

# Search across every corpus you have
vecgrep search "rate hikes"

# Don't persist — index, query, throw away
vecgrep index ./scratch --corpus tmp --ephemeral

# Move a corpus between machines
vecgrep corpora export papers --out papers.tar.gz
vecgrep corpora import papers.tar.gz --rename papers-from-laptop

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
| `discord_jsonl` | `.jsonl` files where each line is a Discord message (DiscordChatExporter format or any one-message-per-line JSONL with `author` + `content`) |
| `claude_export` | Anthropic data export `conversations.json` — one Document per conversation |
| `chatgpt_export` | OpenAI data export `conversations.json` — one Document per conversation, follows main thread |

Pointing `vecgrep index` at a directory walks it recursively and dispatches each file to the matching adapter. Unrecognized files are skipped silently — no errors.

## How it works

```
                          ┌──▶ vector (qdrant) ──┐
docs ──▶ adapters ──▶ chunkers ──┤                      ├──▶ RRF ──▶ [rerank] ──▶ top-k
                          └──▶ bm25 (inverted) ──┘
```

- **Adapters** convert source formats to text. They run once per source; chunkers handle slicing.
- **Chunkers** slice text into overlapping windows. `SentenceWindowChunker` is the default — 3 sentences with 1-sentence overlap. `FixedTokenChunker` (tiktoken-backed) is the alternate for code, logs, anything where sentence boundaries are noisy.
- **Embed backends** are pluggable. Ollama (`nomic-embed-text`, 768-dim) is the default. OpenAI (`text-embedding-3-small`, 1536-dim) takes over when Ollama is unreachable and `OPENAI_API_KEY` is set.
- **Qdrant** runs in embedded mode (no server, no Docker) at `~/.vecgrep/qdrant/`. Each named corpus is its own collection.
- **BM25** index runs alongside Qdrant, persisted as a pickle per corpus. Tokenizer splits identifiers (`sharpe_ratio` → `sharpe`, `ratio`) so code search isn't blind to underscore- or camelCase-style naming.
- **Hybrid retrieval** is the default. Each retriever returns its top 50 candidates; their ranks are fused via Reciprocal Rank Fusion (`score = Σ 1/(60+rank)`). Pure-vector or pure-BM25 are available with `--mode vector` / `--mode bm25`.
- **Cross-encoder reranker** (`--rerank`, off by default) rescores the candidate pool with `BAAI/bge-reranker-base`. Local, ~30ms for 50 chunks on CPU. Lazy-loaded — the heavy `torch` import only happens when you ask for it.

Each corpus pins the embedding backend and dimension at index time, and refuses to mix models within itself. If you change embedding model, recreate the corpus.

## Storage layout

```
~/.vecgrep/
├── qdrant/         # vector store, one collection per corpus
├── bm25/           # BM25 inverted index, one pickle per corpus
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

`vecgrep serve` boots the FastAPI server and serves a single-page React UI from the same port. Index forms, corpus list with delete, search bar with top-k slider, mode toggle (hybrid/vector/bm25), reranker checkbox, results with surrounding context and the matched chunk highlighted. It's deliberately small — every action it supports has a CLI equivalent.

## MCP server

`vecgrep` ships an MCP server so Claude Desktop, Cursor, or any MCP-aware client can search your corpora as a tool. Install with the extra and run over stdio:

```bash
pip install "vecgrep[mcp]"
vecgrep mcp
```

Tools exposed:
- `search(query, corpus?, top_k?, mode?, rerank?)` — returns ranked chunks with surrounding context as JSON
- `list_corpora()` — every corpus and its stats
- `get_corpus(name)` — one corpus's full metadata including source list

Wire it into Claude Desktop with this snippet in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "vecgrep": {
      "command": "vecgrep",
      "args": ["mcp"]
    }
  }
}
```

Then index a corpus once (`vecgrep index ./my-notes --corpus notes`), and the model can call `search("rate hikes", corpus="notes")` instead of asking you to paste documents.

## Roadmap

The plan is short and ordered. Make search good first, connect it to where you actually work second, polish later.

**v0.2 — search quality (in progress)**
- ✅ Hybrid search (BM25 + vector + RRF), default on. Pure vector misses exact-token matches like CVE numbers, ticker symbols, function names; BM25 nails them.
- ✅ Cross-encoder reranking (`--rerank`, off by default). Local, ~30ms for 50 chunks on CPU.

**v0.3 — connect it**
- ✅ MCP server: expose `vecgrep` as a tool to Claude / Cursor / any MCP client. Index a corpus once, let your assistant retrieve from it instead of stuffing context.
- ✅ Discord JSONL adapter: drop in chat exports, search them as a corpus.
- ✅ Claude / ChatGPT export adapters: search your own conversation history.

**v0.4 — operate it like a tool**
- ✅ Incremental indexing — content-hash skip
- ✅ `vecgrep watch` — file-watcher
- ✅ `vecgrep corpora export/import` — tarball roundtrip
- ✅ Search filters — `--filter source:GLOB`, `corpus:NAME`, `meta.KEY=VALUE`

**Later (unsorted)**
EPUB/DOCX/code-aware adapters, OCR fallback, single-binary distribution, multi-tenant mode, query-aware chunking, `--explain`. See [docs/IDEAS.md](docs/IDEAS.md) for the long list.

**v1.0 — stability**
- Locked HTTP API and CLI surface
- Migration tooling for embedding model upgrades
- Documented plugin API for adapters, chunkers, embed backends, rerankers

PRs welcome on anything in v0.2 or v0.3. For larger items in **Later**, open an issue first.

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
