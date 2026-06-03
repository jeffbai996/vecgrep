# vecgrep

> grep for meaning, not keywords

`vecgrep` is a local-first semantic search engine for any corpus you throw at it. Drop in documents — text, markdown, PDFs, URLs — and search by concept instead of exact words. Runs on your machine, no cloud roundtrip required.

![vecgrep web UI — index panel and corpus list on the left, search bar with mode toggle, side-by-side primer on vector vs BM25 retrieval](docs/assets/web-ui.png)

*The web UI: index a corpus from the sidebar, pick a mode (hybrid / vector / bm25), search. Each result shows confidence (high / soft / weak), which retriever found it (V semantic, K keyword, VK both), and the matched chunk in context. Tucked into the bottom of the page is a primer explaining how vecgrep finds things — vector vs BM25 vs hybrid, what the % can and can't tell you.*

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

Alpha (`v0.7.x`). Hybrid retrieval (BM25 + vector + RRF, BM25-weighted by default), cross-encoder rerank, optional per-corpus recency decay, MCP server (stdio + streamable HTTP), named corpora, incremental indexing + file watcher, embedding cache, model/backend migration, optional bearer-token auth, web UI with confidence-tier coloring + V/K/VK match badges, CLI with `vecgrep status` diagnostic, and a hermetic pytest suite. Corpora embedded with *different* models can be served side by side — each queries with its own model. Adapters cover plaintext, markdown, PDF, URLs, Discord JSONL, Claude export, and ChatGPT export. The public API (HTTP + CLI flags) is unstable until v1.0 — expect breaking changes within v0.x.

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
ollama pull bge-m3
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

# Watch mode — re-run the same query at a fixed interval, print a diff
# of source IDs. Pair with `vecgrep watch` while ingesting to confirm
# new chunks are showing up where you expect.
vecgrep search "rate hikes" --watch --interval 5

# Cross-encoder reranking on the candidate pool — slower, more accurate
vecgrep search "what did we decide about rates" --rerank

# Filter results — repeatable, all ANDed
vecgrep search "rate hikes" --filter "source:*2026*.md" --filter "corpus:papers"

# Show the score decomposition for each hit (cosine, BM25, RRF, rerank)
vecgrep search "rate hikes" --explain

# Inspect / clear the embedding cache
vecgrep cache stats
vecgrep cache clear --identity ollama:bge-m3

# Re-embed a corpus to a different backend / model
vecgrep corpora migrate papers --to-backend openai --to-model text-embedding-3-small

# Recency decay: down-rank stale chunks. Half-life in days; a hit one half-life
# old ranks as if half as relevant. Good for chat logs / journals; leave off for
# durable reference. No re-index needed — applied at search time from each
# chunk's parsed date.
vecgrep corpora decay chatlogs --half-life 14
vecgrep corpora decay chatlogs --off

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

# One-shot diagnostic: daemon, auth, per-corpus chunk counts, last-update age
vecgrep status
vecgrep status --json    # for scripting / monitoring

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
docs ──▶ adapters ──▶ chunkers ──┤                      ├──▶ RRF ──▶ [decay] ──▶ dedup ──▶ [rerank] ──▶ top-k
                          └──▶ bm25 (inverted) ──┘
```

- **Adapters** convert source formats to text. They run once per source; chunkers handle slicing.
- **Chunkers** slice text into overlapping windows. `SentenceWindowChunker` is the default — 3 sentences with 1-sentence overlap. `FixedTokenChunker` (tiktoken-backed) is the alternate for code, logs, anything where sentence boundaries are noisy.
- **Embed backends** are pluggable. Ollama (`bge-m3`, 1024-dim) is the default — strong on paraphrase-heavy and multilingual queries. Any other Ollama embedding model works via `VECGREP_EMBED_MODEL` (e.g. `nomic-embed-text`, 768-dim, lighter/faster; `mxbai-embed-large`, 1024-dim). OpenAI (`text-embedding-3-small`, 1536-dim) takes over when Ollama is unreachable and `OPENAI_API_KEY` is set. Each corpus pins the model it was built with; corpora on different models can be served at once, each querying with its own.
- **Qdrant** runs in embedded mode (no server, no Docker) at `~/.vecgrep/qdrant/`. Each named corpus is its own collection.
- **BM25** index runs alongside Qdrant, persisted as a pickle per corpus. Tokenizer splits identifiers (`sharpe_ratio` → `sharpe`, `ratio`) so code search isn't blind to underscore- or camelCase-style naming.
- **Hybrid retrieval** is the default. Each retriever returns its top 50 candidates; their ranks are fused via Reciprocal Rank Fusion (`score = Σ w / (60+rank)`). BM25's weight is `1.5` by default — high enough to float exact-keyword hits over the vector noise floor on short queries, low enough to leave long conceptual queries vector-dominated. Override with `VECGREP_BM25_WEIGHT`. Pure-vector or pure-BM25 are available with `--mode vector` / `--mode bm25`.
- **Recency decay** (optional, per corpus). Set a half-life in days with `vecgrep corpora decay <name> --half-life N` and a hit's fused score is multiplied by `0.5 ** (age_days / half_life)` — a chunk one half-life old ranks as if half as relevant. Applied *before* the top-k cut, so a fresh chunk can rescue itself above a stale one, and lexical closeness alone can't float a stale chunk to the top. Off by default; undated chunks are never penalized. The date comes from a `doc_timestamp` parsed at index time (frontmatter `date:`/`Saved:` lines, a `YYYY-MM-DD` filename, then file mtime). Tune fast for chat/journal corpora, off for durable reference.
- **Dedup.** The sentence-window chunker emits overlapping windows, so one passage can surface as several near-identical hits. vecgrep collapses same-source hits whose character ranges overlap before truncating to top-k, keeping the strongest, so the result list isn't padded with the same text at three ranks.
- **Confidence display.** The displayed `%` is a calibrated relevance estimate, not a raw score. When reranking is on it comes straight from the cross-encoder (the cleanest `P(relevant)` proxy). Otherwise it's a sigmoid over cosine, **calibrated per embedding model** — different models put their noise floor and signal range in different places, so the same `%` means the same thing across corpora. BM25-only hits (which have no cosine) are rescaled rank-relative so a real keyword match doesn't read as the `~1.6%` raw-RRF noise floor. Ranking always uses the underlying fused score. The web UI surfaces V/K/VK badges and tier colors plus a "how search works" panel.
- **Cross-encoder reranker** (`--rerank`, off by default) rescores the candidate pool with `BAAI/bge-reranker-base`. Lazy-loaded — the heavy `torch` import only happens when you ask for it. It's a real quality win on hard, paraphrase-heavy queries where plain hybrid whiffs, but it adds meaningful latency (order ~100ms+ for a small candidate pool) and on easy literal queries it can be a wash or worse — so it's opt-in, not the default. Reach for it when a hybrid search returns near-misses for something you know is in the corpus.

Each corpus pins the embedding backend, model, and dimension at index time and refuses to mix models *within* a single corpus — but the engine resolves each corpus's query embedding from its own pinned model, so multiple corpora on different models coexist fine. To change the model of an existing corpus, `vecgrep corpora migrate` it (or recreate it).

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
| `VECGREP_EMBED_MODEL` | `bge-m3` | Ollama model |
| `VECGREP_OPENAI_EMBED_MODEL` | `text-embedding-3-small` | OpenAI model |
| `OPENAI_API_KEY` | unset | If set, used as fallback when Ollama is down |
| `VECGREP_API_HOST` | `127.0.0.1` | API bind host |
| `VECGREP_API_PORT` | `8765` | API port |
| `VECGREP_API_TOKEN` | unset | If set, `/api/*` requires `Authorization: Bearer <token>` (health stays public) |
| `VECGREP_TOP_K` | `5` | Default `--top` value |
| `VECGREP_BM25_WEIGHT` | `1.5` | Weight on BM25 contribution to RRF fusion. >1 boosts literal-keyword matches over semantic noise on short queries. Set to `1.0` for pure RRF, higher for keyword-leaning ranking. |

## Web UI

`vecgrep serve` boots the FastAPI server and serves a single-page React UI from the same port. Index forms (with a built-in dropdown explainer for source types), corpus list with delete, search bar with top-k slider, mode toggle (hybrid/vector/bm25), reranker checkbox, and results with surrounding context and the matched chunk highlighted. Confidence is shown as a colored tier (high / soft / weak) tied to which retriever placed the hit (V vector, K keyword, VK both) — so a 1.6% BM25 hit reads as the strong literal-keyword match it actually is, not noise.

Sidebar carries a legend mapping V / K / VK and confidence colors, plus a collapsible **"how search works"** panel explaining hybrid retrieval, what reranking does and its latency tradeoff (why it's opt-in), how to read the calibrated `%`, and recency decay — open it once if you're new, ignore it after that. Every action the UI supports has a CLI equivalent.

## MCP server

`vecgrep` ships an MCP server so Claude Desktop, Cursor, or any MCP-aware client can search your corpora as a tool. Install with the extra and run over stdio:

```bash
pip install "vecgrep[mcp]"
vecgrep mcp
```

Tools exposed:
- `search(query, corpus?, top_k?, mode?, rerank?)` — returns ranked chunks with surrounding context as JSON
- `list_corpora()` — every corpus and its stats
- `get_corpus(name)` — one corpus's full metadata including source list, recency-decay setting, and the **filter schema**: which `filters` expressions are accepted (`source:`, `corpus:`, and every discovered `meta.KEY` with sample values) so a caller can pre-filter by actor/channel/date before semantic ranking

Then index a corpus once (`vecgrep index ./my-notes --corpus notes`), and the model can call `search("rate hikes", corpus="notes")` instead of asking you to paste documents.

### Wiring into Claude Code (CLI)

```bash
# basic — uses default OLLAMA_URL (localhost:11434)
claude mcp add vecgrep --scope user -- vecgrep mcp

# pointing at a remote Ollama (e.g. on a homelab box over Tailscale)
claude mcp add vecgrep --scope user \
  --env VECGREP_OLLAMA_URL=http://my-server:11434 \
  -- /absolute/path/to/vecgrep mcp
```

Verify with `claude mcp list` — should show `vecgrep: ... ✓ Connected`.

If `vecgrep` isn't on `PATH` (common when you installed it inside a venv), give the absolute path to the venv's `vecgrep` binary, e.g. `~/repos/vecgrep/venv/bin/vecgrep`.

### Remote MCP over HTTP (one server, many clients)

`vecgrep serve` now exposes `/mcp` for streamable HTTP MCP clients alongside the existing `/api/*` REST routes. Same port, same bearer auth. This means a Claude Code CLI on a different machine can hit your vecgrep box without having to launch its own stdio subprocess (or maintain a separate index).

**Server side.** Bind to a non-loopback interface and set an API token:

```bash
export VECGREP_API_TOKEN=$(openssl rand -hex 32)   # share this with clients
vecgrep serve --host 0.0.0.0
```

The default `api_host` is loopback — `--host 0.0.0.0` opens it to your LAN / VPN. Anything reachable from the network must have an `Authorization: Bearer <token>` header or it gets a 401. There is no off switch for the auth check once `VECGREP_API_TOKEN` is set; that's intentional.

For TLS, run vecgrep behind whatever reverse proxy you already use — Tailscale Serve, Caddy, nginx — and let it terminate HTTPS. vecgrep itself is HTTP-only.

**Client side.** Point Claude Code CLI at the remote endpoint with the bearer token in a header:

```bash
claude mcp add --scope user --transport http vecgrep \
  https://your-server.example/mcp \
  --header "Authorization: Bearer $VECGREP_API_TOKEN"
```

Verify with `claude mcp list` — should show `vecgrep: https://your-server.example/mcp (HTTP) - ✓ Connected`.

The same URL + bearer token works for any MCP client that supports the streamable HTTP transport. Stdio clients (Claude Desktop, Cursor) keep using the local `vecgrep mcp` invocation above.

### Wiring into Claude Desktop / Cursor

Add to `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "vecgrep": {
      "command": "vecgrep",
      "args": ["mcp"],
      "env": {
        "VECGREP_OLLAMA_URL": "http://my-server:11434"
      }
    }
  }
}
```

Restart Claude Desktop after editing. Same shape works for Cursor's `~/.cursor/mcp.json`.

### Wiring into Claude.ai (web)

Claude.ai's web app supports remote MCP servers (HTTP, not stdio). Use the same `vecgrep serve` HTTP endpoint described above and add it under Claude.ai → Settings → Connectors → Custom MCP server. The endpoint must be reachable from Anthropic's servers (so loopback or Tailscale-only addresses won't work — terminate TLS at a public hostname or skip Claude.ai web for this).

## Roadmap

The plan is short and ordered. Make search good first, connect it to where you actually work second, polish later.

**v0.2 — search quality (in progress)**
- ✅ Hybrid search (BM25 + vector + RRF), default on. Pure vector misses exact-token matches like CVE numbers, ticker symbols, function names; BM25 nails them.
- ✅ Cross-encoder reranking (`--rerank`, off by default). Local; adds latency (see "How it works" for the tradeoff), so opt-in.

**v0.3 — connect it**
- ✅ MCP server: expose `vecgrep` as a tool to Claude / Cursor / any MCP client. Index a corpus once, let your assistant retrieve from it instead of stuffing context.
- ✅ Discord JSONL adapter: drop in chat exports, search them as a corpus.
- ✅ Claude / ChatGPT export adapters: search your own conversation history.

**v0.4 — operate it like a tool**
- ✅ Incremental indexing — content-hash skip
- ✅ `vecgrep watch` — file-watcher
- ✅ `vecgrep corpora export/import` — tarball roundtrip
- ✅ Search filters — `--filter source:GLOB`, `corpus:NAME`, `meta.KEY=VALUE`

**v0.5 — power features**
- ✅ `--explain` — per-retriever score breakdown (cosine, BM25, RRF, rerank)
- ✅ Embedding cache — sqlite-backed `(model, sha256(text)) → vector` cache, free re-indexes
- ✅ `vecgrep corpora migrate` — re-embed a corpus to a new backend / model in place
- ✅ Bearer-token auth for `/api/*` — set `VECGREP_API_TOKEN` to lock down a remote-bound server

**v0.6 — quality of life**
- ✅ Test suite — hermetic pytest, service layer + stores + adapters + cache + migration. Caught three real bugs on first run.

**v0.7 — search legibility + correctness**
- ✅ Weighted RRF — BM25 gets `1.5×` over vector by default so genuine keyword hits float above the vector noise floor (`VECGREP_BM25_WEIGHT` overrides). Fixed the "rare token returns 1.6% noise" problem.
- ✅ BM25-only display percentages rescaled per query — 25–100% band (tunable), ranking unchanged.
- ✅ Web UI confidence tiers + match-method badges (V / K / VK) — see what's a literal hit, what's semantic, what's both.
- ✅ In-page primers — index help dropdown, sidebar legend, BM25/vector explainer + "how search works" panel. All `<details>`, default closed.
- ✅ Click-to-expand chunk context — each search result is clickable to lazy-fetch ±2000 chars around the chunk. Backed by `GET /api/chunk/{corpus}/{chunk_id}?window=N`.
- ✅ Per-chunk `doc_timestamp` extraction + tunable per-corpus **recency decay** — stale chunks can't outrank fresh ones on wording alone (`vecgrep corpora decay`).
- ✅ Model-aware confidence calibration — the displayed `%` means the same thing across embedding models; when reranking is on it's the cross-encoder's own score.
- ✅ Post-retrieval dedup of overlapping same-source chunks.
- ✅ Mixed-model serving — corpora on different embed models coexist; each queries with its own. (Was a hard-error before.)
- ✅ Filter schema in `get_corpus` — the accepted `filters` expressions are now discoverable, not a black box.
- `uvx vecgrep` verification + docs
- Plugin API docs (the registries already work, just need an example)
- Per-source TTL on URLs
- Whole-machine `vecgrep backup` / `restore`
- Confidence-gated rerank — auto-rerank only when the top hybrid hit is weak, getting rerank's quality rescue without taxing easy queries

**Later**
Code-aware adapter (tree-sitter), EPUB/DOCX, OCR fallback, RSS feeds, query-aware chunking, query rewriting, single-binary build. See [docs/IDEAS.md](docs/IDEAS.md) for the live list and a "won't do" section explaining what we've explicitly ruled out.

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

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Tests live in `tests/` and run hermetically — no Ollama, no network, no shared `~/.vecgrep`. Each test gets its own `VECGREP_HOME` under `tmp_path` and a deterministic stub embedding backend. The suite covers the service layer, BM25 store, adapters, embedding cache, and migration — the surfaces where every shipped bug has lived. Skip if you don't care; the tooling itself doesn't require them.

## License

MIT
