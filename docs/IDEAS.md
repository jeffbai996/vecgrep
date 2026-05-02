# IDEAS

Stash for things we want to remember but aren't building next. README roadmap is the short, public list. This is the long, ordered-by-honest-utility list.

## Up next (v0.2 → v0.3)

These are committed; they live in the README too.

### v0.2 — Search quality

**Hybrid search (BM25 + vector + RRF).** Single highest-leverage upgrade. Pure vector search misses exact-token matches — CVE numbers, ticker symbols, function names, anything where wording is the signal. BM25 nails those, vector nails paraphrase, RRF fuses them: `score = sum(1 / (k + rank_in_each_list))` with `k=60` is the canonical recipe. ~30 lines of code. Often beats either alone in eval.

Implementation sketch:
- Add `rank-bm25` (or implement BM25 in-process — it's <50 lines)
- Index lowercased tokens to a per-corpus inverted index alongside Qdrant
- At search time: vector top-50 + BM25 top-50, fuse, return top-k
- `--mode {hybrid,vector,bm25}` CLI flag; default hybrid

**Cross-encoder reranking.** Take top 50 from hybrid search, rerank with a small encoder that scores `(query, chunk)` as a pair. `BAAI/bge-reranker-base` is the standard pick; runs on CPU at ~30ms for 50 chunks. Off by default behind `--rerank`. Pulled in via optional `vecgrep[rerank]` extra so the base install stays slim.

### v0.3 — Connect it to where work happens

**MCP server.** Expose vecgrep as a tool to Claude / Cursor / any MCP client. Endpoints map 1:1 to existing API (`search`, `list_corpora`). Same `vecgrep serve` process; MCP runs over stdio. The whole point: stop dumping 10K-token contexts into the model when a focused retrieval would do.

**Discord JSONL adapter.** Reads discord-logger output (one JSON message per line), groups by channel as separate sub-sources, preserves author + timestamp in metadata. Lets you grep months of squad chat semantically.

**Claude / ChatGPT export adapter.** Both export as JSON (different schemas). Each conversation becomes a Document; metadata captures conversation title, date, role-tagged turns. Lets you find that thing you and Claude figured out in March.

## Later (unsorted, by category)

Not committed. We'll cherry-pick when something becomes the right next move.

### Search

- **Query-aware chunking** — at index time, store small atomic chunks. At query time, find the best one, then expand outward by similarity to the query until a token budget is hit. Better recall on long docs without bloating chunk count.
- **Query rewriting / synonym expansion** behind a flag — useful for short queries, harmful for long specific ones, so opt-in.
- ~~Per-corpus filters~~ — shipped v0.4. `source:GLOB`, `corpus:NAME`, `meta.KEY=VALUE`. Repeatable, AND'd. Filtering happens after retrieval rather than at the Qdrant layer (simpler; works equally well for BM25 hits).
- **Faceted result clustering** — group hits by source, by time, by topic (k-means on the result vectors).
- ~~`--explain` flag~~ — shipped v0.5. Per-retriever scores: vector_cosine, vector_rank, bm25_score, bm25_rank, rrf, rerank_score.

### Ingestion

- **EPUB / DOCX adapters** — straightforward, just more `pip install`.
- **Code-aware adapter** — tree-sitter; chunk on function/class boundaries, attach symbol metadata. Big lift, big payoff for code search.
- **OCR fallback** — Tesseract optional dep. When PDFAdapter gets no extractable text, try OCR.
- **Notion / Roam / Obsidian / Apple Notes adapters** — anywhere knowledge actually lives. Each is its own can of worms; do them on demand.
- **RSS / Atom feed adapter** — point it at a feed, it tracks new items.

### Operate it like a tool

- ~~`vecgrep watch ./docs --corpus live`~~ — shipped v0.4.
- ~~Incremental indexing~~ — shipped v0.4. Hashes per source, skip unchanged.
- ~~`vecgrep corpora export/import`~~ — shipped v0.4. Tarball with metadata + qdrant collection + bm25 pickle.
- **Per-source TTL** — `--ttl 30d` on URL ingestion, auto-evict on next index.
- **Backup/restore** — single command, snapshot ~/.vecgrep.

### Power features

- ~~Bearer-token auth~~ — shipped v0.5. `VECGREP_API_TOKEN` env. Health stays public. Frontend reads token from `localStorage.vecgrep_token`. Note: this is single-shared-secret, not multi-user — no per-user namespacing of corpora yet.
- ~~Embedding model migration~~ — shipped v0.5. `vecgrep corpora migrate <name> --to-backend X --to-model Y`. Re-fetches all sources, swaps in atomically.
- **Documented plugin API** — pip-installable third-party adapters and chunkers. Mostly a docs job; the registries already work.
- ~~Embedding cache~~ — shipped v0.5. SQLite at `~/.vecgrep/embed_cache.db` keyed on (backend identity, sha256(text)). Transparent wrapper around any backend.
- **True multi-user namespacing** — separate corpora per token holder. Bigger lift than bearer auth alone; needs token registry, owner-tagged corpus paths.

### Distribution / polish

- **Single-binary build** — `pyinstaller` / `briefcase`. Friction kills adoption.
- **`brew install vecgrep`** — once binary exists.
- **`uvx vecgrep`** support — already works in theory via pyproject; verify and document.
- **Homepage** — vecgrep.dev with asciinema demo. Only worth it if v0.2 + v0.3 actually land and the tool is meaningfully differentiated.

## Decisions worth remembering

- **No tests in v0.1, by spec.** Once API stabilizes (v0.2 onward), tests should land at the service layer first — that's where bugs hide.
- **Each corpus pins (backend, model, dim).** Mixing models within a corpus is rejected, not silently coerced. Migration belongs in a tool, not a runtime fallback.
- **Adapters and Chunkers are the only intended extension surfaces.** Embed backends, the store, the API are concrete by design. Don't add abstract base classes for things with one impl.
- **Local-first is the differentiator, not just a deployment mode.** Anything that requires a cloud roundtrip in the default path is wrong.
- **No telemetry. Ever.**
