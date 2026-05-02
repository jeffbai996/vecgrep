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
- **Per-corpus filters** — `--filter source:papers/2026-*.md`, `--filter metadata.author=jeff`. Implemented at the Qdrant layer; just plumbing.
- **Faceted result clustering** — group hits by source, by time, by topic (k-means on the result vectors).
- **`--explain` flag** — dump similarity decomposition (which dimensions dominate, what BM25 contributed, what reranker thought).

### Ingestion

- **EPUB / DOCX adapters** — straightforward, just more `pip install`.
- **Code-aware adapter** — tree-sitter; chunk on function/class boundaries, attach symbol metadata. Big lift, big payoff for code search.
- **OCR fallback** — Tesseract optional dep. When PDFAdapter gets no extractable text, try OCR.
- **Notion / Roam / Obsidian / Apple Notes adapters** — anywhere knowledge actually lives. Each is its own can of worms; do them on demand.
- **RSS / Atom feed adapter** — point it at a feed, it tracks new items.

### Operate it like a tool

- **`vecgrep watch ./docs --corpus live`** — file watcher, re-indexes on change. Pairs with a `vecgrep daemon` mode.
- **Incremental indexing** — hash sources, re-embed only diffs. Foundation for `watch`.
- **`vecgrep export` / `vecgrep import`** — ship a corpus between machines as a tarball.
- **Per-source TTL** — `--ttl 30d` on URL ingestion, auto-evict on next index.
- **Backup/restore** — single command, snapshot ~/.vecgrep.

### Power features

- **Multi-tenant mode** — namespaced corpora behind a bearer token; for shared servers.
- **Embedding model migration tool** — re-embed a corpus to a new model in place, with rollback.
- **Documented plugin API** — pip-installable third-party adapters and chunkers.
- **Hybrid embedding cache** — cache (text_hash → vector) so re-indexing the same content is free.

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
