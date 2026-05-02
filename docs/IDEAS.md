# IDEAS

Internal stash. README has the public roadmap; this is what we'd consider building if we picked up vecgrep again. Items get pruned when they're shipped or when we decide we won't ever build them.

## Live (real candidates)

Ordered roughly by what we'd actually pick first.

### Quality of life

- **`uvx vecgrep`** — already wired via `pyproject.toml`. Just need to verify it works end-to-end and document. ~10 minutes.
- **Plugin API docs** — adapter / chunker registries already work. A short page on `register_adapter` + an example PyPI package would let third parties extend without touching this repo. Mostly writing.
- **Per-source TTL on URLs** — `vecgrep index URL --corpus X --ttl 30d`. Track fetched-at in payload metadata; on next index, evict expired sources. Pairs with `vecgrep watch` for news/RSS-ish workflows. Small.
- **Backup / restore** — `vecgrep backup ~/snapshots/2026-05-02.tgz` snapshots `~/.vecgrep` whole. Mostly tar plumbing — we already have per-corpus export, this is the all-corpora variant.

### Search depth

- **Query-aware chunking** — store small atomic chunks at index time; at query time, expand the best one outward by similarity until a token budget is hit. Better recall on long docs without bloating chunk count. Real lift, real complexity.
- **Query rewriting / synonym expansion** — opt-in flag (`--expand`) that runs the query through the embedding model's nearest neighbors first, then unions results. Useful for short queries, harmful for long specific ones — that's why opt-in.

### Ingestion breadth

Each one is small in isolation but every adapter is forever maintenance. Build only when you have content sitting in front of you that needs it.

- **EPUB / DOCX adapters** — straightforward.
- **Code-aware adapter (tree-sitter)** — chunk on function/class boundaries, attach symbol metadata. Big lift, big payoff for indexing your own repos.
- **OCR fallback** — when `PDFAdapter` extracts nothing, fall back to Tesseract. Optional `vecgrep[ocr]` extra.
- **RSS / Atom feed adapter** — point it at a feed, tracks new items. Pairs naturally with TTL + `watch`.

### Distribution

- **Single-binary build** (`pyinstaller`) — friction kills adoption. Only worth doing if vecgrep is going to be promoted to a wider audience.
- **`brew install vecgrep`** — only after single-binary exists.

## Won't do (decided no)

Documenting what we've explicitly decided against so we don't re-litigate.

- **Faceted result clustering** — k-means on result vectors to group hits by topic. Looks impressive in screenshots, doesn't actually help. Hits already group naturally by source. Skip unless someone files a real use case.
- **Notion / Roam / Apple Notes adapters** — each one is its own can of worms (auth, format drift, rate limits). The right path is the plugin API: let people who actually use those tools build adapters and pip install them.
- **Homepage / vecgrep.dev** — pointless until traction. README is enough.
- **True multi-user namespacing** — corpora-per-token-holder. Conflicts with the "local-first tool, not framework" stance. Bearer-token auth (shipped) is enough for the "share with my squad over Tailscale" use case.

## Decisions worth remembering

- **Tests live at the service layer.** That's where every shipped bug has been. Don't bother testing FastAPI plumbing or Click.
- **Each corpus pins (backend, model, dim).** Mixing models within a corpus is rejected, not silently coerced. Migration belongs in a tool, not a runtime fallback.
- **Adapters and Chunkers are the only intended extension surfaces.** Embed backends, the store, the API are concrete by design. Don't add abstract base classes for things with one impl.
- **Local-first is the differentiator, not just a deployment mode.** Anything that requires a cloud roundtrip in the default path is wrong.
- **No telemetry. Ever.**
- **BM25 + RRF degenerate cases.** When all docs contain the term or the corpus has one doc, IDF goes to zero and `rank-bm25` returns 0.0 scores. We fall back to token-overlap counting in that case. Don't remove the fallback — caught only by the test suite, easy to silently regress.
