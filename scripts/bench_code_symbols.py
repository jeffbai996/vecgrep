"""Benchmark the code_symbol chunker: heuristic boundaries vs parser-backed symbols.

Deterministic: fixed fixtures under tests/fixtures/code_symbols, hand-written
ground truth in expected.json (qualified name, first and last line, with
decorators and directly attached comments), and vecgrep's own in-memory BM25
(tokenizer, camelCase split, coverage penalty) for retrieval, so no embedding
model, reranker or on-disk index is involved.

    python scripts/bench_code_symbols.py [--json]

Metrics, per design:
  coverage         symbols that own a chunk or are listed in a chunk's `nested`
  owned            symbols with a chunk of their own
  span exact       owned chunks whose reported span equals the ground truth
  qualified@1      query "Outer.inner": the top BM25 chunk contains the symbol
  name@1           query "inner" for names unique in the corpus, same rule
  noise            for qualified@1 hits, chunk lines / symbol lines (1.0 is exact)
  parse failures   files that fell back to the heuristic
  content kept     every non-whitespace character of every file is in some chunk
  ms/file          chunking time, mean over repeated runs
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from vecgrep.backend.ingestion.chunkers.code_symbol import CodeSymbolChunker
from vecgrep.backend.store.bm25_store import BM25Store
from vecgrep.backend.symbol_query import definition_boost, pin_definitions, symbol_query

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "code_symbols"
def line_range(text: str, start: int, end: int) -> tuple[int, int]:
    first = text.count("\n", 0, start) + 1
    last = text.count("\n", 0, max(start, end - 1)) + 1
    return first, last


def run(parse: bool, symbols_first: bool = False, repeats: int = 50) -> dict:
    expected = json.loads((FIXTURES / "expected.json").read_text())
    chunker = CodeSymbolChunker(parse=parse)
    files = {name: (FIXTURES / name).read_text() for name in expected}
    t0 = time.perf_counter()
    for _ in range(repeats):
        for name, text in files.items():
            chunker.chunk_doc(SimpleNamespace(text=text, source_id=f"/fixtures/{name}"))
    ms_per_file = (time.perf_counter() - t0) * 1000 / (repeats * len(files))

    corpus = []        # (file, chunk, first_line, last_line)
    parse_failures = 0
    content_kept = True
    for name, text in files.items():
        chunks = chunker.chunk_doc(SimpleNamespace(text=text, source_id=f"/fixtures/{name}"))
        if any((c.meta or {}).get("parse_error") for c in chunks):
            parse_failures += 1
        covered = set()
        for c in chunks:
            assert text[c.start:c.end] == c.text, f"offsets do not address the chunk text in {name}"
            covered.update(range(c.start, c.end))
            corpus.append((name, c, *line_range(text, c.start, c.end)))
        content_kept &= all(i in covered for i, ch in enumerate(text) if not ch.isspace())

    bm25 = BM25Store(None)
    bm25.upsert("bench", [str(i) for i in range(len(corpus))],
                [c.search_text or c.text for _, c, _, _ in corpus], [{} for _ in corpus])

    def top(query: str):
        hits = bm25.search("bench", query, top_k=50)
        if not symbols_first:
            return corpus[int(hits[0][0])]
        # The same two steps search() applies: the fusion multiplier, then
        # pinning definitions of the queried symbol first.
        symbol = symbol_query(query)
        ranked = sorted(
            ((1.0 / (60 + rank + 1)) * definition_boost(symbol, {"metadata": corpus[int(cid)][1].meta or {}}), int(cid))
            for rank, (cid, _, _) in enumerate(hits))
        pool = [SimpleNamespace(chunk_id=i, metadata=corpus[i][1].meta or {}) for _, i in reversed(ranked)]
        return corpus[pin_definitions(pool, pool, symbol, 1)[0].chunk_id]

    names = [q.split(".")[-1] for symbols in expected.values() for q, _, _ in symbols]
    rows = []
    for name, symbols in expected.items():
        for qual, first, last in symbols:
            mine = [(c, a, b) for f, c, a, b in corpus if f == name]
            owned = [(c, a, b) for c, a, b in mine if (c.meta or {}).get("qualname", (c.meta or {}).get("symbol")) == qual]
            nested = any(qual in [n.strip() for n in (c.meta or {}).get("nested", "").split(",")] for c, _, _ in mine)
            if owned:
                c, a, b = owned[0]
                reported = (c.meta or {}).get("lines")
                span_exact = reported == f"{first}-{last}" if reported else (a, b) == (first, last)
            else:
                span_exact = False

            def hit(query: str) -> tuple[bool, float | None]:
                f, c, a, b = top(query)
                ok = f == name and a <= first and b >= min(last, first)
                return ok, ((b - a + 1) / (last - first + 1) if ok else None)

            q_ok, noise = hit(qual)
            simple = qual.split(".")[-1]
            unique = names.count(simple) == 1
            n_ok = hit(simple)[0] if unique else None
            rows.append(dict(file=name, symbol=qual, owned=bool(owned), covered=bool(owned) or nested,
                             span_exact=span_exact, qualified_hit=q_ok, noise=noise,
                             name_hit=n_ok))
    total = len(rows)
    hits = [r for r in rows if r["qualified_hit"]]
    uniques = [r for r in rows if r["name_hit"] is not None]
    return dict(
        symbols=total,
        coverage=sum(r["covered"] for r in rows),
        owned=sum(r["owned"] for r in rows),
        span_exact=sum(r["span_exact"] for r in rows),
        qualified_at_1=sum(r["qualified_hit"] for r in rows),
        name_at_1=f"{sum(r['name_hit'] for r in uniques)}/{len(uniques)}",
        noise=round(sum(r["noise"] for r in hits) / len(hits), 2) if hits else None,
        parse_failures=parse_failures,
        content_kept=content_kept,
        chunks=len(corpus),
        ms_per_file=round(ms_per_file, 3),
        rows=rows,
    )


def main() -> int:
    results = {"heuristic": run(parse=False), "parser": run(parse=True),
               "parser+rank": run(parse=True, symbols_first=True)}
    if "--json" in sys.argv:
        print(json.dumps(results, indent=2))
        return 0
    keys = ["symbols", "coverage", "owned", "span_exact", "qualified_at_1", "name_at_1",
            "noise", "parse_failures", "content_kept", "chunks", "ms_per_file"]
    print(f"{'metric':<16}" + "".join(f"{name:>13}" for name in results))
    for key in keys:
        print(f"{key:<16}" + "".join(f"{str(r[key]):>13}" for r in results.values()))
    print()
    print("per symbol: Q = qualified@1 hit, S = exact span, O = owns a chunk")
    flag = lambda r: ("Q" if r["qualified_hit"] else "-") + ("S" if r["span_exact"] else "-") + ("O" if r["owned"] else "-")
    for rows in zip(*(r["rows"] for r in results.values())):
        print(f"{rows[0]['file']:<10} {rows[0]['symbol']:<32} " + "  ".join(f"{n} {flag(r)}" for n, r in zip(results, rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
