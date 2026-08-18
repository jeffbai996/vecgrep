"""Gold query set: schema and loader.

The gold file lives OUTSIDE the repo (it names real people, dates and files).
Default location: `$VECGREP_HOME/eval/gold.json`. Shape:

    {
      "cases": [
        {"id": "outage-1", "corpus": "chats",
         "query": "what caused the build box outage and how was it recovered",
         "want": ["2026-08-07"],            # ANY of these substrings in a top-k source id = hit
         "forbid": [],                       # any of these in top-k = a leak
         "tags": ["incident"]},
        {"id": "neg-1", "corpus": "chats", "negative": true,
         "query": "recipe for sourdough starter hydration ratios"}
      ]
    }

`corpus` is LOGICAL (chats / notes / repos). A run config maps
logical names onto the actual eval corpus built for that variant, so the same
gold answers questions about every variant of the same source data.

Also accepted, so older gold files load unchanged:
  - `q` as an alias of `query`
  - `want` as a list of ints (memory/journal ids: matches `memory-<n>` or
    `journal-<n>` in the source id)
  - `want_substr` as an alias of `want`
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GoldCase:
    id: str
    corpus: str
    query: str
    want: tuple[str, ...] = ()
    forbid: tuple[str, ...] = ()
    negative: bool = False
    tags: tuple[str, ...] = ()
    note: str = ""

    def matches_want(self, source_id: str) -> bool:
        return any(_match(w, source_id) for w in self.want)

    def matches_forbid(self, source_id: str) -> bool:
        return any(_match(w, source_id) for w in self.forbid)


_ID_RE = re.compile(r"^(memory|journal)-(\d+)$")


def _match(pattern: str, source_id: str) -> bool:
    """Substring match, except a bare `memory-N` / `journal-N` pattern must
    match the whole file stem (so `memory-1` never matches `memory-105`)."""
    sid = source_id.replace("\\", "/")
    m = _ID_RE.match(pattern)
    if m:
        stem = sid.rsplit("/", 1)[-1]
        stem = stem[:-3] if stem.endswith(".md") else stem
        return stem == pattern
    return pattern in sid


def _normalize_want(raw, default_kinds=("memory", "journal")) -> tuple[str, ...]:
    out: list[str] = []
    for w in raw or []:
        if isinstance(w, int) or (isinstance(w, str) and w.isdigit()):
            # a bare entry id: either kind counts (older gold files were
            # written for a harness that treats them the same way)
            for kind in default_kinds:
                out.append(f"{kind}-{int(w)}")
        else:
            out.append(str(w))
    return tuple(out)


def load_gold(path: str | Path, default_corpus: str | None = None) -> list[GoldCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_cases = data["cases"] if isinstance(data, dict) else data
    cases: list[GoldCase] = []
    for i, c in enumerate(raw_cases):
        if not isinstance(c, dict):
            continue
        query = c.get("query") or c.get("q")
        if not query:
            continue
        corpus = c.get("corpus") or default_corpus
        if not corpus:
            raise ValueError(f"case {i}: no corpus and no default_corpus")
        want = _normalize_want(c.get("want") if "want" in c else c.get("want_substr"))
        negative = bool(c.get("negative", False))
        if not want and not negative:
            raise ValueError(f"case {c.get('id', i)}: no want and not negative")
        cases.append(GoldCase(
            id=str(c.get("id") or f"{corpus}-{i}"),
            corpus=corpus,
            query=query,
            want=want,
            forbid=tuple(str(x) for x in c.get("forbid", []) or []),
            negative=negative,
            tags=tuple(str(x) for x in c.get("tags", []) or []),
            note=str(c.get("note", "") or ""),
        ))
    return cases
