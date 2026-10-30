"""Rank a symbol's definition above the places that merely mention it.

A query shaped like an identifier (`run`, `Importer.run`, `Outer.Inner.method`)
usually asks for a definition. Lexical and vector scores cannot tell a
definition from a call site that repeats the same words, so `orders =
importer.run(lines)` could outrank `def run` inside `class Importer`. Chunks
from the parser-backed code chunker carry `qualname` and `symbol` metadata;
this multiplies the fused score of a candidate whose metadata names the
queried symbol. It only reorders candidates the retrievers already found, so
it adds no index lookups, and it is a no-op for prose queries and for chunks
without symbol metadata.
"""
from __future__ import annotations

import re

_IDENTIFIER_PATH = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*$")

EXACT_QUALNAME = 4.0     # the query is the chunk's full qualified name
QUALNAME_TAIL = 3.0      # the query is a dotted suffix: `Stats.run` for `Importer.Stats.run`
SIMPLE_NAME = 2.0        # a bare name matching the chunk's own symbol


def symbol_query(query: str) -> str | None:
    """The identifier path a query asks for, or None for an ordinary query."""
    q = query.strip()
    return q if _IDENTIFIER_PATH.match(q) else None


def definition_boost(symbol: str | None, payload: dict) -> float:
    """Score multiplier for one candidate (1.0 means no change)."""
    if not symbol:
        return 1.0
    meta = payload.get("metadata") or {}
    qualname = meta.get("qualname")
    if not qualname:
        return 1.0
    if qualname == symbol:
        return EXACT_QUALNAME
    if "." in symbol and qualname.endswith("." + symbol):
        return QUALNAME_TAIL
    if "." not in symbol and meta.get("symbol") == symbol:
        return SIMPLE_NAME
    return 1.0


def pin_definitions(selected: list, pool: list, symbol: str | None, top_k: int) -> list:
    """Put candidates that define the queried symbol first, keeping the rest in order.

    `pool` is everything the retrievers returned before top_k selection, so a
    definition the reranker or diversity selection left out is not lost. Each
    item needs a `metadata` dict and a `chunk_id`.
    """
    if not symbol:
        return selected
    defs = [r for r in pool if definition_boost(symbol, {"metadata": r.metadata}) > 1.0]
    if not defs:
        return selected
    defs.sort(
        key=lambda r: (
            -definition_boost(symbol, {"metadata": r.metadata}),
            -float(getattr(r, "score", 0.0) or 0.0),
            -float(getattr(r, "similarity_pct", 0.0) or 0.0),
        )
    )
    ids = {r.chunk_id for r in defs}
    return (defs + [r for r in selected if r.chunk_id not in ids])[:top_k]
