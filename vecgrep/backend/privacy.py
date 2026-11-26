"""Opt-in-only documents stay out of every read that does not name them.

A doc tagged `opt-in-only` or `quarantine` is private enough that a routine
query, an ambient-recall hook or a corpus summary must never surface it. It is
still indexed and still readable on purpose: a positive filter that names the
doc opts in. Naming means one of

  - `meta.title=<exact title>` or `meta.id=<exact id>`
  - `source:` / `source_path:` with a glob that contains the doc's file stem
    and matches it (`source:*external-abc123*`)

A broad glob (`source:*`), a corpus scope or a negated filter is not naming:
otherwise any caller that scopes a search loosely would read the doc. Reads
by id (get_source, get_chunk) already name the doc and are not gated.
"""
from __future__ import annotations

import fnmatch
from collections.abc import Iterable
from pathlib import PurePath

OPT_IN_TAGS = frozenset({"opt-in-only", "quarantine"})


def _tags(meta: dict | None) -> list[str]:
    raw = (meta or {}).get("tags")
    if isinstance(raw, str):
        # A tag list written as a scalar ("a, b") or a flow list ("[a, b]").
        return [t.strip() for t in raw.strip("[]").split(",")]
    if isinstance(raw, (list, tuple, set, frozenset)):
        return [str(t).strip() for t in raw]
    return []


def is_opt_in_only(meta: dict | None) -> bool:
    return any(t.lower() in OPT_IN_TAGS for t in _tags(meta))


def names_doc(filters: Iterable[str], source_id: str, meta: dict | None) -> bool:
    """True when a positive filter singles out this doc."""
    stem = PurePath(source_id).stem
    for f in filters or ():
        if f.startswith("-"):
            continue
        if f.startswith(("source:", "source_path:")):
            pat = f.split(":", 1)[1]
            if stem and stem in pat and fnmatch.fnmatch(source_id, pat):
                return True
        elif f.startswith(("meta.title=", "meta.id=")):
            key, value = f[len("meta."):].split("=", 1)
            if value and str((meta or {}).get(key, "")) == value:
                return True
    return False


def opt_in_hidden(meta: dict | None, source_id: str,
                  filters: Iterable[str] = ()) -> bool:
    """True when this doc is opt-in-only and the caller did not name it."""
    return is_opt_in_only(meta) and not names_doc(filters, source_id, meta)
