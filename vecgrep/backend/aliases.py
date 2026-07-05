"""Alias / entity expansion: one entity, many surface forms.

A person shows up in chat logs as a nickname, a bot handle, a username, and
a name in another language. Retrieval that only matches the literal query
form misses evidence written under the others. This module expands the
query with an entity's other forms when ANY form appears in it.

The map is USER-SUPPLIED PERSONAL DATA and lives OUTSIDE the repo:
`$VECGREP_HOME/aliases.json` by default, `VECGREP_ALIASES_FILE` overrides.
vecgrep ships only a generic example (docs/aliases.example.json). No map =
exact no-op. Config-driven, not learned.

Format — flat JSON object, canonical form → list of alternates:

    {"alice": ["ali", "小艾"], "relay-service": ["the relay", "rs-1"]}

Matching is symmetric across all of an entity's forms: a query mentioning
"小艾" gains "alice ali", and vice versa. ASCII-word forms match on word
boundaries ("ali" never fires inside "alignment"); forms with CJK match by
containment (CJK has no word boundaries).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

_CACHE: dict[str, tuple[float, dict]] = {}


def aliases_path() -> Path:
    """Resolve the alias-map path: env override, else $VECGREP_HOME/aliases.json."""
    override = os.environ.get("VECGREP_ALIASES_FILE", "").strip()
    if override:
        return Path(override)
    from .config import get_settings

    return get_settings().home / "aliases.json"


def load_alias_map(path: Path | str) -> dict[str, list[str]]:
    """Load + validate the map. Missing/corrupt file → {} (never raises).
    Non-string forms are dropped; non-list values are dropped entirely."""
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[str]] = {}
    for canonical, forms in raw.items():
        if not isinstance(canonical, str) or not isinstance(forms, list):
            continue
        clean = [f for f in forms if isinstance(f, str) and f.strip()]
        if clean:
            out[canonical] = clean
    return out


def load_alias_map_cached(path: Path | str) -> dict[str, list[str]]:
    """mtime-cached load — the per-search hot path shouldn't re-read disk."""
    p = str(path)
    try:
        mtime = os.path.getmtime(p)
    except OSError:
        return {}
    hit = _CACHE.get(p)
    if hit and hit[0] == mtime:
        return hit[1]
    m = load_alias_map(p)
    _CACHE[p] = (mtime, m)
    return m


def _form_in_query(form: str, query: str) -> bool:
    """ASCII-word forms match on word boundaries; forms containing CJK (or
    other non-word chars) match by containment."""
    if re.fullmatch(r"[A-Za-z0-9_\-]+", form):
        return re.search(rf"(?<![A-Za-z0-9]){re.escape(form)}(?![A-Za-z0-9])",
                         query, re.IGNORECASE) is not None
    return form in query


def expand_query(query: str, alias_map: dict[str, list[str]]) -> tuple[str, list[str]]:
    """Append missing surface forms for every entity mentioned in the query.

    Returns (expanded_query, matched_canonical_names). The original query is
    preserved verbatim at the front; only forms not already present are
    appended, so expansion is idempotent.
    """
    matched: list[str] = []
    additions: list[str] = []
    for canonical, alternates in alias_map.items():
        all_forms = [canonical, *alternates]
        if not any(_form_in_query(f, query) for f in all_forms):
            continue
        matched.append(canonical)
        for f in all_forms:
            if not _form_in_query(f, query) and f not in additions:
                additions.append(f)
    if not additions:
        return query, matched
    return f"{query} {' '.join(additions)}", matched
