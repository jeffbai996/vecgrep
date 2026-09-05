"""Source labels: stamp search hits with a caller-defined origin label.

Chat archives freeze whatever display name a speaker had when each message
posted, and display names drift — accounts get renamed, identities get
recycled, and the stale name then misleads every later reader of the
archive. The stable identity anchor is the source PATH (a channel
directory, a per-host archive root), so this module lets a deployment map
path globs to labels and stamps the label onto every matching search hit.

The map is DEPLOYMENT-SPECIFIC DATA and lives OUTSIDE the repo:
`$VECGREP_HOME/source_labels.json` by default, `VECGREP_SOURCE_LABELS_FILE`
overrides. vecgrep ships only a generic example
(docs/source_labels.example.json). No map = exact no-op.

Format — flat JSON object, fnmatch glob over source_id → label. Insertion
order is significant: the FIRST matching glob wins, so put specific
patterns above catch-alls:

    {"*/transcripts/team-a/*": "alice@host-1", "*/transcripts/*": "shared"}
"""
from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path

_CACHE: dict[str, tuple[float, dict]] = {}


def source_labels_path() -> Path:
    """Resolve the map path: env override, else $VECGREP_HOME/source_labels.json."""
    override = os.environ.get("VECGREP_SOURCE_LABELS_FILE", "").strip()
    if override:
        return Path(override)
    from .config import get_settings

    return get_settings().home / "source_labels.json"


def load_source_labels(path: Path | str) -> dict[str, str]:
    """Load + validate the map. Missing/corrupt file → {} (never raises).
    Entries whose key or value is not a string are dropped."""
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        glob: label
        for glob, label in raw.items()
        if isinstance(glob, str) and isinstance(label, str) and label.strip()
    }


def load_source_labels_cached(path: Path | str) -> dict[str, str]:
    """mtime-cached load — the per-search hot path shouldn't re-read disk."""
    p = str(path)
    try:
        mtime = os.path.getmtime(p)
    except OSError:
        return {}
    hit = _CACHE.get(p)
    if hit and hit[0] == mtime:
        return hit[1]
    m = load_source_labels(p)
    _CACHE[p] = (mtime, m)
    return m


def label_for(source_id: str, labels: dict[str, str]) -> str | None:
    """First glob (in map order) matching source_id decides the label."""
    for glob, label in labels.items():
        if fnmatch.fnmatch(source_id, glob):
            return label
    return None


def apply_labels(results: list, labels: dict[str, str]) -> None:
    """Stamp source_label onto each result whose source_id matches a glob.
    Results are mutated in place; non-matching hits keep source_label=None."""
    if not labels:
        return
    for r in results:
        r.source_label = label_for(r.source_id, labels)
