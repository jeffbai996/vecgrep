"""Best-effort document-date extraction.

Adapters call `extract_timestamp()` to find a document's own date so search can
optionally apply recency decay. Strategy, in order of trust:

1. YAML-ish frontmatter `date:` / `timestamp:` line near the top
2. A labeled line in the body (`Saved:`, `Created:`, `Date:`) with an ISO value
3. A `YYYY-MM-DD` date in the filename
4. Filesystem mtime (last resort)

Returns Unix epoch seconds, or None if nothing parseable is found and no path
is available. The parser is intentionally lenient: a missing date is normal for
many sources and must never raise.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

# Labeled date lines we recognize in markdown bodies / frontmatter. Case-insensitive.
_LABELED_RE = re.compile(
    r"^\s*(?:date|saved|created|created_at|timestamp|ts)\s*[:=]\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# Bare YYYY-MM-DD anywhere in a filename stem.
_FILENAME_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _parse_iso(value: str) -> float | None:
    """Parse an ISO-8601-ish date/datetime to epoch seconds, or None."""
    value = value.strip().strip("\"'")
    if not value:
        return None
    # datetime.fromisoformat handles 'YYYY-MM-DD' and full datetimes with
    # offset. Python <3.11 chokes on a trailing 'Z'; normalize it.
    candidate = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        # Fall back to a plain date prefix if the value has trailing junk.
        m = _FILENAME_DATE_RE.search(value)
        if not m:
            return None
        try:
            dt = datetime.fromisoformat(m.group(1))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def extract_timestamp(text: str, path: str | None = None) -> float | None:
    """Best-effort document timestamp (epoch seconds), or None.

    `text` is the document body; `path` (optional) enables filename-date and
    mtime fallbacks. Never raises.
    """
    # 1 + 2: labeled date line, searching only the head so a date mentioned
    # deep in prose doesn't win over the document's own metadata. The squad
    # store writes `Saved: <iso>` near the top; transcripts use frontmatter
    # `date: <iso>`.
    head = "\n".join(text.splitlines()[:15])
    for m in _LABELED_RE.finditer(head):
        ts = _parse_iso(m.group(1))
        if ts is not None:
            return ts

    if path:
        # 3: date embedded in the filename (e.g. transcript `2026-05-27.md`).
        stem = Path(path).stem
        fm = _FILENAME_DATE_RE.search(stem)
        if fm:
            ts = _parse_iso(fm.group(1))
            if ts is not None:
                return ts
        # 4: filesystem mtime, last resort.
        try:
            return Path(path).stat().st_mtime
        except OSError:
            return None

    return None
