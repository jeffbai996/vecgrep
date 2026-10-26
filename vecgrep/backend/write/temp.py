"""Temp corpus: short-lived docs that expire on their own.

Some facts are true only for a while (the current release, a workaround, a
degraded host). Saving them permanently leaves stale text in search; never
saving them loses them. A temp doc carries `expires_at` in its frontmatter:

- search drops it once that time has passed (service.search_with_diagnostics),
  so expiry is exact and never waits on a cleanup job;
- `sweep_expired` deletes the expired files and their index entries, to
  reclaim space and keep browse/get_chunk clean.

The temp corpus is operator-designated (VECGREP_TEMP_CORPUS) and default-off,
like direct write. Every temp doc must carry an expiry, capped at
VECGREP_TEMP_MAX_TTL_DAYS. The sweep is the one deletion path that needs no
human confirm, so it is bounded twice: it only ever reads the temp corpus's own
write dir, and only removes a doc whose own `expires_at` has passed -- the
deletion was agreed when the doc was written.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .proposal import ProposalError

_RELATIVE_RE = re.compile(r"^(\d+)([mhdw])$")
_UNIT = {"m": timedelta(minutes=1), "h": timedelta(hours=1),
         "d": timedelta(days=1), "w": timedelta(weeks=1)}


def temp_corpus() -> str | None:
    """The temp corpus name, or None (feature off). Read at call time."""
    return os.environ.get("VECGREP_TEMP_CORPUS", "").strip() or None


def max_ttl() -> timedelta:
    return timedelta(days=float(os.environ.get("VECGREP_TEMP_MAX_TTL_DAYS", "90")))


def _now(now: datetime | None) -> datetime:
    return now if now is not None else datetime.now(timezone.utc)


def parse_expiry(expires: str | None, now: datetime | None = None) -> datetime:
    """`3d` / `12h` / `2w` / `90m`, or an ISO date or datetime -> aware UTC.

    Raises ProposalError when the value is missing, unparseable, not in the
    future, or past the max TTL. A bare date means the end of that UTC day, so
    `2026-11-02` keeps the doc through the 2nd."""
    now = _now(now)
    value = (expires or "").strip()
    if not value:
        raise ProposalError(
            "a temp doc needs an expiry: pass expires as 3d, 12h, 2w or an ISO date.")
    rel = _RELATIVE_RE.match(value.lower())
    if rel:
        at = now + int(rel.group(1)) * _UNIT[rel.group(2)]
    else:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ProposalError(
                f"expires {value!r} is not 3d, 12h, 2w or an ISO date.") from None
        if len(value) == 10:  # bare YYYY-MM-DD
            parsed = parsed.replace(hour=23, minute=59, second=59)
        at = parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    at = at.astimezone(timezone.utc).replace(microsecond=0)
    if at <= now:
        raise ProposalError(f"expires {value!r} is not in the future.")
    if at - now > max_ttl():
        raise ProposalError(
            f"expires {value!r} is past the {max_ttl().days}-day limit for temp "
            f"docs. Save durable facts in a permanent corpus instead.")
    return at


def expires_at_of(meta: dict) -> datetime | None:
    """The doc's expiry from its metadata, or None when absent or unparseable."""
    raw = meta.get("expires_at") if meta else None
    if not raw:
        return None
    try:
        at = datetime.fromisoformat(str(raw).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return at if at.tzinfo else at.replace(tzinfo=timezone.utc)


def is_expired(meta: dict, now: datetime | None = None) -> bool:
    """True once a doc's own expiry has passed. No expiry -> never expires."""
    at = expires_at_of(meta)
    return at is not None and at <= _now(now)


def time_left(meta: dict, now: datetime | None = None) -> str | None:
    """Coarse remaining time for display (`3d`, `5h`, `40m`), or None."""
    at = expires_at_of(meta)
    if at is None:
        return None
    secs = (at - _now(now)).total_seconds()
    if secs <= 0:
        return "expired"
    if secs >= 86400:
        return f"{int(secs // 86400)}d"
    if secs >= 3600:
        return f"{int(secs // 3600)}h"
    return f"{max(1, int(secs // 60))}m"


def sweep_expired(svc, write_dir: Path, now: datetime | None = None) -> list[str]:
    """Delete expired docs of the temp corpus. Returns the removed doc ids.

    Takes no corpus argument: it can only act on the operator-designated temp
    corpus, whose docs live in `write_dir`. A doc without a readable expiry is
    left alone -- the sweep never guesses."""
    from ..ingestion.adapters.base import AdapterError
    from ..ingestion.adapters.markdown import parse_frontmatter

    corpus = temp_corpus()
    if not corpus or not write_dir.is_dir():
        return []
    now = _now(now)
    removed: list[str] = []
    with svc.locks.write(corpus):
        for path in sorted(write_dir.glob("*.md")):
            try:
                meta = parse_frontmatter(path.read_text(encoding="utf-8"))
            except (OSError, AdapterError):
                continue
            if not is_expired(meta, now):
                continue
            svc.delete_file(corpus, str(path))
            path.unlink(missing_ok=True)
            removed.append(path.stem)
    return removed
