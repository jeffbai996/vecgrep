"""Retired docs: stale entries that leave search without a human confirm.

An agent that finds an entry out of date retires it instead of deleting it.
Retiring sets `status: retired` with a reason. Search returns only active docs
(service.search_with_diagnostics), so the entry leaves recall at once while the
file stays on disk, and `unretire` puts it back.

After the retention period (VECGREP_RETIRE_RETENTION_DAYS, default 30)
`sweep_retired` removes the doc from the index and moves its file to a `.bak`,
the same recovery copy an unreviewed edit keeps. Hard deletes stay behind
propose_delete. A bad or injected retire costs an unretire, not a restore.

Retire acts only on the direct-write corpus (VECGREP_DIRECT_WRITE_CORPUS),
the one corpus agents may already write and edit without a confirm.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .proposal import ProposalError

RETIRED = "retired"
ACTIVE = "active"
RETIRE_FIELDS = ("retired_at", "retired_reason", "superseded_by")
REASON_MAX_CHARS = 300
REF_MAX_CHARS = 200


def retire_corpus() -> str | None:
    """The corpus retire acts on, or None (off). Read at call time.

    Same variable the MCP server's direct write reads; read here too so the
    CLI sweep does not import the MCP server."""
    return os.environ.get("VECGREP_DIRECT_WRITE_CORPUS", "").strip() or None


def retention() -> timedelta:
    return timedelta(days=float(os.environ.get("VECGREP_RETIRE_RETENTION_DAYS", "30")))


def _now(now: datetime | None) -> datetime:
    return now if now is not None else datetime.now(timezone.utc)


def _one_line(value: str | None) -> str:
    return " ".join((value or "").split())


def retire_meta(meta: dict, reason: str | None, superseded_by: str | None = None,
                now: datetime | None = None) -> dict:
    """`meta` marked retired. Raises ProposalError on a missing or long reason."""
    why = _one_line(reason)
    if not why:
        raise ProposalError("retire needs a reason: say what made the entry stale.")
    if len(why) > REASON_MAX_CHARS:
        raise ProposalError(f"reason is over {REASON_MAX_CHARS} characters.")
    ref = _one_line(superseded_by)
    if len(ref) > REF_MAX_CHARS:
        raise ProposalError(f"superseded_by is over {REF_MAX_CHARS} characters.")
    out = {k: v for k, v in meta.items() if k not in RETIRE_FIELDS}
    out.update(status=RETIRED,
               retired_at=_now(now).astimezone(timezone.utc).isoformat(timespec="seconds"),
               retired_reason=why)
    if ref:
        out["superseded_by"] = ref
    return out


def unretire_meta(meta: dict) -> dict:
    """`meta` active again, with every retirement field removed."""
    out = {k: v for k, v in meta.items() if k not in RETIRE_FIELDS}
    out["status"] = ACTIVE
    return out


def retired_at_of(meta: dict) -> datetime | None:
    raw = meta.get("retired_at") if meta else None
    if not raw:
        return None
    try:
        at = datetime.fromisoformat(str(raw).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return at if at.tzinfo else at.replace(tzinfo=timezone.utc)


def is_purgeable(meta: dict, now: datetime | None = None) -> bool:
    """True once a retired doc has been retired longer than the retention."""
    if (meta or {}).get("status") != RETIRED:
        return False
    at = retired_at_of(meta)
    return at is not None and at + retention() <= _now(now)


def sweep_retired(svc, write_dir: Path, now: datetime | None = None) -> list[str]:
    """Purge docs retired past the retention. Returns the purged doc ids.

    Takes no corpus argument: it acts only on the direct-write corpus, whose
    docs live in `write_dir`. A doc without a readable retired_at is left
    alone. Each purged file is renamed to `<id>.md.bak-<ns>`, so the text stays
    recoverable from disk after it leaves the index."""
    from ..ingestion.adapters.base import AdapterError
    from ..ingestion.adapters.markdown import parse_frontmatter

    corpus = retire_corpus()
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
            if not is_purgeable(meta, now):
                continue
            svc.delete_file(corpus, str(path))
            path.rename(path.with_suffix(f".md.bak-{time.time_ns()}"))
            removed.append(path.stem)
    return removed
