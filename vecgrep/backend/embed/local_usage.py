"""Private reported embedding-token receipts in the local collector format.

Two tables in one local database:

- `usage`: one row per HTTP attempt that reported its tokens. Other local
  tools write this table with positional inserts, so its six columns are
  fixed.
- `embed_route`: one row per bulk batch attempt: which endpoint role answered
  (`primary` or `overflow`), whether the overflow fell back to the primary,
  and a short reason code. A deployment maps the roles to its own host names.

Neither table stores a prompt, an answer, a URL or a host address.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
import sqlite3
import time
import uuid

log = logging.getLogger(__name__)

# Route rows are an operational trace, not an account; a fortnight covers any
# question about when bulk embedding last switched endpoints.
ROUTE_RETENTION_S = 14 * 86400


def _database() -> Path:
    target = Path(os.environ.get("LOCAL_LLM_USAGE_DB") or
                  "~/.local/state/local-inference/usage.sqlite3").expanduser()
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(target, os.O_CREAT | os.O_RDWR, 0o600)
    os.close(fd)
    os.chmod(target, 0o600)
    return target


def _write_usage(db: sqlite3.Connection, model: str, tin: int, tout: int) -> str:
    db.execute("CREATE TABLE IF NOT EXISTS usage ("
               "request_id TEXT PRIMARY KEY, at REAL NOT NULL, host TEXT NOT NULL,"
               " model TEXT NOT NULL, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL)")
    request_id = "vecgrep:" + uuid.uuid4().hex
    # Each HTTP attempt owns one receipt, and streamed deltas never become
    # estimated tokens.
    db.execute("INSERT INTO usage VALUES (?,?,?,?,?,?)",
               (request_id, time.time(), "vecgrep", model, tin, tout))
    return request_id


def _write_route(db: sqlite3.Connection, route: str, outcome: str, reason: str,
                 request_id: str | None) -> None:
    db.execute("CREATE TABLE IF NOT EXISTS embed_route ("
               "at REAL NOT NULL, route TEXT NOT NULL, outcome TEXT NOT NULL,"
               " reason TEXT NOT NULL DEFAULT '', request_id TEXT)")
    db.execute("CREATE INDEX IF NOT EXISTS embed_route_at ON embed_route(at)")
    now = time.time()
    db.execute("INSERT INTO embed_route VALUES (?,?,?,?,?)",
               (now, route, outcome, reason, request_id))
    db.execute("DELETE FROM embed_route WHERE at < ?", (now - ROUTE_RETENTION_S,))


def _tokens(data: dict, model: str) -> tuple[str, int, int] | None:
    tin, tout = data.get("prompt_eval_count"), 0
    if any(type(v) is not int or not 0 <= v <= 2**63 - 1 for v in (tin, tout)):
        return None
    actual_model = data.get("model") or model
    if not isinstance(actual_model, str) or not actual_model:
        return None
    return actual_model, tin, tout


def record(data: dict, *, model: str, route: str | None = None,
           outcome: str = "ok", reason: str = "") -> None:
    """Record reported tokens, and the route of a bulk batch when `route` is
    given. An answer without a valid token count records only the route."""
    counted = _tokens(data, model) if isinstance(data, dict) else None
    if counted is None and route is None:
        return
    try:
        target = _database()
        with sqlite3.connect(target, timeout=1) as db:
            request_id = _write_usage(db, *counted) if counted else None
            if route is not None:
                _write_route(db, route, outcome, reason, request_id)
    except (OSError, sqlite3.Error):
        log.warning("Local token usage could not be recorded")


def record_route(*, route: str, outcome: str, reason: str = "") -> None:
    """Record a bulk batch attempt that reported no tokens."""
    record({}, model="", route=route, outcome=outcome, reason=reason)
