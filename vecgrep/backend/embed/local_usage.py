"""Private reported embedding-token receipts in the local collector format."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import sqlite3
import time
import uuid

log = logging.getLogger(__name__)


def record(data: dict, *, model: str) -> None:
    tin, tout = data.get("prompt_eval_count"), 0
    if any(type(v) is not int or not 0 <= v <= 2**63 - 1 for v in (tin, tout)):
        return
    actual_model = data.get("model") or model
    if not isinstance(actual_model, str) or not actual_model:
        return
    target = Path(os.environ.get("LOCAL_LLM_USAGE_DB") or
                  "~/.local/state/local-inference/usage.sqlite3").expanduser()
    try:
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(target, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        os.chmod(target, 0o600)
        with sqlite3.connect(target, timeout=1) as db:
            db.execute("CREATE TABLE IF NOT EXISTS usage ("
                       "request_id TEXT PRIMARY KEY, at REAL NOT NULL, host TEXT NOT NULL,"
                       " model TEXT NOT NULL, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL)")
            # Each HTTP attempt owns one receipt; no prompt, answer or endpoint
            # is persisted, and streamed deltas never become estimated tokens.
            db.execute("INSERT INTO usage VALUES (?,?,?,?,?,?)",
                       ("vecgrep:" + uuid.uuid4().hex, time.time(), "vecgrep",
                        actual_model, tin, tout))
    except (OSError, sqlite3.Error):
        log.warning("Local token usage could not be recorded")
