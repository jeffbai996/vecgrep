"""WAL bounds are enforced, not just requested.

Past incident (2026-09-08): repos.db-wal reached 239 MB -- larger than the
database it journals. Passive checkpoints during bulk indexing were deferred
or never followed by a log reset, and journal_size_limit only truncates when
a checkpoint actually resets the WAL. The oversized file then sat pinned in
disk and page cache until a manual TRUNCATE checkpoint. These tests pin the
escalation path that prevents a recurrence.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from vecgrep.backend.store.bm25_sqlite import BM25SqliteStore


def _wal_path(root: Path, corpus: str) -> Path:
    return root / f"{corpus}.db-wal"


def _fill(lite: BM25SqliteStore, corpus: str, rows: int = 400) -> None:
    ids = [f"id{i}" for i in range(rows)]
    texts = [f"alpha beta gamma delta epsilon token{i} " * 8 for i in range(rows)]
    payloads = [{"source_id": f"/src/{i % 7}", "text": texts[i]} for i in range(rows)]
    with lite.bulk(corpus):
        lite.upsert(corpus, ids, texts, payloads)


def test_bulk_exceeding_limit_truncates_wal(tmp_path, monkeypatch):
    from vecgrep.backend.store import bm25_sqlite as module

    limit = 8192
    monkeypatch.setattr(module, "WAL_JOURNAL_SIZE_LIMIT_BYTES", limit)
    root = tmp_path / "db"
    lite = BM25SqliteStore(root)

    _fill(lite, "c")

    wal = _wal_path(root, "c")
    # With no reader pinning frames, the post-bulk checkpoint must bring the
    # WAL back under its configured limit rather than stranding it at bulk
    # size until some future writer happens to reset the log.
    assert wal.stat().st_size <= limit


def test_reader_defers_truncation_then_next_mutation_reclaims(tmp_path, monkeypatch):
    from vecgrep.backend.store import bm25_sqlite as module

    limit = 8192
    monkeypatch.setattr(module, "WAL_JOURNAL_SIZE_LIMIT_BYTES", limit)
    root = tmp_path / "db"
    lite = BM25SqliteStore(root)
    _fill(lite, "c", rows=50)  # create the database file first

    # A reader holding an open snapshot pins WAL frames; escalation must give
    # up quickly instead of blocking the mutation path behind it.
    reader = sqlite3.connect(str(root / "c.db"))
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM chunks").fetchone()
    try:
        _fill(lite, "c")
        # Deferred, not raised: the bulk completed even though the WAL could
        # not be truncated past the reader.
        assert _wal_path(root, "c").stat().st_size > limit
    finally:
        reader.rollback()
        reader.close()

    # Reader gone: the next mutation's checkpoint reclaims the file.
    with lite.bulk("c"):
        lite.upsert("c", ["late"], ["omega token"], [{"source_id": "/late"}])
    assert _wal_path(root, "c").stat().st_size <= limit


def test_small_writes_do_not_escalate(tmp_path, caplog):
    root = tmp_path / "db"
    lite = BM25SqliteStore(root)

    with caplog.at_level(logging.INFO, logger="vecgrep.backend.store.bm25_sqlite"):
        with lite.bulk("c"):
            lite.upsert("c", ["a"], ["alpha token"], [{"source_id": "/a"}])

    # A WAL already inside its limit needs no TRUNCATE attempt.
    assert "truncate" not in caplog.text.lower()
