"""Compact, disposable source catalog for the corpus explorer.

Search indexes chunks; Browse navigates sources.  Keeping one small SQLite row
per source lets the web UI start without deserializing a multi-hundred-megabyte
BM25 sidecar.  Qdrant/BM25 remain canonical: a missing or stale catalog is
rebuilt from them and can be deleted at any time.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import TypeAlias


CatalogGeneration: TypeAlias = tuple[float, int, int]


class ExplorerStore:
    """Process-safe source metadata store keyed by corpus and source ID."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            ":memory:" if path is None else str(path),
            timeout=30,
            check_same_thread=False,
        )
        if path is not None:
            self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                corpus TEXT NOT NULL,
                source_id TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                doc_timestamp REAL,
                chunk_count INTEGER NOT NULL,
                PRIMARY KEY (corpus, source_id)
            );
            CREATE INDEX IF NOT EXISTS documents_corpus
                ON documents (corpus);
            CREATE TABLE IF NOT EXISTS generations (
                corpus TEXT PRIMARY KEY,
                updated_at REAL NOT NULL,
                doc_count INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL
            );
            """
        )
        self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def generation(self, corpus: str) -> CatalogGeneration | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT g.updated_at, g.doc_count, g.chunk_count, "
                "COUNT(d.source_id) "
                "FROM generations AS g "
                "LEFT JOIN documents AS d ON d.corpus = g.corpus "
                "WHERE g.corpus = ? "
                "GROUP BY g.corpus",
                (corpus,),
            ).fetchone()
        if row is None:
            return None
        if int(row[3]) != int(row[1]):
            return None
        return float(row[0]), int(row[1]), int(row[2])

    def records(self, corpus: str) -> list[dict]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT source_id, metadata_json, doc_timestamp, chunk_count "
                "FROM documents WHERE corpus = ? ORDER BY source_id",
                (corpus,),
            ).fetchall()
        return [
            {
                "source_id": str(source_id),
                "metadata": json.loads(metadata_json),
                "doc_timestamp": (
                    float(doc_timestamp) if doc_timestamp is not None else None
                ),
                "chunk_count": int(chunk_count),
            }
            for source_id, metadata_json, doc_timestamp, chunk_count in rows
        ]

    def replace(
        self,
        corpus: str,
        records: list[dict],
        generation: CatalogGeneration,
    ) -> None:
        """Atomically replace one derived catalog and mark it complete."""
        rows = [self._record_row(corpus, record) for record in records]
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM documents WHERE corpus = ?", (corpus,)
            )
            if rows:
                self._connection.executemany(
                    "INSERT INTO documents "
                    "(corpus, source_id, metadata_json, doc_timestamp, chunk_count) "
                    "VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
            self._write_generation(corpus, generation)

    def upsert(self, corpus: str, record: dict) -> None:
        """Write one source row and invalidate the completeness marker."""
        row = self._record_row(corpus, record)
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM generations WHERE corpus = ?", (corpus,)
            )
            self._connection.execute(
                "INSERT INTO documents "
                "(corpus, source_id, metadata_json, doc_timestamp, chunk_count) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(corpus, source_id) DO UPDATE SET "
                "metadata_json = excluded.metadata_json, "
                "doc_timestamp = excluded.doc_timestamp, "
                "chunk_count = excluded.chunk_count",
                row,
            )

    def delete_source(self, corpus: str, source_id: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM generations WHERE corpus = ?", (corpus,)
            )
            self._connection.execute(
                "DELETE FROM documents WHERE corpus = ? AND source_id = ?",
                (corpus, source_id),
            )

    def set_generation(
        self, corpus: str, generation: CatalogGeneration
    ) -> None:
        with self._lock, self._connection:
            self._write_generation(corpus, generation)

    def drop(self, corpus: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM documents WHERE corpus = ?", (corpus,)
            )
            self._connection.execute(
                "DELETE FROM generations WHERE corpus = ?", (corpus,)
            )

    def _write_generation(
        self, corpus: str, generation: CatalogGeneration
    ) -> None:
        updated_at, doc_count, chunk_count = generation
        self._connection.execute(
            "INSERT INTO generations (corpus, updated_at, doc_count, chunk_count) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(corpus) DO UPDATE SET "
            "updated_at = excluded.updated_at, "
            "doc_count = excluded.doc_count, "
            "chunk_count = excluded.chunk_count",
            (corpus, updated_at, doc_count, chunk_count),
        )

    @staticmethod
    def _record_row(corpus: str, record: dict) -> tuple:
        source_id = str(record.get("source_id") or "")
        if not source_id:
            raise ValueError("explorer record requires source_id")
        metadata = record.get("metadata") or {}
        timestamp = record.get("doc_timestamp")
        return (
            corpus,
            source_id,
            json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str),
            float(timestamp) if timestamp is not None else None,
            max(1, int(record.get("chunk_count") or 1)),
        )
