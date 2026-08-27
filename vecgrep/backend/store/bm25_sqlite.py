"""BM25 keyword index in SQLite, as an alternative to the pickled sidecar.

The pickle backend keeps a whole corpus resident: ids, tokenised docs, and the
full payload dicts, loaded in one go and held until evicted. That is fine for a
small corpus and expensive for a large one — and the cost is dominated by the
part that is not the keyword index at all. Measured on a real 2,619-chunk
corpus: payloads 3.52 MB, docs 1.06 MB, ids 0.10 MB, by_source 0.02 MB. Three
quarters of the sidecar is payload, because payloads carry the surrounding
context text.

So moving only the scoring to FTS5 would save a fifth. This moves the whole
thing: rows live on disk, SQLite memory-maps what it needs, and nothing is held
across calls. FTS5 provides BM25 ranking natively, and the same tokeniser feeds
it, so ranking semantics are preserved rather than approximated.

Nothing here is enabled by default. The pickle store remains the shipped
backend until an equivalence run says this one ranks identically.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path

from vecgrep.backend.store.bm25_store import (
    _coverage_factor,
    tokenize,
)

# FTS5 ranks with bm25(), which returns a NEGATIVE number where more negative
# is a better match. Every caller here expects "higher is better", the same
# contract the pickle store exposes, so scores are negated on the way out.
#
# Candidates are over-fetched before coverage filtering: `_coverage_factor`
# can drop a row entirely, so asking FTS5 for exactly top_k would return fewer
# than top_k after filtering. The multiplier is generous because the filter is
# cheap and a short result list is the failure we are avoiding.
_OVERFETCH = 8
_MIN_CANDIDATES = 200

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    pos       INTEGER PRIMARY KEY,
    cid       TEXT NOT NULL,
    source_id TEXT NOT NULL DEFAULT '',
    tokens    TEXT NOT NULL,
    payload   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source_id);
CREATE INDEX IF NOT EXISTS chunks_cid ON chunks(cid);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5(tokens, content='chunks', content_rowid='pos', tokenize='unicode61');
"""


def _fts_query(q_tokens: list[str]) -> str:
    """An FTS5 MATCH expression that ORs the query tokens.

    FTS5 defaults to AND between bare terms; BM25Okapi scores a document that
    contains ANY query term, so the two only agree if the OR is explicit. Each
    token is double-quoted so a token that happens to collide with FTS5 syntax
    (NEAR, OR, a bare digit run) is treated as a literal.
    """
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in q_tokens)


class BM25SqliteStore:
    """The BM25Store interface, backed by one SQLite file per corpus."""

    def __init__(self, root: Path | None) -> None:
        # root=None mirrors the pickle store's ephemeral mode: an in-memory
        # database per corpus, discarded with the process.
        self.root = root
        self._conns: dict[str, sqlite3.Connection] = {}
        self._lock = threading.RLock()
        self._bulk: set[str] = set()

    # ── connections ───────────────────────────────────────────────────────
    def _path(self, corpus: str) -> Path | None:
        return None if self.root is None else self.root / f"{corpus}.db"

    def _conn(self, corpus: str) -> sqlite3.Connection:
        with self._lock:
            c = self._conns.get(corpus)
            if c is not None:
                return c
            p = self._path(corpus)
            if p is None:
                c = sqlite3.connect(":memory:", check_same_thread=False)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                c = sqlite3.connect(str(p), check_same_thread=False)
                c.execute("PRAGMA journal_mode=WAL")
                c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=10000")
            # Keep SQLite's own page cache small. The point of this backend is
            # that the index does NOT live in the process; a large cache here
            # would quietly reintroduce what it was built to remove.
            c.execute("PRAGMA cache_size=-2000")
            c.executescript(_SCHEMA)
            c.commit()
            self._conns[corpus] = c
            return c

    def close(self, corpus: str) -> None:
        with self._lock:
            c = self._conns.pop(corpus, None)
        if c is not None:
            c.close()

    def close_all(self) -> None:
        with self._lock:
            names = list(self._conns)
        for n in names:
            self.close(n)

    # `evict` exists so callers written against the pickle store keep working.
    # There is nothing resident to release, so it just drops the handle.
    evict = close

    # ── reads ─────────────────────────────────────────────────────────────
    def exists(self, corpus: str) -> bool:
        p = self._path(corpus)
        return bool(p and p.exists())

    def count(self, corpus: str) -> int:
        return int(self._conn(corpus).execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def get_by_id(self, corpus: str, cid: str) -> dict | None:
        row = self._conn(corpus).execute(
            "SELECT payload FROM chunks WHERE cid = ? LIMIT 1", (cid,)
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def iter_payloads(self, corpus: str) -> Iterator[dict]:
        """Every payload, streamed. The replacement for reaching into
        `_load(corpus).payloads`: rows arrive one at a time and are not
        retained, so a full scan costs one row of memory rather than a corpus."""
        cur = self._conn(corpus).execute("SELECT payload FROM chunks ORDER BY pos")
        for (raw,) in cur:
            yield json.loads(raw)

    def iter_sources(self, corpus: str) -> Iterator[tuple[str, dict]]:
        """(source_id, payload of its first chunk) for each distinct source,
        replacing iteration over `by_source`."""
        cur = self._conn(corpus).execute(
            "SELECT source_id, payload FROM chunks WHERE pos IN "
            "(SELECT MIN(pos) FROM chunks GROUP BY source_id) ORDER BY pos"
        )
        for sid, raw in cur:
            yield sid, json.loads(raw)

    def payload_for_source(self, corpus: str, source_id: str) -> dict | None:
        row = self._conn(corpus).execute(
            "SELECT payload FROM chunks WHERE source_id = ? ORDER BY pos LIMIT 1",
            (source_id,),
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def source_counts(self, corpus: str) -> dict[str, int]:
        return {
            sid: n
            for sid, n in self._conn(corpus).execute(
                "SELECT source_id, COUNT(*) FROM chunks GROUP BY source_id"
            )
        }

    # ── writes ────────────────────────────────────────────────────────────
    def _next_pos(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT COALESCE(MAX(pos), -1) + 1 FROM chunks").fetchone()
        return int(row[0])

    def _insert(self, conn: sqlite3.Connection, pos: int, cid: str,
                source_id: str, tokens: str, payload: dict) -> None:
        conn.execute(
            "INSERT INTO chunks(pos, cid, source_id, tokens, payload) VALUES (?,?,?,?,?)",
            (pos, cid, source_id, tokens, json.dumps(payload)),
        )
        # External-content FTS5 is not written by the table itself; each row is
        # mirrored explicitly so the two stay in lockstep.
        conn.execute("INSERT INTO chunks_fts(rowid, tokens) VALUES (?, ?)", (pos, tokens))

    def upsert(self, corpus: str, ids: list[str], texts: list[str],
               payloads: list[dict]) -> None:
        if not ids:
            return
        conn = self._conn(corpus)
        pos = self._next_pos(conn)
        for cid, text, payload in zip(ids, texts, payloads):
            self._insert(conn, pos, cid, payload.get("source_id", "") or "",
                         " ".join(tokenize(text)), payload)
            pos += 1
        if corpus not in self._bulk:
            conn.commit()

    def replace(self, corpus: str, records: Iterable[tuple[str, str, dict]]) -> int:
        """Atomically replace a corpus. Readers see the old contents or the
        whole new one, never a half-written index."""
        conn = self._conn(corpus)
        n = 0
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM chunks_fts")
            conn.execute("DELETE FROM chunks")
            for cid, text, payload in records:
                self._insert(conn, n, cid, payload.get("source_id", "") or "",
                             " ".join(tokenize(text)), payload)
                n += 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return n

    def delete_by_source(self, corpus: str, source_id: str) -> None:
        conn = self._conn(corpus)
        rows = [r[0] for r in conn.execute(
            "SELECT pos FROM chunks WHERE source_id = ?", (source_id,))]
        if not rows:
            return
        conn.executemany("DELETE FROM chunks_fts WHERE rowid = ?", [(r,) for r in rows])
        conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
        if corpus not in self._bulk:
            conn.commit()

    def drop(self, corpus: str) -> None:
        self.close(corpus)
        p = self._path(corpus)
        if p and p.exists():
            p.unlink()
        for suffix in ("-wal", "-shm"):
            side = None if p is None else p.with_name(p.name + suffix)
            if side is not None and side.exists():
                side.unlink()

    # ── search ────────────────────────────────────────────────────────────
    def search(self, corpus: str, query: str, top_k: int) -> list[tuple[str, float, dict]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        conn = self._conn(corpus)
        limit = max(_MIN_CANDIDATES, top_k * _OVERFETCH)
        try:
            cur = conn.execute(
                "SELECT c.cid, c.tokens, c.payload, bm25(chunks_fts) "
                "FROM chunks_fts JOIN chunks c ON c.pos = chunks_fts.rowid "
                "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                (_fts_query(q_tokens), limit),
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            # A malformed MATCH expression must not take down a search.
            return []

        ranked: list[tuple[float, str, dict]] = []
        for cid, tokens, raw_payload, score in rows:
            factor = _coverage_factor(q_tokens, tokens.split())
            if factor is None:
                continue
            # bm25() is negative-better; flip it so higher is better, matching
            # the pickle store's contract and the RRF fusion above it.
            ranked.append((-float(score) * factor, cid, json.loads(raw_payload)))

        # Ties broken by cid so the same query returns the same order every
        # time. FTS5 would otherwise hand back rowid order, which is insertion
        # order, which changes when a corpus is rebuilt.
        ranked.sort(key=lambda r: (-r[0], r[1]))
        return [(cid, s, payload) for s, cid, payload in ranked[:top_k]]
