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
import logging
import os
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from vecgrep.backend.store.bm25_store import (
    _coverage_factor,
    query_tokens,
    tokenize,
)

logger = logging.getLogger(__name__)

# Keep WAL growth bounded without blocking readers. SQLite's default automatic
# checkpoint is also 1,000 pages, but spelling it out makes the contract
# inspectable and lets operators tune unusual filesystems. The byte limit is
# applied when a checkpoint can reset the WAL; a busy reader may defer the
# shrink, never the committed data.
WAL_AUTOCHECKPOINT_PAGES = max(
    1, int(os.environ.get("VECGREP_BM25_WAL_AUTOCHECKPOINT_PAGES", "1000"))
)
WAL_JOURNAL_SIZE_LIMIT_BYTES = max(
    0,
    int(os.environ.get("VECGREP_BM25_WAL_LIMIT_MB", "64")) * 1024 * 1024,
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


def _fts_and_query(q_tokens: list[str]) -> str:
    """An FTS5 MATCH expression that ANDs distinct query tokens."""
    unique = list(dict.fromkeys(q_tokens))
    return " AND ".join('"' + t.replace('"', '""') + '"' for t in unique)


def _without_source_text(payload: dict) -> dict:
    if "source_text" not in payload:
        return payload
    return {k: v for k, v in payload.items() if k != "source_text"}


class BM25SqliteStore:
    """The BM25Store interface, backed by one SQLite file per corpus."""

    def __init__(self, root: Path | None) -> None:
        # root=None mirrors the pickle store's ephemeral mode: an in-memory
        # database per corpus, discarded with the process.
        self.root = root
        self._conns: dict[str, sqlite3.Connection] = {}
        self._corpus_locks: dict[str, threading.RLock] = {}
        self._counts: dict[str, tuple[tuple[int, int] | None, int]] = {}
        self._lock = threading.RLock()
        self._bulk: set[str] = set()

    # ── connections ───────────────────────────────────────────────────────
    def _path(self, corpus: str) -> Path | None:
        return None if self.root is None else self.root / f"{corpus}.db"

    def _corpus_lock(self, corpus: str) -> threading.RLock:
        with self._lock:
            lock = self._corpus_locks.get(corpus)
            if lock is None:
                lock = threading.RLock()
                self._corpus_locks[corpus] = lock
            return lock

    def _disk_version(
        self, corpus: str, conn: sqlite3.Connection | None = None
    ) -> tuple[int, int, int, int, int] | None:
        p = self._path(corpus)
        if p is None:
            return None
        try:
            st = p.stat()
            wal = Path(f"{p}-wal")
            if wal.exists():
                wst = wal.stat()
                wal_mtime, wal_size = wst.st_mtime_ns, wst.st_size
            else:
                wal_mtime, wal_size = 0, 0
            c = conn or self._conns.get(corpus)
            data_ver = (
                int(c.execute("PRAGMA data_version").fetchone()[0])
                if c is not None
                else 0
            )
            return (st.st_mtime_ns, st.st_size, wal_mtime, wal_size, data_ver)
        except OSError:
            return None

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
                c.execute(f"PRAGMA wal_autocheckpoint={WAL_AUTOCHECKPOINT_PAGES}")
                c.execute(f"PRAGMA journal_size_limit={WAL_JOURNAL_SIZE_LIMIT_BYTES}")
            c.execute("PRAGMA busy_timeout=10000")
            # Keep SQLite's own page cache small. The point of this backend is
            # that the index does NOT live in the process; a large cache here
            # would quietly reintroduce what it was built to remove.
            c.execute("PRAGMA cache_size=-2000")
            c.executescript(_SCHEMA)
            c.commit()
            self._conns[corpus] = c
            return c

    def _checkpoint(self, corpus: str, conn: sqlite3.Connection) -> tuple[int, int, int] | None:
        """Best-effort passive checkpoint after a potentially large mutation.

        PASSIVE never waits for readers, so searches keep their existing
        latency and snapshot semantics. If a reader pins old WAL frames, a
        later mutation or clean connection close gets another chance.
        """
        try:
            row = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        except sqlite3.Error as exc:
            logger.warning("BM25 WAL checkpoint deferred for %s: %s", corpus, exc)
            return None
        if row is None:
            return None
        busy, log_frames, checkpointed = (int(row[0]), int(row[1]), int(row[2]))
        if busy:
            logger.info(
                "BM25 WAL checkpoint busy for %s frames=%d checkpointed=%d",
                corpus,
                log_frames,
                checkpointed,
            )
        self._truncate_oversized_wal(corpus, conn)
        return busy, log_frames, checkpointed

    def _truncate_oversized_wal(self, corpus: str, conn: sqlite3.Connection) -> None:
        """Reclaim a WAL that outgrew its limit and would otherwise stay big.

        journal_size_limit only truncates when a checkpoint RESETS the log. A
        bulk index whose writer then goes quiet leaves the full-sized WAL on
        disk indefinitely — repos.db-wal reached 239 MB, larger than its
        database, exactly this way (2026-09-08). PASSIVE checkpoints never
        reset past a reader, so escalate to TRUNCATE, but with a short busy
        timeout: a pinning reader defers the shrink (a later mutation retries)
        rather than stalling the mutation path behind it.
        """
        limit = WAL_JOURNAL_SIZE_LIMIT_BYTES
        path = self._path(corpus)
        if path is None or limit <= 0:
            return
        wal = Path(f"{path}-wal")
        try:
            if not wal.exists() or wal.stat().st_size <= limit:
                return
        except OSError:
            return
        conn.execute("PRAGMA busy_timeout=100")
        try:
            row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        except sqlite3.Error as exc:
            logger.warning("BM25 WAL truncate deferred for %s: %s", corpus, exc)
            return
        finally:
            conn.execute("PRAGMA busy_timeout=10000")
        if row is not None and int(row[0]):
            logger.info(
                "BM25 WAL truncate deferred for %s: reader pins the log", corpus
            )
        else:
            logger.info("BM25 WAL truncated for %s", corpus)

    def close(self, corpus: str) -> None:
        with self._lock:
            c = self._conns.pop(corpus, None)
            self._counts.pop(corpus, None)
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

    def dirty_corpora(self) -> list[str]:
        """SQLite transactions need no external crash-recovery marker."""
        return []

    def clear_dirty(self, corpus: str) -> None:
        """Compatibility no-op for the pickle store's marker protocol."""

    @contextmanager
    def bulk(self, corpus: str):
        """Commit a multi-source update once, or roll it back as a unit."""
        with self._corpus_lock(corpus):
            if corpus in self._bulk:
                yield
                return
            conn = self._conn(corpus)
            self._bulk.add(corpus)
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield
                conn.commit()
                self._counts.pop(corpus, None)
                self._checkpoint(corpus, conn)
            except Exception:
                conn.rollback()
                self._counts.pop(corpus, None)
                raise
            finally:
                self._bulk.discard(corpus)

    # ── reads ─────────────────────────────────────────────────────────────
    def exists(self, corpus: str) -> bool:
        p = self._path(corpus)
        return corpus in self._conns if p is None else p.exists()

    def count(self, corpus: str) -> int:
        with self._corpus_lock(corpus):
            conn = self._conn(corpus)
            ver = self._disk_version(corpus, conn)
            cached = self._counts.get(corpus)
            if cached is not None and ver is not None and cached[0] == ver:
                return cached[1]
            n = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            self._counts[corpus] = (self._disk_version(corpus, conn), n)
            return n

    def get_by_id(self, corpus: str, cid: str) -> dict | None:
        with self._corpus_lock(corpus):
            row = self._conn(corpus).execute(
                "SELECT payload FROM chunks WHERE cid = ? LIMIT 1", (cid,)
            ).fetchone()
            return None if row is None else json.loads(row[0])

    def iter_payloads(self, corpus: str) -> Iterator[dict]:
        """Every payload, streamed. The replacement for reaching into
        `_load(corpus).payloads`: rows arrive one at a time and are not
        retained, so a full scan costs one row of memory rather than a corpus."""
        conn = self._conn(corpus)
        cur = conn.execute("SELECT payload FROM chunks ORDER BY pos")
        for (raw,) in cur:
            yield json.loads(raw)

    def iter_sources(self, corpus: str) -> Iterator[tuple[str, dict, int]]:
        """(source_id, payload of its first chunk, chunk count) per source.

        Ordered by first appearance, which is what iterating the pickle
        store's `by_source` gave, so callers see the same sequence."""
        conn = self._conn(corpus)
        cur = conn.execute(
            "SELECT c.source_id, c.payload, g.n FROM chunks c "
            "JOIN (SELECT source_id, MIN(pos) AS first_pos, COUNT(*) AS n "
            "      FROM chunks GROUP BY source_id) g "
            "  ON g.first_pos = c.pos "
            "ORDER BY c.pos"
        )
        for sid, raw, n in cur:
            yield sid, json.loads(raw), int(n)

    def first_chunk_for_source(self, corpus: str, source_id: str) -> tuple[str, dict] | None:
        """(chunk id, payload) of a source's first chunk, or None."""
        with self._corpus_lock(corpus):
            row = self._conn(corpus).execute(
                "SELECT cid, payload FROM chunks WHERE source_id = ? ORDER BY pos LIMIT 1",
                (source_id,),
            ).fetchone()
            return None if row is None else (row[0], json.loads(row[1]))

    def update_payloads(self, corpus: str, mutate) -> int:
        """Apply `mutate(payload)` to every payload, writing back the ones it
        changed. `mutate` edits in place and returns True when it changed
        something. Rows are streamed and written in one transaction."""
        with self._corpus_lock(corpus):
            conn = self._conn(corpus)
            changed: list[tuple[str, int]] = []
            for pos, raw in conn.execute("SELECT pos, payload FROM chunks ORDER BY pos"):
                payload = json.loads(raw)
                if mutate(payload):
                    changed.append((json.dumps(_without_source_text(payload)), pos))
            if changed:
                conn.executemany("UPDATE chunks SET payload = ? WHERE pos = ?", changed)
                conn.commit()
                self._counts.pop(corpus, None)
                self._checkpoint(corpus, conn)
            return len(changed)

    def source_counts(self, corpus: str) -> dict[str, int]:
        with self._corpus_lock(corpus):
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
        clean_payload = _without_source_text(payload)
        conn.execute(
            "INSERT INTO chunks(pos, cid, source_id, tokens, payload) VALUES (?,?,?,?,?)",
            (pos, cid, source_id, tokens, json.dumps(clean_payload)),
        )
        # External-content FTS5 is not written by the table itself; each row is
        # mirrored explicitly so the two stay in lockstep.
        conn.execute("INSERT INTO chunks_fts(rowid, tokens) VALUES (?, ?)", (pos, tokens))

    def upsert(self, corpus: str, ids: list[str], texts: list[str],
               payloads: list[dict]) -> None:
        if not ids:
            return
        with self._corpus_lock(corpus):
            conn = self._conn(corpus)
            pos = self._next_pos(conn)
            for cid, text, payload in zip(ids, texts, payloads):
                self._insert(conn, pos, cid, payload.get("source_id", "") or "",
                             " ".join(tokenize(text)), payload)
                pos += 1
            if corpus not in self._bulk:
                conn.commit()
            self._counts.pop(corpus, None)

    def replace(self, corpus: str, records: Iterable[tuple[str, str, dict]]) -> int:
        """Atomically replace a corpus. Readers see the old contents or the
        whole new one, never a half-written index."""
        with self._corpus_lock(corpus):
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
                self._counts.pop(corpus, None)
                self._checkpoint(corpus, conn)
            except Exception:
                conn.rollback()
                self._counts.pop(corpus, None)
                raise
            return n

    def delete_by_source(self, corpus: str, source_id: str) -> None:
        with self._corpus_lock(corpus):
            conn = self._conn(corpus)
            rows = [r[0] for r in conn.execute(
                "SELECT pos FROM chunks WHERE source_id = ?", (source_id,))]
            if not rows:
                return
            conn.executemany("DELETE FROM chunks_fts WHERE rowid = ?", [(r,) for r in rows])
            conn.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
            if corpus not in self._bulk:
                conn.commit()
                self._checkpoint(corpus, conn)
            self._counts.pop(corpus, None)

    def delete_ids(self, corpus: str, cids: list[str]) -> None:
        if not cids:
            return
        with self._corpus_lock(corpus):
            conn = self._conn(corpus)
            placeholders = ",".join("?" for _ in cids)
            rows = [
                r[0]
                for r in conn.execute(
                    f"SELECT pos FROM chunks WHERE cid IN ({placeholders})",
                    list(cids),
                )
            ]
            if not rows:
                return
            conn.executemany("DELETE FROM chunks_fts WHERE rowid = ?", [(r,) for r in rows])
            conn.execute(
                f"DELETE FROM chunks WHERE cid IN ({placeholders})",
                list(cids),
            )
            if corpus not in self._bulk:
                conn.commit()
                self._checkpoint(corpus, conn)
            self._counts.pop(corpus, None)

    def drop(self, corpus: str) -> None:
        self.close(corpus)
        p = self._path(corpus)
        if p is None:
            return
        # The .db, its sidecars, AND the legacy pickle this backend replaced.
        # convert() builds <corpus>.db from <corpus>.pkl, so a converted corpus
        # carries both on disk. Dropping only the .db stranded a pickle owning a
        # corpus that no longer existed: deleting three eval corpora freed 639 MB
        # of qdrant and left 298 MB of pickle behind (2026-08-30).
        for stale in (
            p,
            p.with_name(p.name + "-wal"),
            p.with_name(p.name + "-shm"),
            p.with_name(f"{corpus}.pkl"),
            p.with_name(f"{corpus}.pkl.dirty"),
        ):
            stale.unlink(missing_ok=True)

    # ── search ────────────────────────────────────────────────────────────
    def search(
        self,
        corpus: str,
        query: str,
        top_k: int,
        *,
        source_ids: set[str] | None = None,
    ) -> list[tuple[str, float, dict]]:
        if source_ids is not None and not source_ids:
            return []
        q_tokens = query_tokens(query)
        if not q_tokens:
            return []
        with self._corpus_lock(corpus):
            conn = self._conn(corpus)
            limit = max(_MIN_CANDIDATES, top_k * _OVERFETCH)
            source_clause = ""
            source_params: list[str] = []
            if source_ids is not None and len(source_ids) <= 200:
                source_params = sorted(source_ids)
                placeholders = ",".join("?" for _ in source_params)
                source_clause = f" AND c.source_id IN ({placeholders})"

            # Two-stage retrieval:
            # 1) Fetch lightweight (pos, cid, tokens, bm25) rows WITHOUT
            #    joining/deserializing the large payload column on all 400
            #    overfetched candidates.
            # 2) For multi-token queries, first pull up to top_k full-quorum
            #    (AND) matches so single-term high-TF noise in the 400-row OR
            #    window can never crowd out exact multi-term matches.
            by_pos: dict[int, tuple[str, str, float]] = {}
            unique_tokens = list(dict.fromkeys(q_tokens))
            if len(unique_tokens) >= 2:
                and_cur = conn.execute(
                    "SELECT c.pos, c.cid, c.tokens, bm25(chunks_fts) "
                    "FROM chunks_fts JOIN chunks c ON c.pos = chunks_fts.rowid "
                    f"WHERE chunks_fts MATCH ?{source_clause} "
                    "ORDER BY bm25(chunks_fts) LIMIT ?",
                    (_fts_and_query(unique_tokens), *source_params, top_k),
                )
                for pos, cid, tokens, score in and_cur.fetchall():
                    by_pos[int(pos)] = (str(cid), str(tokens), float(score))

            cur = conn.execute(
                "SELECT c.pos, c.cid, c.tokens, bm25(chunks_fts) "
                "FROM chunks_fts JOIN chunks c ON c.pos = chunks_fts.rowid "
                f"WHERE chunks_fts MATCH ?{source_clause} "
                "ORDER BY bm25(chunks_fts) LIMIT ?",
                (_fts_query(q_tokens), *source_params, limit),
            )
            # Query tokens are quoted by _fts_query. Operational failures indicate
            # an unhealthy store and must not silently become vector-only results.
            for pos, cid, tokens, score in cur.fetchall():
                by_pos[int(pos)] = (str(cid), str(tokens), float(score))

            if not by_pos:
                return []

            ranked: list[tuple[float, str, int]] = []
            for pos, (cid, tokens, score) in by_pos.items():
                factor = _coverage_factor(q_tokens, tokens.split())
                if factor is None:
                    continue
                # bm25() is negative-better; flip it so higher is better, matching
                # the pickle store's contract and the RRF fusion above it.
                ranked.append((-float(score) * factor, cid, pos))

            # Ties broken by cid so the same query returns the same order every
            # time. FTS5 would otherwise hand back rowid order, which is insertion
            # order, which changes when a corpus is rebuilt.
            ranked.sort(key=lambda r: (-r[0], r[1]))
            winners = ranked[:top_k]
            if not winners:
                return []

            win_positions = [pos for _, _, pos in winners]
            placeholders = ",".join("?" for _ in win_positions)
            payload_rows = conn.execute(
                f"SELECT pos, payload FROM chunks WHERE pos IN ({placeholders})",
                win_positions,
            ).fetchall()
            payload_by_pos = {
                int(pos): json.loads(raw_payload)
                for pos, raw_payload in payload_rows
            }
            return [
                (cid, s, payload_by_pos[pos])
                for s, cid, pos in winners
                if pos in payload_by_pos
            ]


def migrate_from_pickle(root: Path, corpus: str, *, force: bool = False) -> int:
    """Build `<corpus>.db` from an existing `<corpus>.pkl`. Returns rows written.

    Switching backends without this is a silent downgrade: the SQLite store
    would start empty, every lexical query would return nothing, and hybrid
    search would quietly become vector-only — with no error anywhere, because
    an empty index is a legitimate state. Convert first, verify counts, then
    switch.

    Reads the pickle directly rather than going through BM25Store so the
    conversion does not depend on cache behaviour, and writes through
    `replace()` so the result is atomic.
    """
    import pickle

    src = root / f"{corpus}.pkl"
    if not src.exists():
        raise FileNotFoundError(src)
    dest = root / f"{corpus}.db"
    if dest.exists() and not force:
        raise FileExistsError(f"{dest} exists; pass force=True to rebuild it")

    idx = pickle.loads(src.read_bytes())
    store = BM25SqliteStore(root)
    try:
        # The pickle holds tokens, not the original text. `replace()` tokenises
        # what it is given, and tokenising already-split tokens is idempotent
        # for this tokeniser, so joining them back is lossless here.
        written = store.replace(
            corpus,
            (
                (cid, " ".join(doc), payload)
                for cid, doc, payload in zip(idx.ids, idx.docs, idx.payloads)
            ),
        )
    finally:
        store.close(corpus)
    return written


def migrate_all(root: Path, *, force: bool = False) -> dict[str, int]:
    """Convert every pickled sidecar under `root`. Returns corpus -> rows."""
    out: dict[str, int] = {}
    for src in sorted(root.glob("*.pkl")):
        out[src.stem] = migrate_from_pickle(root, src.stem, force=force)
    return out
