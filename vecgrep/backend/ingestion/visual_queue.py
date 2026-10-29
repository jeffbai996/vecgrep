"""Durable optional enrichment, independent of search and embedding services."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

PROCESSOR_VERSION = 'visual-v1'
MAX_SOURCE_BYTES = 200 * 1024 * 1024


def enabled() -> bool:
    return os.environ.get('VECGREP_VISUAL_ENABLED', '').lower() in {'1', 'true', 'yes'}


def fingerprint(path: Path) -> str:
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError('Visual source exceeds the 200 MiB processing limit')
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class VisualQueue:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS jobs (
                    key TEXT PRIMARY KEY, source TEXT NOT NULL, digest TEXT NOT NULL,
                    version TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
                    text TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '',
                    due REAL NOT NULL DEFAULT 0, lease REAL NOT NULL DEFAULT 0,
                    token TEXT NOT NULL DEFAULT '', attempts INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS targets (
                    source TEXT NOT NULL, corpus TEXT NOT NULL, chunker TEXT NOT NULL,
                    delivered TEXT NOT NULL DEFAULT '', PRIMARY KEY(source, corpus));
            ''')

    @classmethod
    def default(cls) -> VisualQueue:
        from ..config import get_settings
        return cls(get_settings().home / 'visual' / 'queue.db')

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def enqueue(self, path: Path) -> str:
        source = str(path.resolve())
        digest = fingerprint(path)
        version = os.environ.get('VECGREP_VISUAL_VERSION', PROCESSOR_VERSION)
        key = hashlib.sha256(f'{source}\0{digest}\0{version}'.encode()).hexdigest()
        with self.connect() as db:
            # A changed source invalidates unfinished work; cached results are
            # selected by content hash, so an old description cannot leak in.
            db.execute("UPDATE jobs SET state='cancelled' WHERE source=? AND key!=? AND state IN ('pending','running')", (source, key))
            db.execute('INSERT OR IGNORE INTO jobs(key,source,digest,version) VALUES(?,?,?,?)', (key, source, digest, version))
            db.execute("UPDATE jobs SET state='pending',due=0,error='' WHERE key=? AND state='cancelled'", (key,))
        return key

    def result(self, path: Path) -> str | None:
        key = self.enqueue(path)
        with self.connect() as db:
            row = db.execute('SELECT text FROM jobs WHERE key=?', (key,)).fetchone()
        return row['text'] or None

    def state(self, path: Path) -> str:
        key = self.enqueue(path)
        with self.connect() as db:
            return db.execute('SELECT state FROM jobs WHERE key=?', (key,)).fetchone()['state']

    def target(self, path: Path, corpus: str, chunker: str) -> None:
        with self.connect() as db:
            db.execute('INSERT INTO targets(source,corpus,chunker) VALUES(?,?,?) ON CONFLICT(source,corpus) DO UPDATE SET chunker=excluded.chunker', (str(path.resolve()), corpus, chunker))

    def cancel_target(self, source: str | None, corpus: str) -> None:
        with self.connect() as db:
            if source is None:
                db.execute('DELETE FROM targets WHERE corpus=?', (corpus,))
            else:
                db.execute('DELETE FROM targets WHERE source=? AND corpus=?', (source, corpus))

    def claim(self, *, now: float | None = None, lease_s: int = 7200) -> dict | None:
        now = time.time() if now is None else now
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT * FROM jobs WHERE (state='pending' AND due<=?) OR (state='running' AND lease<=?) ORDER BY due,key LIMIT 1", (now, now)).fetchone()
            if row is None:
                return None
            token = uuid.uuid4().hex
            db.execute("UPDATE jobs SET state='running',token=?,lease=?,attempts=attempts+1 WHERE key=?", (token, now + lease_s, row['key']))
            return {**dict(row), 'token': token, 'attempts': row['attempts'] + 1}

    def current(self, job: dict) -> bool:
        try:
            return fingerprint(Path(job['source'])) == job['digest']
        except (OSError, ValueError):
            return False

    def complete(self, job: dict, text: str, *, now: float | None = None) -> bool:
        if not text.strip():
            raise ValueError('Enrichment returned no searchable text')
        current = self.current(job)
        with self.connect() as db:
            changed = db.execute("UPDATE jobs SET state=?,text=?,error='',lease=0 WHERE key=? AND token=? AND state='running'", ('done' if current else 'cancelled', text if current else '', job['key'], job['token'])).rowcount
        return bool(changed and current)

    def retry(self, job: dict, error: str, *, text: str = '', now: float | None = None) -> None:
        now = time.time() if now is None else now
        current = self.current(job)
        delay = min(3600, 60 * 2 ** min(job['attempts'] - 1, 6))
        with self.connect() as db:
            db.execute("UPDATE jobs SET state=?,text=CASE WHEN ?!='' THEN ? ELSE text END,error=?,due=?,lease=0 WHERE key=? AND token=? AND state='running'", ('pending' if current else 'cancelled', text if current else '', text if current else '', error[:1000], now + delay, job['key'], job['token']))

    def deliveries(self) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT j.*,t.corpus,t.chunker,t.delivered FROM jobs j JOIN targets t ON j.source=t.source WHERE j.text!='' AND j.state IN ('done','pending')").fetchall()
        out = []
        for row in rows:
            item = dict(row)
            revision = hashlib.sha256((item['key'] + item['text']).encode()).hexdigest()
            if revision != item['delivered'] and self.current(item):
                out.append({**item, 'revision': revision})
        return out

    def allows_delivery(self, source: str, corpus: str, key: str) -> bool:
        with self.connect() as db:
            row = db.execute("SELECT j.* FROM jobs j JOIN targets t ON j.source=t.source WHERE j.key=? AND j.source=? AND t.corpus=? AND j.state IN ('done','pending') AND j.text!=''", (key, str(Path(source).resolve()), corpus)).fetchone()
        return row is not None and self.current(dict(row))

    def delivered(self, item: dict) -> None:
        with self.connect() as db:
            db.execute('UPDATE targets SET delivered=? WHERE source=? AND corpus=?', (item['revision'], item['source'], item['corpus']))

    def status(self) -> list[dict]:
        with self.connect() as db:
            return [dict(row) for row in db.execute('SELECT key,source,state,error,attempts,due FROM jobs ORDER BY source')]
