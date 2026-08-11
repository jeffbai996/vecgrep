"""Crash journal for cross-store corpus mutations.

The journal is intentionally small and boring: one atomic JSON intent per
corpus.  Corpus admission guarantees there is at most one outstanding writer
for that corpus.  A source index also records the old Qdrant points because an
upsert can span batches; recovery can then choose a complete old or new state,
never bless a half-batch mixture as healthy.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class MutationJournal:
    def __init__(self, root: Path | None) -> None:
        self.root = root
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe(corpus: str) -> str:
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in corpus)

    def path(self, corpus: str) -> Path | None:
        return None if self.root is None else self.root / f"{self._safe(corpus)}.json"

    def write(self, record: dict) -> None:
        path = self.path(str(record["corpus"]))
        if path is None:
            return
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
        fd, temporary = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            self._fsync_dir(path.parent)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def read(self, corpus: str) -> dict | None:
        path = self.path(corpus)
        if path is None or not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def pending_corpora(self) -> list[str]:
        if self.root is None:
            return []
        out: list[str] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                corpus = record.get("corpus")
                if isinstance(corpus, str) and corpus:
                    out.append(corpus)
            except (OSError, json.JSONDecodeError):
                # Leave a malformed intent in place. Startup recovery will
                # report it loudly instead of silently discarding evidence.
                raise RuntimeError(f"unreadable mutation journal: {path}")
        return out

    def finish(self, corpus: str) -> None:
        path = self.path(corpus)
        if path is None:
            return
        path.unlink(missing_ok=True)
        self._fsync_dir(path.parent)

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
