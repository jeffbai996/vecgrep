"""Named corpus registry.

A corpus is a logical grouping of documents stored in a single Qdrant
collection. Metadata pins the embedding backend + dimension so we
refuse to mix models within one corpus.
"""
from __future__ import annotations

import contextlib

import json
import os
import re
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..mutation import CorpusLocks

EPHEMERAL_NAME = "__ephemeral__"
# Internal corpus names that bypass the user-facing name rule (they're never
# user-supplied — created transiently by migrate).
_INTERNAL_PREFIX = "__migrate__"
_VALID_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$")


class CorpusError(RuntimeError):
    pass


@dataclass
class Corpus:
    name: str
    embed_backend: str  # "ollama" | "openai"
    embed_model: str
    dim: int
    chunker: str = "sentence_window"
    doc_count: int = 0
    chunk_count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    sources: list[str] = field(default_factory=list)
    # Maps source_id -> content sha256. Used to skip re-embedding sources
    # whose contents haven't changed since the last index. Empty for old
    # corpora — they fall back to "always re-embed" until the next index.
    source_hashes: dict[str, str] = field(default_factory=dict)
    # Recency-decay half-life in days. When set, a hit's fused score is
    # multiplied by 0.5 ** (age_days / half_life), so a chunk one half-life
    # old ranks as if half as relevant. None = no decay (default; preserves
    # prior behavior). Tune per corpus: fast for chat/journal, slow for
    # reference, off for static seed material.
    decay_half_life_days: float | None = None
    # Cross-corpus rank weight. Multiplies a hit's fused score (and biases
    # final display order) when a search spans corpora, so a small curated
    # reference corpus can outrank a high-volume transcript corpus at
    # comparable relevance. 1.0 = neutral (default, no change).
    rank_weight: float = 1.0
    # Per-corpus BM25 fusion weight. None = the install-wide default
    # (VECGREP_BM25_WEIGHT). Short semantic notes want less lexical pull than a
    # transcript corpus full of names, dates and exact phrases; the 2026-08
    # eval measured the two optima a factor of three apart.
    bm25_weight: float | None = None
    # On-disk vector storage type: "float32" (default) or "float16". Halves
    # the dominant term of a large collection's footprint at no measured recall
    # cost (docs/STORAGE_RETRIEVAL_2026-08, 60 chats gold cases: identical
    # hit@1/3/10, qdrant -35%, vector p50 312ms -> 100ms). It is pinned when a
    # qdrant collection is CREATED and cannot be changed in place, so it lives
    # here rather than in a setting: every path that can (re)create a
    # collection -- index, create, rename-migrate, crash recovery, backup
    # restore -- has to read it back, or the collection silently reverts to
    # float32 and the corpus quietly gets slower and bigger again.
    datatype: str = "float32"
    # Operator-authored routing context for agents choosing the smallest
    # relevant search scope. These fields are metadata only: changing them
    # never requires a corpus rebuild.
    description: str = ""
    use_for: list[str] = field(default_factory=list)
    avoid_for: list[str] = field(default_factory=list)


class CorpusRegistry:
    def __init__(self, path: Path, locks: CorpusLocks | None = None,
                 *, in_memory: bool = False) -> None:
        self.path = path
        self.in_memory = in_memory
        self.locks = locks or CorpusLocks(path.parent / "locks")
        # corpus name -> pending entry while deferred_saves() is open
        self._deferred: dict[str, Corpus | None] = {}
        self._corpora: dict[str, Corpus] = {}
        if not in_memory:
            with self.locks.registry_read():
                self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except json.JSONDecodeError as exc:
            # The file exists but is unparseable — almost always a torn read of a
            # concurrent (previously non-atomic) write. DO NOT fall back to {}:
            # an empty load here would let the next upsert save an empty registry
            # over good data, silently dropping every corpus. Raise so the caller
            # aborts the mutation and the on-disk registry stays intact. Paired
            # with the atomic _save below, a torn read is now transient, not fatal.
            raise CorpusError(
                f"corpus registry at {self.path} is unreadable ({exc}); "
                "refusing to load an empty registry over it"
            ) from exc
        for name, payload in data.items():
            self._corpora[name] = Corpus(**payload)

    def _reload(self) -> None:
        """Replace in-memory state with the current on-disk state.

        Called right before a mutation so a long-lived process (e.g. the search
        server) doesn't clobber another writer's changes (e.g. a CLI index) with
        a stale in-memory copy. This was a real bug: migrate/index writes done
        by the CLI were silently reverted when the running server next saved.
        """
        self._corpora = {}
        self._load()

    def _on_disk_text(self) -> str | None:
        """Current file contents, or None if unreadable for any reason.

        None always means "write it": a missing, truncated or permission-denied
        file must never be mistaken for "already up to date".
        """
        try:
            return self.path.read_text()
        except OSError:
            return None

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: asdict(c) for name, c in self._corpora.items()}
        # Atomic write: serialize to a temp file in the same dir, fsync, then
        # os.replace (atomic on POSIX). A crashed/half-finished save can no
        # longer leave a truncated corpora.json for the next reader to choke on
        # — the root of the recurring "corpora vanished" losses. Mirrors
        # config._atomic_write_json, which this used to lack.
        text = json.dumps(payload, indent=2, sort_keys=True)
        # Skip a byte-identical rewrite. The registry is saved once per indexed
        # source so each journal record has a full recovery boundary, and a
        # no-op re-ingest walks every source without changing any of them --
        # which used to cost the whole file (2.2 MB on a real install) plus two
        # fsyncs per source, for zero new information. Comparing first is one
        # read of a file the OS already has cached.
        if self._on_disk_text() == text:
            return
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=self.path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
            # Persist the directory entry as well as the file contents. This is
            # the difference between atomic visibility and crash durability.
            dir_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    @staticmethod
    def validate_user_name(name: object) -> None:
        """Validate names accepted from CLI, HTTP, or imported metadata."""
        if not isinstance(name, str) or not _VALID_NAME.fullmatch(name):
            raise CorpusError(
                f"Invalid corpus name '{name}'. Use letters, digits, underscore, hyphen "
                "(1-64 chars, must start with a letter or digit)."
            )

    @staticmethod
    def validate_name(name: str) -> None:
        if name == EPHEMERAL_NAME or name.startswith(_INTERNAL_PREFIX):
            return
        CorpusRegistry.validate_user_name(name)

    def list(self) -> list[Corpus]:
        if self.in_memory:
            return sorted(
                (deepcopy(c) for c in self._corpora.values()), key=lambda c: c.name
            )
        with self.locks.registry_read():
            self._reload()
            return sorted(
                (deepcopy(c) for c in self._corpora.values()), key=lambda c: c.name
            )

    def get(self, name: str) -> Corpus:
        if self.in_memory:
            if name not in self._corpora:
                raise CorpusError(f"No such corpus: {name}")
            return deepcopy(self._corpora[name])
        # A deferred entry is the truth for this process until it lands; the
        # file only lags it. Readers in this process (the journal-retire check
        # that compares BM25 rows to chunk_count) must see the live one.
        pending = self._deferred.get(name)
        if pending is not None:
            return deepcopy(pending)
        with self.locks.registry_read():
            self._reload()
            if name not in self._corpora:
                raise CorpusError(f"No such corpus: {name}")
            return deepcopy(self._corpora[name])

    def has(self, name: str) -> bool:
        if self.in_memory:
            return name in self._corpora
        if self._deferred.get(name) is not None:
            return True
        with self.locks.registry_read():
            self._reload()
            return name in self._corpora

    @contextlib.contextmanager
    def deferred_saves(self, name: str):
        """Hold this corpus's upserts in memory and write corpora.json once.

        An index run upserts the registry after every source, and each save
        rewrites the whole file with two fsyncs: on a real install that was
        2.4 MB per source, 15% of everything an index run put on disk
        (2026-09-13). The caller holds the corpus write lock for the whole
        run, so nothing else can move this entry meanwhile, and a crash loses
        only hashes that recovery rebuilds from Qdrant anyway. Other corpora
        keep saving immediately. The pending entry lands on exit, exception
        or not, so partial progress is never thrown away.
        """
        if self.in_memory or name in self._deferred:
            yield
            return
        self._deferred[name] = None
        try:
            yield
        finally:
            pending = self._deferred.pop(name, None)
            if pending is not None:
                with self.locks.registry_write():
                    self._reload()
                    self._corpora[name] = pending
                    self._save()

    def upsert(self, c: Corpus) -> None:
        self.validate_name(c.name)
        if self.in_memory:
            self._corpora[c.name] = deepcopy(c)
            return
        if c.name in self._deferred:
            self._deferred[c.name] = deepcopy(c)
            self._corpora[c.name] = deepcopy(c)
            return
        # The lock spans reload -> modify -> replace. Atomic replace alone kept
        # readers from seeing torn JSON but still allowed two writers to reload
        # the same generation and silently clobber one another.
        with self.locks.registry_write():
            self._reload()
            self._corpora[c.name] = deepcopy(c)
            self._save()

    def delete(self, name: str) -> None:
        if self.in_memory:
            if name not in self._corpora:
                raise CorpusError(f"No such corpus: {name}")
            del self._corpora[name]
            return
        with self.locks.registry_write():
            self._reload()
            if name not in self._corpora:
                raise CorpusError(f"No such corpus: {name}")
            del self._corpora[name]
            self._save()
