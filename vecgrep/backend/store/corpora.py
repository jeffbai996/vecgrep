"""Named corpus registry.

A corpus is a logical grouping of documents stored in a single Qdrant
collection. Metadata pins the embedding backend + dimension so we
refuse to mix models within one corpus.
"""
from __future__ import annotations

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


class CorpusRegistry:
    def __init__(self, path: Path, locks: CorpusLocks | None = None,
                 *, in_memory: bool = False) -> None:
        self.path = path
        self.in_memory = in_memory
        self.locks = locks or CorpusLocks(path.parent / "locks")
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

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: asdict(c) for name, c in self._corpora.items()}
        # Atomic write: serialize to a temp file in the same dir, fsync, then
        # os.replace (atomic on POSIX). A crashed/half-finished save can no
        # longer leave a truncated corpora.json for the next reader to choke on
        # — the root of the recurring "corpora vanished" losses. Mirrors
        # config._atomic_write_json, which this used to lack.
        text = json.dumps(payload, indent=2, sort_keys=True)
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
    def validate_name(name: str) -> None:
        if name == EPHEMERAL_NAME or name.startswith(_INTERNAL_PREFIX):
            return
        if not _VALID_NAME.match(name):
            raise CorpusError(
                f"Invalid corpus name '{name}'. Use letters, digits, underscore, hyphen "
                "(1-64 chars, must start with a letter or digit)."
            )

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
        with self.locks.registry_read():
            self._reload()
            if name not in self._corpora:
                raise CorpusError(f"No such corpus: {name}")
            return deepcopy(self._corpora[name])

    def has(self, name: str) -> bool:
        if self.in_memory:
            return name in self._corpora
        with self.locks.registry_read():
            self._reload()
            return name in self._corpora

    def upsert(self, c: Corpus) -> None:
        self.validate_name(c.name)
        if self.in_memory:
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
