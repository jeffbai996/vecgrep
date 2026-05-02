"""Named corpus registry.

A corpus is a logical grouping of documents stored in a single Qdrant
collection. Metadata pins the embedding backend + dimension so we
refuse to mix models within one corpus.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

EPHEMERAL_NAME = "__ephemeral__"
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


class CorpusRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._corpora: dict[str, Corpus] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            data = {}
        for name, payload in data.items():
            self._corpora[name] = Corpus(**payload)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: asdict(c) for name, c in self._corpora.items()}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @staticmethod
    def validate_name(name: str) -> None:
        if name == EPHEMERAL_NAME:
            return
        if not _VALID_NAME.match(name):
            raise CorpusError(
                f"Invalid corpus name '{name}'. Use letters, digits, underscore, hyphen "
                "(1-64 chars, must start with a letter or digit)."
            )

    def list(self) -> list[Corpus]:
        return sorted(self._corpora.values(), key=lambda c: c.name)

    def get(self, name: str) -> Corpus:
        if name not in self._corpora:
            raise CorpusError(f"No such corpus: {name}")
        return self._corpora[name]

    def has(self, name: str) -> bool:
        return name in self._corpora

    def upsert(self, c: Corpus) -> None:
        self.validate_name(c.name)
        self._corpora[c.name] = c
        self._save()

    def delete(self, name: str) -> None:
        if name not in self._corpora:
            raise CorpusError(f"No such corpus: {name}")
        del self._corpora[name]
        self._save()
