"""Adapter interface.

Adapters convert a source (file path or URL) into a stream of Documents.
A Document is a (source_id, text, metadata) bundle. Chunkers run on the
text afterward; adapters do not chunk.

To add an adapter (e.g., Claude/ChatGPT export, EPUB), subclass Adapter,
implement matches() and load(), then register it via register_adapter().
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


class AdapterError(RuntimeError):
    """Raised when a source can't be loaded."""


def try_load_json_file(p: Path) -> object | None:
    """Parse a JSON file for matches() sniffing — None on any read/parse failure.

    Shared by the chat-export adapters (ChatGPT, Claude), whose matches()
    bodies are otherwise byte-identical except for the sniff predicate.
    """
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None


def load_json_file(p: Path, source: str) -> object:
    """Parse a JSON file for load() — raises AdapterError on failure."""
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError) as e:
        raise AdapterError(f"Could not parse {source}: {e}") from e


@dataclass
class Document:
    source_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    # Unix epoch seconds for the document's own time (authored/saved date),
    # when an adapter can determine it. Drives optional recency decay at
    # search time. None when no date is discoverable.
    timestamp: float | None = None


class Adapter(ABC):
    name: str

    @classmethod
    @abstractmethod
    def matches(cls, source: str) -> bool:
        """Return True if this adapter can handle the given source string."""

    @abstractmethod
    def load(self, source: str) -> Iterator[Document]:
        ...
