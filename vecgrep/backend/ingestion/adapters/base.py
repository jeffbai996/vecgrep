"""Adapter interface.

Adapters convert a source (file path or URL) into a stream of Documents.
A Document is a (source_id, text, metadata) bundle. Chunkers run on the
text afterward; adapters do not chunk.

To add an adapter (e.g., Claude/ChatGPT export, EPUB), subclass Adapter,
implement matches() and load(), then register it via register_adapter().
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


class AdapterError(RuntimeError):
    """Raised when a source can't be loaded."""


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
