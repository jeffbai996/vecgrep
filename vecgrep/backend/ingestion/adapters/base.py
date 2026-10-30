"""Adapter interface.

Adapters convert a source (file path or URL) into a stream of Documents.
A Document is a (source_id, text, metadata) bundle. Chunkers run on the
text afterward; adapters do not chunk.

To add an adapter (e.g., Claude/ChatGPT export, EPUB), subclass Adapter,
implement matches() and load(), then register it via register_adapter().
"""
from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


class AdapterError(RuntimeError):
    """Raised when a source can't be loaded."""


_JSON_PARSE_CACHE: dict[tuple[str, int, int], object] = {}
_JSON_PARSE_LOCK = threading.Lock()
_JSON_CACHE_MAX = 32


def _get_cached_json_file(p: Path) -> object | None:
    try:
        st = p.stat()
        key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    except OSError:
        return None
    with _JSON_PARSE_LOCK:
        if key in _JSON_PARSE_CACHE:
            return _JSON_PARSE_CACHE[key]
    try:
        obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    with _JSON_PARSE_LOCK:
        if len(_JSON_PARSE_CACHE) >= _JSON_CACHE_MAX:
            _JSON_PARSE_CACHE.pop(next(iter(_JSON_PARSE_CACHE)), None)
        _JSON_PARSE_CACHE[key] = obj
    return obj


def try_load_json_file(p: Path) -> object | None:
    """Parse a JSON file for matches() sniffing — None on any read/parse failure.

    Shared by the chat-export adapters (ChatGPT, Claude), whose matches()
    bodies are otherwise byte-identical except for the sniff predicate.
    """
    return _get_cached_json_file(p)


def load_json_file(p: Path, source: str) -> object:
    """Parse a JSON file for load() — raises AdapterError on failure."""
    obj = _get_cached_json_file(p)
    if obj is not None:
        return obj
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
