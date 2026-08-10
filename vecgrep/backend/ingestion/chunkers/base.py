from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    # Char offsets into the original document — used to rebuild surrounding
    # context at search time without re-chunking.
    start: int
    end: int
    # Optional ordinal within the source doc, set by the chunker.
    index: int = 0
    # Optional per-chunk metadata contributed by the chunker (e.g. the
    # code_symbol chunker's symbol name + language). Merged into the stored
    # payload metadata at index time.
    meta: dict | None = None


class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        ...
