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


class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        ...
