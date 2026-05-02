"""Sentence-window chunker.

Splits text into sentences, then groups them into overlapping windows.
Default window=3 sentences, stride=2 (1 sentence overlap). Keeps char
offsets so context can be rebuilt at search time.
"""
from __future__ import annotations

import re

from .base import Chunk, Chunker

# Cheap sentence splitter. Handles ., !, ? and Chinese 。！？ — good enough
# for general docs without spaCy. We deliberately don't pull in nltk.
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n{2,}")


class SentenceWindowChunker(Chunker):
    def __init__(self, window: int = 3, stride: int = 2, min_chars: int = 30) -> None:
        if stride <= 0 or window <= 0 or stride > window:
            raise ValueError("Need 0 < stride <= window")
        self.window = window
        self.stride = stride
        self.min_chars = min_chars

    def _split_sentences(self, text: str) -> list[tuple[str, int, int]]:
        sents: list[tuple[str, int, int]] = []
        cursor = 0
        for piece in _SENT_SPLIT.split(text):
            if not piece:
                continue
            idx = text.find(piece, cursor)
            if idx == -1:
                idx = cursor
            stripped = piece.strip()
            if not stripped:
                cursor = idx + len(piece)
                continue
            start = idx + (len(piece) - len(piece.lstrip()))
            end = start + len(stripped)
            sents.append((stripped, start, end))
            cursor = end
        return sents

    def chunk(self, text: str) -> list[Chunk]:
        sents = self._split_sentences(text)
        if not sents:
            return []

        chunks: list[Chunk] = []
        i = 0
        ordinal = 0
        while i < len(sents):
            window = sents[i : i + self.window]
            if not window:
                break
            joined = " ".join(s[0] for s in window)
            start = window[0][1]
            end = window[-1][2]
            if len(joined) >= self.min_chars or i + self.window >= len(sents):
                chunks.append(Chunk(text=joined, start=start, end=end, index=ordinal))
                ordinal += 1
            i += self.stride
        return chunks
