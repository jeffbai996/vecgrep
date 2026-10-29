"""Fixed-token chunker (tiktoken-based).

Slices text into N-token windows with optional overlap. Tokens are
counted with cl100k_base — accurate for OpenAI models, close enough
for nomic. Use when sentence boundaries are unreliable (code, logs).
"""
from __future__ import annotations

import tiktoken

from .base import Chunk, Chunker


class FixedTokenChunker(Chunker):
    def __init__(self, tokens: int = 256, overlap: int = 32, encoding: str = "cl100k_base") -> None:
        if overlap >= tokens:
            raise ValueError("overlap must be smaller than tokens")
        self.tokens = tokens
        self.overlap = overlap
        self._enc = tiktoken.get_encoding(encoding)

    def chunk(self, text: str) -> list[Chunk]:
        if not text.strip():
            return []
        ids = self._enc.encode(text)
        if not ids:
            return []

        token_offsets = [0]
        for token in ids:
            token_offsets.append(token_offsets[-1] + len(self._enc.decode_single_token_bytes(token)))
        import bisect
        char_offsets = [0]
        for char in text:
            char_offsets.append(char_offsets[-1] + len(char.encode("utf-8")))
        chunks: list[Chunk] = []
        step = self.tokens - self.overlap
        previous = None
        for token_start in range(0, len(ids), step):
            token_end = min(token_start + self.tokens, len(ids))
            start = bisect.bisect_right(char_offsets, token_offsets[token_start]) - 1
            end = bisect.bisect_left(char_offsets, token_offsets[token_end])
            span = text[start:end]
            piece = span.strip()
            start += len(span) - len(span.lstrip())
            end = start + len(piece)
            if piece and (start, end) != previous:
                chunks.append(Chunk(text=piece, start=start, end=end, index=len(chunks)))
                previous = (start, end)
            if token_end == len(ids):
                break
        return chunks
