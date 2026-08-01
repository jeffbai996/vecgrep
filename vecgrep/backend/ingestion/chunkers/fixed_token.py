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

        chunks: list[Chunk] = []
        step = self.tokens - self.overlap
        ordinal = 0
        char_cursor = 0
        for start_tok in range(0, len(ids), step):
            slice_ids = ids[start_tok : start_tok + self.tokens]
            if not slice_ids:
                break
            piece = self._enc.decode(slice_ids)
            stripped = piece.strip()
            if not stripped:
                continue
            char_start = text.find(stripped, char_cursor)
            if char_start == -1:
                # Decoded slice doesn't match exactly (BPE merge edge); fall
                # back to approximate offsets so context preview still works.
                char_start = char_cursor
            char_end = char_start + len(stripped)
            chunks.append(Chunk(text=stripped, start=char_start, end=char_end, index=ordinal))
            ordinal += 1
            # Resume the next search at THIS chunk's start, not past its end.
            # Chunks overlap by `overlap` tokens, so the next chunk's text
            # begins before this one ends; advancing the cursor beyond that
            # made text.find() miss the true position and fall back to the
            # approximate offset, so start/end no longer sliced back to the
            # chunk's own text.
            char_cursor = char_start
            if start_tok + self.tokens >= len(ids):
                break
        return chunks
