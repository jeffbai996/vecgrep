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
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])\s*|\n{2,}")


class SentenceWindowChunker(Chunker):
    def __init__(
        self,
        window: int = 3,
        stride: int = 2,
        min_chars: int = 30,
        max_chars: int = 4000,
    ) -> None:
        if stride <= 0 or window <= 0 or stride > window:
            raise ValueError("Need 0 < stride <= window")
        self.window = window
        self.stride = stride
        self.min_chars = min_chars
        self.max_chars = max_chars

    @staticmethod
    def _split_span(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        cur = start
        while cur < end:
            if end - cur <= max_chars:
                spans.append((cur, end))
                break
            target_end = cur + max_chars
            # 1. Try newline boundary
            nl = text.rfind("\n", cur, target_end)
            if nl != -1 and nl > cur:
                split_pt = nl + 1
            else:
                # 2. Try whitespace boundary
                ws = -1
                for idx in range(target_end, cur, -1):
                    if text[idx - 1].isspace():
                        ws = idx
                        break
                if ws != -1 and ws > cur:
                    split_pt = ws
                else:
                    # 3. Character boundary
                    split_pt = target_end
            spans.append((cur, split_pt))
            cur = split_pt
        return spans

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
            if self.max_chars > 0 and (end - start) > self.max_chars:
                for sub_start, sub_end in self._split_span(text, start, end, self.max_chars):
                    sub_str = text[sub_start:sub_end]
                    sents.append((sub_str, sub_start, sub_end))
            else:
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
            end_idx = min(i + self.window, len(sents))
            while end_idx < len(sents) and window[-1][2] - window[0][1] < self.min_chars:
                end_idx += 1
                window = sents[i:end_idx]
            if self.max_chars > 0:
                while end_idx > i + 1 and sents[end_idx - 1][2] - window[0][1] > self.max_chars:
                    end_idx -= 1
                    window = sents[i:end_idx]
            joined = text[window[0][1]:window[-1][2]]
            start = window[0][1]
            end = window[-1][2]
            if self.max_chars > 0 and len(joined) > self.max_chars:
                for sub_start, sub_end in self._split_span(text, start, end, self.max_chars):
                    sub_text = text[sub_start:sub_end]
                    if len(sub_text) >= self.min_chars or end_idx >= len(sents):
                        chunks.append(Chunk(text=sub_text, start=sub_start, end=sub_end, index=ordinal))
                        ordinal += 1
            elif len(joined) >= self.min_chars or end_idx >= len(sents):
                chunks.append(Chunk(text=joined, start=start, end=end, index=ordinal))
                ordinal += 1
                if len(joined) < self.min_chars:
                    break
            step = min(self.stride, max(1, end_idx - i))
            i += step
        return chunks
