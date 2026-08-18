"""Turn-window chunker — groups whole chat messages, never splits one.

Transcripts are recorded as `**speaker** · HH:MM` followed by the message
body. The sentence-window chunker cuts every 3 sentences regardless of who
is talking, so a chunk routinely starts mid-answer and ends mid-question,
and the speaker attached to a statement is often absent from its own chunk.

This groups by TURN instead: N messages per chunk with an overlap of one, so
a question and its answer stay together and every chunk begins with a
speaker line. Falls back to the sentence-window behaviour on text that has
no turn markers, so it is safe on a mixed corpus.
"""
from __future__ import annotations

import re

from .base import Chunk, Chunker
from .sentence_window import SentenceWindowChunker

# `**speaker** · 19:33` at the start of a line. The separator is U+00B7.
# Bot names carry a `[bot]` suffix and speakers can contain CJK, so the name
# class is deliberately permissive — anything up to the closing `**`.
_TURN_RE = re.compile(r"^\*\*(.+?)\*\*\s*·\s*(\d{1,2}:\d{2})", re.MULTILINE)


class TurnWindowChunker(Chunker):
    # Upper bound on a chunk. bge-m3 under Ollama accepts 8192 tokens; four
    # bot turns with code blocks blew past that on the first real build
    # ("the input length exceeds the context length"). 4000 chars is
    # ~1000-1500 tokens for English and stays inside every model in the
    # dimensions table with room for CJK, which tokenizes denser.
    def __init__(self, turns: int = 4, stride: int = 3, min_chars: int = 30,
                 max_chars: int = 4000) -> None:
        if stride <= 0 or turns <= 0 or stride > turns:
            raise ValueError("Need 0 < stride <= turns")
        self.turns = turns
        self.stride = stride
        self.min_chars = min_chars
        self.max_chars = max_chars
        # Used verbatim when a document carries no turn markers at all.
        self._fallback = SentenceWindowChunker(min_chars=min_chars)

    def _turn_spans(self, text: str) -> list[tuple[int, int]]:
        """(start, end) char offsets of each message, header included."""
        starts = [m.start() for m in _TURN_RE.finditer(text)]
        if not starts:
            return []
        bounds = starts + [len(text)]
        return [(bounds[i], bounds[i + 1]) for i in range(len(starts))]

    def chunk(self, text: str) -> list[Chunk]:
        spans = self._turn_spans(text)
        if len(spans) < 2:
            # Not a transcript (or a single-message one) — the sentence
            # windower is strictly better than emitting one giant chunk.
            return self._fallback.chunk(text)

        chunks: list[Chunk] = []
        idx = 0
        for i in range(0, len(spans), self.stride):
            group = spans[i:i + self.turns]
            if not group:
                break
            start, end = group[0][0], group[-1][1]
            body = text[start:end].strip()
            if len(body) < self.min_chars:
                continue
            # Re-derive offsets after stripping so highlight ranges stay true.
            lead = len(text[start:end]) - len(text[start:end].lstrip())
            if len(body) > self.max_chars:
                # Too big to embed as one piece: sentence-window this span
                # instead, re-based to document offsets. Loses the
                # whole-message guarantee for this window only.
                for sub in self._fallback.chunk(body):
                    if len(sub.text) > self.max_chars:
                        # a single run-on "sentence" (a code block, a log
                        # dump): hard-split it rather than ship it whole
                        for off in range(0, len(sub.text), self.max_chars):
                            piece = sub.text[off:off + self.max_chars]
                            chunks.append(Chunk(
                                text=piece, start=start + lead + sub.start + off,
                                end=start + lead + sub.start + off + len(piece),
                                index=idx, meta={"turns": len(group), "split": True}))
                            idx += 1
                        continue
                    chunks.append(Chunk(
                        text=sub.text, start=start + lead + sub.start,
                        end=start + lead + sub.end, index=idx,
                        meta={"turns": len(group), "split": True}))
                    idx += 1
            else:
                chunks.append(Chunk(
                    text=body,
                    start=start + lead,
                    end=start + lead + len(body),
                    index=idx,
                    meta={"turns": len(group)},
                ))
                idx += 1
            if i + self.turns >= len(spans):
                break  # last window already covered the tail
        return chunks or self._fallback.chunk(text)
