"""Markdown-section chunker.

Sentence-window chunking that respects markdown heading boundaries: a window
never spans a heading line (# .. ######), so a chunk stays inside one section
and doesn't blend context across topics (e.g. a "Sensors" sentence and an
"Accuracy" sentence from different sections ending up in the same window). The
heading line travels with the section it introduces, giving each chunk its
topical context.

Opt-in (chunker name "markdown_section"); the default stays sentence_window.
With no headings present it degrades to plain sentence-window over the whole doc.
"""
from __future__ import annotations

import re

from .base import Chunk, Chunker
from .sentence_window import SentenceWindowChunker

# A markdown ATX heading line: 1-6 '#', then a space/tab, at line start.
_HEADER = re.compile(r"^#{1,6}[ \t]", re.MULTILINE)


class MarkdownSectionChunker(Chunker):
    def __init__(self, window: int = 3, stride: int = 2, min_chars: int = 30) -> None:
        # Reuse the sentence-window logic; we only add section boundaries.
        self._inner = SentenceWindowChunker(window, stride, min_chars)

    def _sections(self, text: str) -> list[tuple[str, int]]:
        """(section_text, char_offset) per heading-delimited section. The heading
        starts its section; text before the first heading is its own section."""
        starts = sorted({0, *(m.start() for m in _HEADER.finditer(text))})
        out: list[tuple[str, int]] = []
        for i, s in enumerate(starts):
            e = starts[i + 1] if i + 1 < len(starts) else len(text)
            if e > s:
                out.append((text[s:e], s))
        return out

    def chunk(self, text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        ordinal = 0
        for sec_text, off in self._sections(text):
            for c in self._inner.chunk(sec_text):
                # offset the section-relative spans back to the original doc
                chunks.append(Chunk(text=c.text, start=c.start + off,
                                    end=c.end + off, index=ordinal))
                ordinal += 1
        return chunks
