"""Symbol-boundary code chunker.

Sentence windows are wrong for code: a window happily slices a function in
half and welds the tail to its neighbor, so a hit shows arbitrary lines with
no idea what they belong to. This chunker cuts on symbol boundaries instead —
function, class, top-level block — and never splits a symbol across chunks
(with one escape hatch: a single symbol larger than `max_chars` is split at
internal blank lines rather than embedded as one oversized blob, and every
piece keeps the symbol's name in its metadata).

Boundary detection is heuristic, not a parser: a column-0 line matching the
language's declaration shapes starts a new block, and everything indented or
brace-nested below it belongs to that block. Formatted code satisfies this;
minified or deliberately perverse code degrades to coarse blocks, which is
still strictly better than mid-function windows.

Each chunk carries `meta = {"symbol": ..., "language": ...}` so a search hit
names what it found and where it lives without the reader opening the file.

Opt-in (chunker name "code_symbol"). `chunk_doc(doc)` uses the document's
source path to pick the language; plain `chunk(text)` falls back to the
generic block heuristics.
"""
from __future__ import annotations

import re

from .base import Chunk, Chunker

LANG_BY_EXT = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".rs": "rust", ".java": "java", ".kt": "kotlin",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".scala": "scala",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".css": "css", ".scss": "css", ".less": "css",
    ".sql": "sql", ".lua": "lua", ".r": "r", ".jl": "julia",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".ini": "config",
    ".json": "json", ".html": "html", ".vue": "vue", ".svelte": "svelte",
}

# Column-0 declaration shapes per language. The generic pattern backs up any
# language without a specific entry (and plain `chunk()` calls).
_DECL = {
    "python": re.compile(
        r"^(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)"),
    "javascript": re.compile(
        r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?"
        r"(?:function\s*\*?\s*|class\s+|const\s+|let\s+|var\s+)"
        r"([A-Za-z_$][A-Za-z0-9_$]*)"),
    "go": re.compile(r"^(?:func|type)\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)"),
    "rust": re.compile(
        r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?"
        r"(?:fn|struct|enum|trait|impl|mod|macro_rules!)\s*<?\s*"
        r"([A-Za-z_][A-Za-z0-9_]*)?"),
    "java": re.compile(
        r"^(?:public|private|protected|abstract|final|static|\s)*"
        r"(?:class|interface|enum|record)\s+([A-Za-z_][A-Za-z0-9_]*)"),
    "shell": re.compile(r"^(?:function\s+)?([A-Za-z_][A-Za-z0-9_-]*)\s*\(\)\s*\{"),
    "css": re.compile(r"^([^\s{][^{]{0,120})\{\s*$"),
}
_DECL["typescript"] = _DECL["javascript"]
_DECL["c"] = _DECL["cpp"] = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_*\s]*\s\**([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# Generic block starter: any non-indented, non-closing line. Used only to
# decide "could a new top-level thing start here"; coalescing keeps trivia
# from becoming confetti.
_GENERIC_START = re.compile(r"^[^\s})\]#/]")
# lines that continue the previous symbol even at column 0
_ATTACHED = re.compile(r"^(?:@|#\[|//|#|\*|/\*|'''|\"\"\"|\})")

TARGET_CHARS = 2200      # coalesce small neighbors up to roughly this
MAX_CHARS = 7000         # a single symbol past this splits at blank lines


def language_for_path(path: str) -> str | None:
    dot = path.rfind(".")
    return LANG_BY_EXT.get(path[dot:].lower()) if dot != -1 else None


class CodeSymbolChunker(Chunker):
    def chunk(self, text: str) -> list[Chunk]:
        return self._chunk(text, language=None)

    def chunk_doc(self, doc) -> list[Chunk]:  # duck-typed: needs .source_id/.text
        return self._chunk(doc.text, language=language_for_path(doc.source_id or ""))

    # ── mechanics ────────────────────────────────────────────────────
    def _chunk(self, text: str, language: str | None) -> list[Chunk]:
        if not text.strip():
            return []
        lines = text.splitlines(keepends=True)
        decl = _DECL.get(language or "")
        starts = self._block_starts(lines, decl)
        blocks = self._blocks(lines, starts)
        blocks = self._coalesce(blocks, decl)
        chunks: list[Chunk] = []
        for start_off, block_text in blocks:
            symbol = self._symbol_of(block_text, decl)
            for piece_off, piece in self._bounded(block_text):
                chunks.append(Chunk(
                    text=piece,
                    start=start_off + piece_off,
                    end=start_off + piece_off + len(piece),
                    index=len(chunks),
                    meta={k: v for k, v in (
                        ("symbol", symbol), ("language", language)) if v},
                ))
        return chunks

    def _block_starts(self, lines: list[str], decl: re.Pattern | None) -> list[int]:
        starts = [0]
        prev_blank = True
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                prev_blank = True
                continue
            at_col0 = not line[0].isspace()
            if i and at_col0:
                if decl is not None and decl.match(line):
                    starts.append(i)
                elif decl is None and prev_blank and _GENERIC_START.match(line):
                    starts.append(i)
            prev_blank = False
        return sorted(set(starts))

    def _blocks(self, lines: list[str], starts: list[int]) -> list[tuple[int, str]]:
        """(char_offset, text) per block. Decorator/comment lines directly above
        a declaration travel with it, not with the block they trail."""
        # pull attached prefix lines (decorators, doc comments) into the block below
        adjusted = []
        for s in starts:
            while s > 0 and lines[s - 1].strip() and _ATTACHED.match(lines[s - 1]):
                s -= 1
            adjusted.append(s)
        adjusted = sorted(set(adjusted))
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        out = []
        for i, s in enumerate(adjusted):
            e = adjusted[i + 1] if i + 1 < len(adjusted) else len(lines)
            block = "".join(lines[s:e])
            if block.strip():
                out.append((offsets[s], block))
        return out

    def _coalesce(self, blocks: list[tuple[int, str]],
                  decl: re.Pattern | None) -> list[tuple[int, str]]:
        """Merge runs of small NON-declaration blocks (imports, constants,
        one-liners) so trivia doesn't become one chunk per line. A block that
        declares a symbol always starts its own chunk — that boundary is the
        whole point of this chunker."""
        out: list[tuple[int, str]] = []
        for off, text in blocks:
            is_decl = decl is not None and any(
                decl.match(line) for line in text.splitlines()
                if line.strip() and not line[0].isspace()
                and not _ATTACHED.match(line))
            if (out and not is_decl
                    and len(out[-1][1]) + len(text) <= TARGET_CHARS):
                prev_off, prev_text = out[-1]
                out[-1] = (prev_off, prev_text + text)
            else:
                out.append((off, text))
        return out

    def _bounded(self, block: str):
        """Yield (offset, piece) — the whole block, or blank-line splits when a
        single symbol exceeds MAX_CHARS (the never-split rule's escape hatch)."""
        if len(block) <= MAX_CHARS:
            yield 0, block
            return
        pieces = re.split(r"(\n\s*\n)", block)
        acc, acc_off, cursor = "", 0, 0
        for part in pieces:
            if len(acc) + len(part) > MAX_CHARS and acc.strip():
                yield acc_off, acc
                acc, acc_off = "", cursor
            acc += part
            cursor += len(part)
        if acc.strip():
            yield acc_off, acc

    def _symbol_of(self, block: str, decl: re.Pattern | None) -> str | None:
        first_real: str | None = None
        for line in block.splitlines():
            if not line.strip() or _ATTACHED.match(line):
                continue
            if first_real is None:
                first_real = line
            if decl is not None and not line[0].isspace():
                m = decl.match(line)
                if m:
                    name = next((g for g in m.groups() if g), None)
                    if name:
                        return name.strip()
        if decl is not None or first_real is None:
            # known language, no declaration in the block (import/const trivia)
            return None
        # generic: first identifier-looking token of the first real line
        m = re.match(r"^[^\sA-Za-z_]*([A-Za-z_][A-Za-z0-9_.$-]{1,60})", first_real)
        return m.group(1) if m else None
