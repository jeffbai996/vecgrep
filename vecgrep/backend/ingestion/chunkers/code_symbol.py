"""Symbol-boundary code chunker.

Sentence windows are wrong for code: a window happily slices a function in
half and welds the tail to its neighbor, so a hit shows arbitrary lines with
no idea what they belong to. This chunker cuts on symbol boundaries instead —
function, class, top-level block — and never splits a symbol across chunks
(with one escape hatch: a single symbol larger than `max_chars` is split at
internal blank lines rather than embedded as one oversized blob, and every
piece keeps the symbol's name in its metadata).

Python and TypeScript are parsed (`code_parse.py`): every class member and
nested class gets its own chunk, and each chunk names its qualified name
(`Outer.Inner.method`), kind, parent and exact line span. Functions stay whole;
the definitions nested inside them are listed in the chunk's metadata. The
chunk's `search_text` puts that name and the imports it uses in front of the
code, so embedding and BM25 can find `Outer.run` while the stored text and its
offsets stay exactly the source. TypeScript needs the optional tree-sitter
dependency; a file that fails to parse is chunked by the heuristic below, so
no content is lost, and its chunks say `parse_error`.

Other languages, and Python or TypeScript without a parser, use the heuristic:
a column-0 line matching the language's declaration shapes starts a new block,
and everything indented or brace-nested below it belongs to that block.

Each chunk carries `meta = {"symbol": ..., "language": ...}` so a search hit
names what it found and where it lives without the reader opening the file.

Opt-in (chunker name "code_symbol"). `chunk_doc(doc)` uses the document's
source path to pick the language; plain `chunk(text)` falls back to the
generic block heuristics.
"""
from __future__ import annotations

import re

from .base import Chunk, Chunker
from .code_parse import ParsedFile, Symbol, parse_symbols, used_imports

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


PARSED_LANGUAGES = ("python", "typescript")


def parser_available(language: str | None) -> bool:
    if language == "python":
        return True
    if language == "typescript":
        try:
            import tree_sitter  # noqa: F401
            import tree_sitter_typescript  # noqa: F401
        except ImportError:
            return False
        return True
    return False


class CodeSymbolChunker(Chunker):
    def __init__(self, parse: bool = True) -> None:
        # parse=False keeps the heuristic everywhere (benchmarks, comparisons).
        self.parse = parse

    def chunk(self, text: str) -> list[Chunk]:
        return self._chunk(text, language=None)

    def chunk_doc(self, doc) -> list[Chunk]:  # duck-typed: needs .source_id/.text
        path = doc.source_id or ""
        language = language_for_path(path)
        if self.parse and language in PARSED_LANGUAGES and parser_available(language):
            parsed = parse_symbols(doc.text, language, path)
            if parsed is None:
                chunks = self._chunk(doc.text, language)
                for c in chunks:
                    c.meta = {**(c.meta or {}), "parse_error": True}
                return chunks
            if parsed.symbols:
                return self._parsed_chunks(doc.text, parsed)
        return self._chunk(doc.text, language)

    # ── parser-backed chunks ─────────────────────────────────────────
    def _parsed_chunks(self, text: str, parsed: ParsedFile) -> list[Chunk]:
        # The parser's own line split: its line numbers index this table.
        lines = parsed.lines
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        segments: list[tuple[int, int, Symbol | None, str]] = []   # (first, last+1, symbol, role)

        def place(sym: Symbol, first: int, stop: int) -> None:
            members = sorted(sym.children, key=lambda m: m.start_line) if sym.kind == "class" else []
            if not members:
                segments.append((first, stop, sym, "symbol"))
                return
            head_stop = members[0].start_line - 1
            if head_stop > first:
                segments.append((first, head_stop, sym, "header"))
            for i, member in enumerate(members):
                member_stop = members[i + 1].start_line - 1 if i + 1 < len(members) else stop
                place(member, max(member.start_line - 1, first), member_stop)

        cursor = 0
        for sym in sorted(parsed.symbols, key=lambda s: s.start_line):
            first = max(sym.start_line - 1, cursor)
            if first > cursor:
                segments.append((cursor, first, None, "module"))
            place(sym, first, sym.end_line)
            cursor = max(cursor, sym.end_line)
        if cursor < len(lines):
            segments.append((cursor, len(lines), None, "module"))

        chunks: list[Chunk] = []
        for first, stop, sym, role in segments:
            body = "".join(lines[first:stop])
            if not body.strip():
                continue
            meta: dict = {"language": parsed.language, "parser": parsed.parser}
            header = ""
            if sym is not None:
                nested = [d.qualname for d in sym.walk() if d is not sym] if role == "symbol" else []
                meta.update({"symbol": sym.name, "qualname": sym.qualname, "kind": sym.kind,
                             "lines": f"{sym.start_line}-{sym.end_line}"})
                if sym.parent:
                    meta["parent"] = sym.parent
                if role == "header":
                    meta["role"] = "header"
                if nested:
                    meta["nested"] = ", ".join(nested)
                header = f"{sym.kind} {sym.qualname} (lines {sym.start_line}-{sym.end_line})\n"
                if nested:
                    header += f"defines: {', '.join(nested)}\n"
            uses = used_imports(body, parsed.imports)
            if uses:
                header += "imports: " + ", ".join(f"{n} from {parsed.imports[n]}" for n in uses) + "\n"
            for piece_off, piece in self._bounded(body):
                start = offsets[first] + piece_off
                chunks.append(Chunk(
                    text=piece, start=start, end=start + len(piece), index=len(chunks),
                    meta=dict(meta), search_text=(header + piece) if header else None,
                ))
        return chunks

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
            if i and at_col0 and (
                decl is not None and decl.match(line) or decl is None and prev_blank and _GENERIC_START.match(line)
            ):
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

    @staticmethod
    def _bounded_lines(base_off: int, text: str):
        if len(text) <= MAX_CHARS:
            yield base_off, text
            return
        line_acc, line_off, pos = "", base_off, base_off
        for line in text.splitlines(keepends=True):
            if len(line_acc) + len(line) > MAX_CHARS and line_acc.strip():
                yield line_off, line_acc
                line_acc, line_off = "", pos
            if len(line) > MAX_CHARS:
                for off in range(0, len(line), MAX_CHARS):
                    yield pos + off, line[off : off + MAX_CHARS]
                pos += len(line)
                line_off = pos
            else:
                line_acc += line
                pos += len(line)
        if line_acc.strip():
            yield line_off, line_acc

    def _bounded(self, block: str):
        """Yield (offset, piece) — the whole block, or blank-line / line splits when a
        single symbol exceeds MAX_CHARS (the never-split rule's escape hatch)."""
        if len(block) <= MAX_CHARS:
            yield 0, block
            return
        pieces = re.split(r"(\n\s*\n)", block)
        acc, acc_off, cursor = "", 0, 0
        for part in pieces:
            if len(acc) + len(part) > MAX_CHARS and acc.strip():
                yield from self._bounded_lines(acc_off, acc)
                acc, acc_off = "", cursor
            acc += part
            cursor += len(part)
        if acc.strip():
            yield from self._bounded_lines(acc_off, acc)

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
