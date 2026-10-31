"""Parse Python and TypeScript source into a tree of named definitions.

Python uses the standard library's `ast`. TypeScript and TSX use tree-sitter,
an optional dependency (`pip install "vecgrep[code]"`); without it, or for any
other language, `parse_symbols` returns None and the caller keeps its
heuristic chunking. A file that does not parse also returns None, so a syntax
error never costs content: the caller falls back rather than dropping it.

Spans are whole lines. A definition's span starts at its first decorator, or
at a comment block directly above it, and ends at the last line of its body.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tree_sitter


@dataclass
class Symbol:
    name: str
    qualname: str
    kind: str                 # class, function, async_function, method, interface, enum, type
    start_line: int           # 1-based, inclusive, decorators and leading comments included
    end_line: int             # 1-based, inclusive
    parent: str | None = None
    exported: bool = False
    children: list[Symbol] = field(default_factory=list)

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass
class ParsedFile:
    language: str
    parser: str
    symbols: list[Symbol]                # top level, in source order
    imports: dict[str, str]              # bound name -> module
    lines: list[str]                     # the text split where the parser counts a line, endings kept


def parse_symbols(text: str, language: str | None, path: str = "") -> ParsedFile | None:
    # Both walks recurse once or twice per nesting level, so generated or
    # bundled code nests past Python's recursion limit. That is a file the
    # parser cannot handle, not a reason to stop indexing.
    try:
        if language == "python":
            return _parse_python(text)
        if language == "typescript":
            return _parse_typescript(text, tsx=path.lower().endswith(".tsx"))
    except RecursionError:
        return None
    return None


def _split_lines(text: str, newline: re.Pattern) -> list[str]:
    """Lines with their endings, broken only where `newline` matches.

    str.splitlines() also breaks on \\x0b, \\x0c, \\x1c-\\x1e, \\x85, U+2028 and
    U+2029. A parser that does not count those would have its line numbers
    index the wrong entries, so each parser's lines are split its own way.
    """
    lines: list[str] = []
    start = 0
    for match in newline.finditer(text):
        lines.append(text[start:match.end()])
        start = match.end()
    if start < len(text):
        lines.append(text[start:])
    return lines


# ── Python ────────────────────────────────────────────────────────────────
_PY_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_PY_NEWLINE = re.compile(r"\r\n|\r|\n")


def _parse_python(text: str) -> ParsedFile | None:
    try:
        # ast.parse drops a byte order mark from bytes but refuses one in a
        # str. It is one character on line 1, so line numbers do not move.
        tree = ast.parse(text.removeprefix("﻿"))
    except (SyntaxError, ValueError):
        return None
    lines = _split_lines(text, _PY_NEWLINE)
    imports: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[(alias.asname or alias.name).split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    imports[alias.asname or alias.name] = f"{node.module or ''}.{alias.name}".lstrip(".")
    symbols = [_py_symbol(n, None, lines, in_class=False) for n in tree.body if isinstance(n, _PY_DEFS)]
    return ParsedFile("python", "ast", symbols, imports, lines)


def _py_symbol(node, parent: str | None, lines: list[str], *, in_class: bool) -> Symbol:
    qual = f"{parent}.{node.name}" if parent else node.name
    if isinstance(node, ast.ClassDef):
        kind = "class"
    elif in_class:
        kind = "method"
    else:
        kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    start = _leading_comments(lines, start, "#")
    sym = Symbol(node.name, qual, kind, start, node.end_lineno or node.lineno, parent)
    for child in ast.iter_child_nodes(node):
        sym.children.extend(_py_nested(child, qual, lines, in_class=isinstance(node, ast.ClassDef)))
    return sym


def _py_nested(node, parent: str, lines: list[str], *, in_class: bool) -> list[Symbol]:
    """Definitions anywhere inside a body (under if/try/with too), not inside other defs."""
    if isinstance(node, _PY_DEFS):
        return [_py_symbol(node, parent, lines, in_class=in_class)]
    found: list[Symbol] = []
    for child in ast.iter_child_nodes(node):
        found.extend(_py_nested(child, parent, lines, in_class=in_class))
    return found


def _leading_comments(lines: list[str], start: int, marker: str) -> int:
    """Extend a 1-based start line up over comment lines directly above it."""
    while start > 1 and lines[start - 2].lstrip("﻿").strip().startswith(marker):
        start -= 1
    return start


# ── TypeScript (tree-sitter) ──────────────────────────────────────────────
_TS_LANGUAGES: dict[bool, tree_sitter.Language] = {}


def _ts_parser(tsx: bool):
    try:
        import tree_sitter
        import tree_sitter_typescript
    except ImportError:
        return None
    if tsx not in _TS_LANGUAGES:
        raw = tree_sitter_typescript.language_tsx() if tsx else tree_sitter_typescript.language_typescript()
        _TS_LANGUAGES[tsx] = tree_sitter.Language(raw)
    return tree_sitter.Parser(_TS_LANGUAGES[tsx])


_TS_CLASS = {"class_declaration", "abstract_class_declaration", "class"}
_TS_FUNC = {"function_declaration", "generator_function_declaration"}
_TS_SIMPLE = {"interface_declaration": "interface", "enum_declaration": "enum",
              "type_alias_declaration": "type"}
_TS_FUNC_VALUE = {"arrow_function", "function_expression", "function", "generator_function"}
_TS_NEWLINE = re.compile(r"\n")      # tree-sitter rows advance on \n alone


def _parse_typescript(text: str, *, tsx: bool) -> ParsedFile | None:
    parser = _ts_parser(tsx)
    if parser is None:
        return None
    source = text.encode("utf-8")
    tree = parser.parse(source)
    if tree.root_node.has_error:
        return None
    lines = _split_lines(text, _TS_NEWLINE)
    imports: dict[str, str] = {}
    symbols: list[Symbol] = []
    for node in tree.root_node.named_children:
        if node.type == "import_statement":
            imports.update(_ts_imports(node))
            continue
        symbols.extend(_ts_symbols(node, None, lines, exported=False))
    return ParsedFile("typescript", "tree-sitter", symbols, imports, lines)


def _name(node) -> str | None:
    child = node.child_by_field_name("name")
    return child.text.decode("utf-8") if child is not None else None


def _span(node, lines: list[str], anchor=None) -> tuple[int, int]:
    """1-based whole-line span; decorators and a JSDoc/line-comment block above count."""
    start_node = anchor or node
    start = start_node.start_point[0] + 1
    prev = start_node.prev_named_sibling
    while prev is not None and prev.type == "decorator":
        start = prev.start_point[0] + 1
        prev = prev.prev_named_sibling
    while prev is not None and prev.type == "comment" and prev.end_point[0] + 1 >= start - 1:
        start = prev.start_point[0] + 1
        prev = prev.prev_named_sibling
    return start, node.end_point[0] + 1


def _ts_symbols(node, parent: str | None, lines: list[str], *, exported: bool, anchor=None) -> list[Symbol]:
    kind = node.type
    if kind == "export_statement":
        found: list[Symbol] = []
        for child in node.named_children:
            if child.type in ("decorator", "comment"):
                continue
            found.extend(_ts_symbols(child, parent, lines, exported=True, anchor=node))
            # `export default <expression>` names nothing; keep it findable
        if not found and any(c.type in _TS_FUNC_VALUE | _TS_CLASS for c in node.named_children):
            start, end = _span(node, lines)
            found.append(Symbol("default", f"{parent}.default" if parent else "default", "function",
                                start, end, parent, True))
        return found
    if kind in _TS_CLASS:
        name = _name(node) or "default"
        qual = f"{parent}.{name}" if parent else name
        start, end = _span(node, lines, anchor)
        sym = Symbol(name, qual, "class", start, end, parent, exported)
        body = node.child_by_field_name("body")
        if body is not None:
            for member in body.named_children:
                sym.children.extend(_ts_member(member, qual, lines))
        return [sym]
    if kind in _TS_FUNC:
        name = _name(node) or "default"
        qual = f"{parent}.{name}" if parent else name
        start, end = _span(node, lines, anchor)
        is_async = any(c.type == "async" for c in node.children)
        sym = Symbol(name, qual, "async_function" if is_async else "function", start, end, parent, exported)
        sym.children.extend(_ts_inner(node.child_by_field_name("body"), qual, lines))
        return [sym]
    if kind in _TS_SIMPLE:
        name = _name(node) or "anonymous"
        qual = f"{parent}.{name}" if parent else name
        start, end = _span(node, lines, anchor)
        return [Symbol(name, qual, _TS_SIMPLE[kind], start, end, parent, exported)]
    if kind in ("lexical_declaration", "variable_declaration"):
        found = []
        for decl in node.named_children:
            if decl.type != "variable_declarator":
                continue
            value = decl.child_by_field_name("value")
            if value is None or value.type not in _TS_FUNC_VALUE | _TS_CLASS:
                continue
            name = _name(decl) or "anonymous"
            qual = f"{parent}.{name}" if parent else name
            start, end = _span(node, lines, anchor)
            is_async = any(c.type == "async" for c in value.children)
            sym_kind = "class" if value.type in _TS_CLASS else ("async_function" if is_async else "function")
            sym = Symbol(name, qual, sym_kind, start, end, parent, exported)
            sym.children.extend(_ts_inner(value.child_by_field_name("body"), qual, lines))
            found.append(sym)
        return found
    return []


def _ts_member(member, parent: str, lines: list[str]) -> list[Symbol]:
    if member.type == "method_definition":
        name = _name(member) or "anonymous"
        start, end = _span(member, lines)
        sym = Symbol(name, f"{parent}.{name}", "method", start, end, parent)
        sym.children.extend(_ts_inner(member.child_by_field_name("body"), sym.qualname, lines))
        return [sym]
    if member.type in ("public_field_definition", "field_definition"):
        value = member.child_by_field_name("value")
        if value is not None and value.type in _TS_FUNC_VALUE:
            name = _name(member) or "anonymous"
            start, end = _span(member, lines)
            return [Symbol(name, f"{parent}.{name}", "method", start, end, parent)]
    return []


def _ts_inner(body, parent: str, lines: list[str]) -> list[Symbol]:
    """Named definitions nested inside a function or method body."""
    if body is None:
        return []
    found: list[Symbol] = []
    for child in body.named_children:
        found.extend(_ts_symbols(child, parent, lines, exported=False))
    return found


def _ts_imports(node) -> dict[str, str]:
    source = node.child_by_field_name("source")
    module = source.text.decode("utf-8").strip("'\"") if source is not None else ""
    names: dict[str, str] = {}
    for ident in _descendants(node, {"identifier"}):
        if ident.parent is not None and ident.parent.type == "import_specifier":
            alias = ident.parent.child_by_field_name("alias")
            if alias is not None and alias.id != ident.id:
                continue
        names[ident.text.decode("utf-8")] = module
    return names


def _descendants(node, types: set[str]):
    for child in node.named_children:
        if child.type in types:
            yield child
        yield from _descendants(child, types)


IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")


def used_imports(text: str, imports: dict[str, str], limit: int = 8) -> list[str]:
    """Imported names this chunk mentions, in first-use order."""
    seen: list[str] = []
    for token in IDENT.findall(text):
        if token in imports and token not in seen:
            seen.append(token)
            if len(seen) == limit:
                break
    return seen
