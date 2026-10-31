"""Parser-backed code_symbol chunks: qualified names, exact spans, safe fallback."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from vecgrep.backend.ingestion.chunkers import CodeSymbolChunker, code_symbol

FIXTURES = Path(__file__).parent / "fixtures" / "code_symbols"


def chunks_for(name, **kw):
    text = (FIXTURES / name).read_text()
    return text, CodeSymbolChunker(**kw).chunk_doc(SimpleNamespace(text=text, source_id=f"/repo/{name}"))


def by_qualname(chunks):
    return {c.meta.get("qualname"): c for c in chunks if c.meta is not None and c.meta.get("qualname")}


def test_each_method_gets_its_own_chunk_with_its_qualified_name():
    _, chunks = chunks_for("orders.py")
    named = by_qualname(chunks)
    assert {"Importer.run", "Exporter.run", "Importer.Stats.run", "Importer._parse"} <= set(named)
    imp_run = named["Importer.run"]
    stats_run = named["Importer.Stats.run"]
    exp_run = named["Exporter.run"]
    assert imp_run.meta is not None
    assert stats_run.meta is not None
    assert exp_run.meta is not None
    assert imp_run.meta["parent"] == "Importer"
    assert stats_run.meta["parent"] == "Importer.Stats"
    assert exp_run.meta["kind"] == "method"


def test_identical_names_in_different_classes_stay_apart():
    _, chunks = chunks_for("orders.py")
    named = by_qualname(chunks)
    assert "json.loads" not in named["Importer.run"].text
    assert "json.dumps" in named["Exporter.run"].text
    imp_run = named["Importer.run"]
    exp_run = named["Exporter.run"]
    assert imp_run.meta is not None
    assert exp_run.meta is not None
    assert imp_run.meta["lines"] == "24-25"
    assert exp_run.meta["lines"] == "37-39"


def test_decorators_and_a_comment_directly_above_belong_to_the_definition():
    _unused_text, chunks = chunks_for("orders.py")
    named = by_qualname(chunks)
    assert named["Exporter.header"].text.lstrip().startswith("@staticmethod")
    assert named["Exporter.run"].text.lstrip().startswith("# Writes orders back out.")
    order = named["Order"]
    assert order.meta is not None
    assert order.meta["lines"] == "10-16"


def test_functions_stay_whole_and_list_what_they_define():
    _, chunks = chunks_for("orders.py")
    sync = by_qualname(chunks)["sync_orders"]
    assert sync.meta is not None
    assert sync.meta["kind"] == "async_function"
    assert sync.meta["nested"] == "sync_orders.push, sync_orders.retryable"
    assert "async def push" in sync.text


def test_search_text_names_the_symbol_and_the_imports_it_uses():
    _, chunks = chunks_for("orders.py")
    parse = by_qualname(chunks)["Importer._parse"]
    assert parse.search_text is not None
    assert parse.search_text.startswith("method Importer._parse (lines 27-29)\n")
    assert "imports: json from json" in parse.search_text
    assert parse.search_text.endswith(parse.text)


def test_offsets_address_the_text_and_no_content_is_lost():
    for name in ("orders.py", "broken.py", "api.ts", "view.tsx", "broken.ts"):
        text, chunks = chunks_for(name)
        covered = set()
        for c in chunks:
            assert text[c.start:c.end] == c.text
            covered.update(range(c.start, c.end))
        assert all(i in covered for i, ch in enumerate(text) if not ch.isspace()), name


def test_a_syntax_error_falls_back_to_the_heuristic_and_says_so():
    _unused_text, chunks = chunks_for("broken.py")
    assert chunks and all((c.meta or {}).get("parse_error") for c in chunks)
    assert "class StillHere" in "".join(c.text for c in chunks)


def test_parse_false_keeps_the_heuristic():
    _, chunks = chunks_for("orders.py", parse=False)
    assert not any(c.meta is not None and "qualname" in c.meta for c in chunks)


requires_tree_sitter = pytest.mark.skipif(
    not code_symbol.parser_available("typescript"), reason="tree-sitter is the optional [code] extra")


@requires_tree_sitter
def test_typescript_classes_methods_and_arrow_functions():
    _, chunks = chunks_for("api.ts")
    named = by_qualname(chunks)
    handle = named["OrderService.handle"]
    on_error = named["OrderService.onError"]
    audit_run = named["AuditService.run"]
    health = named["healthCheck"]
    internal = named["internalHelper"]
    router = named["createRouter"]
    handler = named["Handler"]
    assert handle.meta is not None
    assert on_error.meta is not None
    assert audit_run.meta is not None
    assert health.meta is not None
    assert internal.meta is not None
    assert router.meta is not None
    assert handler.meta is not None
    assert handle.meta["lines"] == "21-27"
    assert on_error.meta["kind"] == "method"
    assert audit_run.meta["parent"] == "AuditService"
    assert health.meta["kind"] == "async_function"
    assert internal.meta["kind"] == "function"
    assert router.meta["lines"] == "53-55"
    assert handler.meta["kind"] == "interface"
    assert handle.meta["nested"] == "OrderService.handle.remember"


@requires_tree_sitter
def test_typescript_decorator_and_doc_comment_open_the_class_span():
    _, chunks = chunks_for("api.ts")
    header = by_qualname(chunks)["OrderService"]
    assert header.meta is not None
    assert header.meta["lines"] == "16-37"
    assert header.meta["role"] == "header"
    assert header.text.startswith("/** Serves orders. */\n@Injectable()")


@requires_tree_sitter
def test_tsx_components():
    _, chunks = chunks_for("view.tsx")
    named = by_qualname(chunks)
    head = named["Header"]
    foot = named["Footer"]
    assert head.meta is not None
    assert foot.meta is not None
    assert head.meta["lines"] == "5-7"
    assert foot.meta["lines"] == "9-12"


def test_typescript_without_tree_sitter_uses_the_heuristic(monkeypatch):
    monkeypatch.setattr(code_symbol, "parser_available", lambda language: language == "python")
    _, chunks = chunks_for("api.ts")
    assert chunks and not any(c.meta is not None and "qualname" in c.meta for c in chunks)


def chunks_of(path, text):
    return CodeSymbolChunker().chunk_doc(SimpleNamespace(text=text, source_id=f"/repo/{path}"))


def assert_offsets_exact(text, chunks):
    for c in chunks:
        assert text[c.start:c.end] == c.text


def test_python_nested_past_the_recursion_limit_falls_back_instead_of_raising():
    text = "def total():\n    return " + " + ".join(["1"] * 3000) + "\n"
    chunks = chunks_of("generated.py", text)
    assert chunks and all((c.meta or {}).get("parse_error") for c in chunks)
    assert_offsets_exact(text, chunks)


@requires_tree_sitter
def test_typescript_nested_past_the_recursion_limit_falls_back_instead_of_raising():
    depth = 2000
    text = "".join(f"const f{i} = () => {{\n" for i in range(depth)) + "};\n" * depth
    chunks = chunks_of("bundle.ts", text)
    assert chunks and all((c.meta or {}).get("parse_error") for c in chunks)
    assert_offsets_exact(text, chunks)


# str.splitlines() also breaks on \x0b, \x0c, \x1c-\x1e, \x85, U+2028 and
# U+2029. ast breaks only on \r\n, \r and \n; tree-sitter only on \n.
@pytest.mark.parametrize("path, text", [
    ("form_feed.py", "x = 1\n\x0cdef f():\n    return 42\n"),
    ("separator.py", 'x = "a b"\ndef f():\n    return 42\n'),
    ("bare_cr.py", "x = 1\rdef f():\r    return 42\r"),
])
def test_python_symbol_lines_follow_the_parsers_line_breaks(path, text):
    chunks = chunks_of(path, text)
    f = by_qualname(chunks)["f"]
    assert f.meta is not None
    assert f.meta["lines"] == "2-3"
    assert f.text.lstrip("\x0c").startswith("def f():") and "return 42" in f.text
    assert_offsets_exact(text, chunks)


@requires_tree_sitter
@pytest.mark.parametrize("path, text, body", [
    ("separator.ts", 'const s = "a b";\nfunction f() {\n  return 1;\n}\n', "  return 1;\n}\n"),
    ("form_feed.ts", "const a = 1;\n\x0cfunction f() {\n  return 1;\n}\n", "  return 1;\n}\n"),
    ("bare_cr.ts", "function f() {\r  return 1;\r}\r\nclass A {\r m() {}\r}\r", "return 1;\r}\r\n"),
])
def test_typescript_symbol_lines_follow_the_parsers_line_breaks(path, text, body):
    chunks = chunks_of(path, text)
    assert by_qualname(chunks)["f"].text.endswith(body)
    assert_offsets_exact(text, chunks)


def test_a_byte_order_mark_does_not_stop_python_parsing():
    text = "﻿import helper\n\ndef f():\n    return helper.go()\n"
    chunks = chunks_of("bom.py", text)
    assert not any((c.meta or {}).get("parse_error") for c in chunks)
    f = by_qualname(chunks)["f"]
    assert f.meta is not None
    assert f.meta["lines"] == "3-4"
    assert chunks[0].start == 0 and chunks[0].text.startswith("﻿")
    assert_offsets_exact(text, chunks)
