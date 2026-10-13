"""Parser-backed code_symbol chunks: qualified names, exact spans, safe fallback."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from vecgrep.backend.ingestion.chunkers import CodeSymbolChunker
from vecgrep.backend.ingestion.chunkers import code_symbol

FIXTURES = Path(__file__).parent / "fixtures" / "code_symbols"


def chunks_for(name, **kw):
    text = (FIXTURES / name).read_text()
    return text, CodeSymbolChunker(**kw).chunk_doc(SimpleNamespace(text=text, source_id=f"/repo/{name}"))


def by_qualname(chunks):
    return {c.meta.get("qualname"): c for c in chunks if c.meta.get("qualname")}


def test_each_method_gets_its_own_chunk_with_its_qualified_name():
    _, chunks = chunks_for("orders.py")
    named = by_qualname(chunks)
    assert {"Importer.run", "Exporter.run", "Importer.Stats.run", "Importer._parse"} <= set(named)
    assert named["Importer.run"].meta["parent"] == "Importer"
    assert named["Importer.Stats.run"].meta["parent"] == "Importer.Stats"
    assert named["Exporter.run"].meta["kind"] == "method"


def test_identical_names_in_different_classes_stay_apart():
    _, chunks = chunks_for("orders.py")
    named = by_qualname(chunks)
    assert "json.loads" not in named["Importer.run"].text
    assert "json.dumps" in named["Exporter.run"].text
    assert named["Importer.run"].meta["lines"] == "24-25"
    assert named["Exporter.run"].meta["lines"] == "37-39"


def test_decorators_and_a_comment_directly_above_belong_to_the_definition():
    text, chunks = chunks_for("orders.py")
    named = by_qualname(chunks)
    assert named["Exporter.header"].text.lstrip().startswith("@staticmethod")
    assert named["Exporter.run"].text.lstrip().startswith("# Writes orders back out.")
    assert named["Order"].meta["lines"] == "10-16"


def test_functions_stay_whole_and_list_what_they_define():
    _, chunks = chunks_for("orders.py")
    sync = by_qualname(chunks)["sync_orders"]
    assert sync.meta["kind"] == "async_function"
    assert sync.meta["nested"] == "sync_orders.push, sync_orders.retryable"
    assert "async def push" in sync.text


def test_search_text_names_the_symbol_and_the_imports_it_uses():
    _, chunks = chunks_for("orders.py")
    parse = by_qualname(chunks)["Importer._parse"]
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
    text, chunks = chunks_for("broken.py")
    assert chunks and all(c.meta.get("parse_error") for c in chunks)
    assert "class StillHere" in "".join(c.text for c in chunks)


def test_parse_false_keeps_the_heuristic():
    _, chunks = chunks_for("orders.py", parse=False)
    assert not any("qualname" in c.meta for c in chunks)


requires_tree_sitter = pytest.mark.skipif(
    not code_symbol.parser_available("typescript"), reason="tree-sitter is the optional [code] extra")


@requires_tree_sitter
def test_typescript_classes_methods_and_arrow_functions():
    _, chunks = chunks_for("api.ts")
    named = by_qualname(chunks)
    assert named["OrderService.handle"].meta["lines"] == "21-27"
    assert named["OrderService.onError"].meta["kind"] == "method"
    assert named["AuditService.run"].meta["parent"] == "AuditService"
    assert named["healthCheck"].meta["kind"] == "async_function"
    assert named["internalHelper"].meta["kind"] == "function"
    assert named["createRouter"].meta["lines"] == "53-55"
    assert named["Handler"].meta["kind"] == "interface"
    assert named["OrderService.handle"].meta["nested"] == "OrderService.handle.remember"


@requires_tree_sitter
def test_typescript_decorator_and_doc_comment_open_the_class_span():
    _, chunks = chunks_for("api.ts")
    header = by_qualname(chunks)["OrderService"]
    assert header.meta["lines"] == "16-37"
    assert header.meta["role"] == "header"
    assert header.text.startswith("/** Serves orders. */\n@Injectable()")


@requires_tree_sitter
def test_tsx_components():
    _, chunks = chunks_for("view.tsx")
    named = by_qualname(chunks)
    assert named["Header"].meta["lines"] == "5-7"
    assert named["Footer"].meta["lines"] == "9-12"


def test_typescript_without_tree_sitter_uses_the_heuristic(monkeypatch):
    monkeypatch.setattr(code_symbol, "parser_available", lambda language: language == "python")
    _, chunks = chunks_for("api.ts")
    assert chunks and not any("qualname" in c.meta for c in chunks)
