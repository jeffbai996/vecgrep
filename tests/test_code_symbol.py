"""code_symbol chunker: symbol boundaries, never-split, metadata."""
from types import SimpleNamespace

from vecgrep.backend.ingestion.chunkers import CodeSymbolChunker
from vecgrep.backend.ingestion.chunkers.code_symbol import language_for_path

PY = '''import os
import sys

CONST = 1


@decorator
def alpha(x):
    if x:
        return os.path.join("a", "b")
    return sys.argv


class Beta:
    """doc"""

    def method(self):
        return CONST
'''


def _doc(text, path):
    return SimpleNamespace(text=text, source_id=path)


def test_python_symbols_and_metadata():
    chunks = CodeSymbolChunker().chunk_doc(_doc(PY, "/repo/app.py"))
    symbols = [c.meta.get("symbol") for c in chunks]
    assert "alpha" in symbols
    assert "Beta" in symbols
    assert all(c.meta.get("language") == "python" for c in chunks)


def test_function_bodies_never_split():
    chunks = CodeSymbolChunker().chunk_doc(_doc(PY, "/repo/app.py"))
    alpha = next(c for c in chunks if c.meta.get("symbol") == "alpha")
    assert "def alpha" in alpha.text
    assert 'return sys.argv' in alpha.text          # whole body in one chunk
    assert "@decorator" in alpha.text               # decorator travels with it


def test_offsets_reconstruct_source():
    chunks = CodeSymbolChunker().chunk_doc(_doc(PY, "/repo/app.py"))
    for c in chunks:
        assert PY[c.start:c.end] == c.text


def test_javascript_decls():
    js = 'export function widget(a) {\n  return a + 1\n}\n\nconst helper = () => {\n  return 2\n}\n'
    chunks = CodeSymbolChunker().chunk_doc(_doc(js, "/repo/ui.js"))
    symbols = {c.meta.get("symbol") for c in chunks}
    assert "widget" in symbols


def test_oversized_symbol_splits_but_keeps_name():
    body = "def giant():\n" + "".join(
        f"    x{i} = {i}\n\n" for i in range(3000))
    chunks = CodeSymbolChunker().chunk_doc(_doc(body, "/repo/big.py"))
    assert len(chunks) > 1
    assert all(c.meta.get("symbol") == "giant" for c in chunks)


def test_language_map():
    assert language_for_path("a/b/x.tsx") == "typescript"
    assert language_for_path("noext") is None


def test_plain_text_fallback_produces_chunks():
    text = "para one line a\nline b\n\n\npara two starts here\nmore\n"
    assert CodeSymbolChunker().chunk(text)
