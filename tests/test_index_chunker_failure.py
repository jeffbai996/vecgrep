"""A document whose chunker raises is skipped and reported; the rest index.

Real bug: a deeply nested generated file raised RecursionError out of the
code_symbol chunker, and nothing between it and /api/index caught it, so every
file after it in the directory went unindexed and the request returned 500.
"""
from __future__ import annotations

from vecgrep.backend.api import routes
from vecgrep.backend.api.schemas import IndexRequest


class _RaisesOn:
    def __init__(self, inner, suffix: str) -> None:
        self.inner, self.suffix = inner, suffix

    def chunk_doc(self, doc):
        if doc.source_id.endswith(self.suffix):
            raise RecursionError("maximum recursion depth exceeded")
        return self.inner.chunk_doc(doc)


def _repo(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    (d / "a.py").write_text("def alpha():\n    return 1\n")
    (d / "b.py").write_text("def beta():\n    return 2\n")
    (d / "c.py").write_text("def gamma():\n    return 3\n")
    return d


def test_a_chunker_failure_skips_that_source_and_reports_it(svc, tmp_path, monkeypatch):
    d = _repo(tmp_path)
    real = svc.chunker("code_symbol")
    monkeypatch.setattr(svc, "chunker", lambda name: _RaisesOn(real, "b.py"))
    monkeypatch.setattr(routes, "_service", lambda: svc)

    out = routes.index(IndexRequest(source=str(d), corpus="code", chunker="code_symbol"))

    assert out.docs == 2
    assert len(out.failed) == 1
    assert out.failed[0].startswith(str((d / "b.py").resolve())) and "RecursionError" in out.failed[0]
    corpus = svc.registry.get("code")
    assert sorted(s.rsplit("/", 1)[-1] for s in corpus.sources) == ["a.py", "c.py"]


def test_a_skipped_source_is_tried_again_on_the_next_run(svc, tmp_path, monkeypatch):
    d = _repo(tmp_path)
    real = svc.chunker("code_symbol")
    monkeypatch.setattr(svc, "chunker", lambda name: _RaisesOn(real, "b.py"))
    svc.index(str(d), "code", "code_symbol")

    monkeypatch.setattr(svc, "chunker", lambda name: real)
    failed: list[str] = []
    docs, _, skipped = svc.index(str(d), "code", "code_symbol", failed=failed)

    assert (docs, skipped, failed) == (1, 2, [])
