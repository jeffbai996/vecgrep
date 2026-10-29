"""Paths that rebuild an index from stored payloads read `search_text` too.

Code chunks are embedded and BM25-indexed with their qualified name and
imports in front of the code (`search_text`); the payload keeps `text` for
display. Ingest, tarball import and backup restore used `search_text or text`,
but three paths read `text` alone (review R45, F1-F3):

- a BM25 rebuild (`vecgrep bm25 rebuild`, `doctor --reindex`, dirty-sidecar
  recovery, rename) dropped the qualified names from the lexical index;
- mutation-journal recovery did the same after an interrupted write;
- the cache keep-set hashed `text`, so a sweep deleted every code-chunk vector
  and the next index re-embedded the corpus.
"""
from __future__ import annotations

import shutil
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "code_symbols"
QUERY = "importer stats"        # matches Importer.Stats.run only through its qualified name


def _index_orders(svc, tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy(FIXTURES / "orders.py", src / "orders.py")
    svc.index(str(src), "code", chunker_name="code_symbol")
    return src


def _hits(svc) -> list[str]:
    return [h.chunk_id for h in svc.search(QUERY, "code", mode="bm25", top_k=5)]


def _definition(svc) -> str:
    """The chunk that defines Importer.Stats.run, found by its own text."""
    for _pid, payload in svc.store.iter_payloads("vecgrep__code"):
        if "def run(self, orders: list[Order]) -> float" in payload.get("text", ""):
            return _pid
    raise AssertionError("fixture has no Importer.Stats.run chunk")


def test_a_bm25_rebuild_keeps_qualified_names_searchable(svc, tmp_path) -> None:
    _index_orders(svc, tmp_path)
    before = _hits(svc)
    assert _definition(svc) in before
    svc.rebuild_bm25("code")
    assert _hits(svc) == before


def test_journal_recovery_keeps_qualified_names_searchable(svc, tmp_path, monkeypatch) -> None:
    src = _index_orders(svc, tmp_path)
    before = _hits(svc)
    assert _definition(svc) in before
    real = svc.bm25.upsert
    calls: list[int] = []

    def fails_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("disk failure in the BM25 write")
        return real(*args, **kwargs)

    # Qdrant accepts the next source and the BM25 half fails, leaving the
    # journal at qdrant_done for recovery to finish.
    monkeypatch.setattr(svc.bm25, "upsert", fails_once)
    (src / "extra.py").write_text("VALUE = 1\n", encoding="utf-8")
    try:
        svc.index(str(src), "code", chunker_name="code_symbol")
    except Exception:  # noqa: BLE001, S110 -- The following assertions verify recovery after the injected indexing failure.
        pass
    monkeypatch.setattr(svc.bm25, "upsert", real)
    assert calls, "the simulated failure never ran"
    assert "code" in svc.recover_pending_mutations()
    assert set(before) <= set(_hits(svc))


def test_a_cache_sweep_keeps_the_vectors_code_chunks_were_embedded_with(svc, tmp_path) -> None:
    src = _index_orders(svc, tmp_path)
    stub = svc._backend_cache["auto"]._inner
    svc.cache_sweep()
    calls = type(stub).calls
    svc.delete_corpus("code")
    svc.index(str(src), "code", chunker_name="code_symbol")
    assert type(stub).calls == calls, "the sweep deleted vectors the re-index needed"
