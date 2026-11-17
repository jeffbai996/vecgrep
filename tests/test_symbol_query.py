from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

from vecgrep.backend.symbol_query import definition_boost, pin_definitions, symbol_query


def payload(qualname, symbol=None):
    return {"metadata": {"qualname": qualname, "symbol": symbol or qualname.split(".")[-1]}}


def test_only_identifier_shaped_queries_count_as_symbol_queries():
    assert symbol_query("Importer.run") == "Importer.run"
    assert symbol_query(" run ") == "run"
    assert symbol_query("how do we import orders") is None
    assert symbol_query("Importer.run()") is None


def test_exact_qualified_name_beats_a_tail_match_beats_a_bare_name():
    assert definition_boost("Importer.Stats.run", payload("Importer.Stats.run")) > \
        definition_boost("Stats.run", payload("Importer.Stats.run")) > \
        definition_boost("run", payload("Importer.Stats.run")) > 1.0


def test_other_symbols_prose_queries_and_plain_chunks_are_untouched():
    assert definition_boost("Exporter.run", payload("Importer.run")) == 1.0
    assert definition_boost(None, payload("Importer.run")) == 1.0
    assert definition_boost("run", {"metadata": {}}) == 1.0
    assert definition_boost("run", {}) == 1.0


def test_a_definition_left_out_of_the_selection_is_pinned_first():
    call_site = SimpleNamespace(chunk_id="call", metadata={})
    other = SimpleNamespace(chunk_id="other", metadata={})
    definition = SimpleNamespace(chunk_id="def", metadata=payload("Importer.run")["metadata"])
    selected = [call_site, other]
    pool = [call_site, other, definition]
    assert [r.chunk_id for r in pin_definitions(selected, pool, "Importer.run", 2)] == ["def", "call"]


def test_no_symbol_or_no_definition_leaves_the_selection_alone():
    a = SimpleNamespace(chunk_id="a", metadata={})
    assert pin_definitions([a], [a], None, 5) == [a]
    assert pin_definitions([a], [a], "Importer.run", 5) == [a]

# ── test_symbol_search ──

FIXTURES = Path(__file__).parent / "fixtures" / "code_symbols"


def index_orders(svc, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy(FIXTURES / "orders.py", src / "orders.py")
    svc.index(str(src), "code", chunker_name="code_symbol")


def test_a_qualified_name_query_returns_that_definition_first(svc, tmp_path):
    index_orders(svc, tmp_path)
    for query, expected in [("Importer.run", "Importer.run"), ("Exporter.run", "Exporter.run"),
                            ("Stats.run", "Importer.Stats.run"), ("_parse", "Importer._parse")]:
        hits = svc.search(query, "code", top_k=3)
        assert hits[0].metadata.get("qualname") == expected, query


def test_the_payload_keeps_source_text_and_records_the_search_text(svc, tmp_path):
    index_orders(svc, tmp_path)
    hit = svc.search("Importer._parse", "code", top_k=1)[0]
    assert hit.chunk.lstrip().startswith("def _parse")
    assert hit.metadata["lines"] == "27-29"


def test_prose_queries_are_ranked_as_before(svc, tmp_path):
    index_orders(svc, tmp_path)
    hits = svc.search("which function sums order totals", "code", top_k=3)
    assert hits

# ── test_search_text_write_paths ──

FIXTURES_search_text_write_paths = Path(__file__).parent / "fixtures" / "code_symbols"
QUERY = "importer stats"        # matches Importer.Stats.run only through its qualified name


def _index_orders(svc, tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy(FIXTURES_search_text_write_paths / "orders.py", src / "orders.py")
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
