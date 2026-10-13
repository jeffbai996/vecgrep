"""Identifier queries rank and pin the symbol's definition."""
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
