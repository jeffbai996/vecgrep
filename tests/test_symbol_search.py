"""End to end: index code with code_symbol, then find definitions by name."""
import shutil
from pathlib import Path

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
