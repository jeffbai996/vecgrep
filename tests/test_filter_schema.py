"""Filterable-field discovery (item 6).

`filterable_fields` turns the opaque `filters` param into a discoverable schema:
the always-present source:/corpus: filters plus every meta.<key> actually
present in the corpus, with sample values.
"""
from __future__ import annotations

import json

import pytest

from vecgrep.backend.store import CorpusError


def test_reports_source_and_corpus_always(svc, make_doc):
    p = make_doc("d.md", "plain content here. another sentence follows.")
    svc.index(str(p), "c")
    schema = svc.filterable_fields("c")
    assert schema["corpus"] == "c"
    assert "source" in schema["filters"]
    assert "corpus" in schema["filters"]
    assert schema["filters"]["source"]["form"] == "source:GLOB"


def test_discovers_meta_keys_and_values(svc, make_doc):
    """Markdown adapter records adapter+path metadata; those keys show up."""
    p = make_doc("d.md", "content. more content.")
    svc.index(str(p), "c")
    schema = svc.filterable_fields("c")
    meta_keys = schema["filters"]["meta"]["keys"]
    # The markdown adapter writes {"adapter": "markdown", "path": ...}.
    assert "adapter" in meta_keys
    assert "markdown" in meta_keys["adapter"]


def test_reports_doc_timestamp_presence(svc, make_doc):
    dated = make_doc("d.md", "Saved: 2026-05-18T00:00:00+00:00\n\nbody content.")
    svc.index(str(dated), "c")
    schema = svc.filterable_fields("c")
    assert schema["has_doc_timestamp"] is True


def test_unknown_corpus_raises(svc):
    with pytest.raises(CorpusError):
        svc.filterable_fields("nonexistent")


def test_schema_is_json_serializable(svc, make_doc):
    """get_corpus serializes this to JSON — sets must be converted to lists."""
    p = make_doc("d.md", "content here.")
    svc.index(str(p), "c")
    schema = svc.filterable_fields("c")
    json.dumps(schema)  # must not raise (no raw sets)
