from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from vecgrep.backend.store.corpora import Corpus, CorpusError, CorpusRegistry


def test_old_registry_loads_empty_routing_context(tmp_path) -> None:
    corpus = Corpus("old", "stub", "stub-1", 16)
    payload = asdict(corpus)
    payload.pop("description")
    payload.pop("use_for")
    payload.pop("avoid_for")
    path = tmp_path / "corpora.json"
    path.write_text(json.dumps({"old": payload}))

    loaded = CorpusRegistry(path).get("old")

    assert loaded.description == ""
    assert loaded.use_for == []
    assert loaded.avoid_for == []


def test_context_round_trips_clears_and_survives_export_import(
    svc, make_doc, tmp_path
) -> None:
    svc.index(str(make_doc("doc.md", "alpha beta gamma")), "notes")
    updated = svc.set_corpus_context(
        "notes",
        "  Durable notes.  ",
        [" facts ", " project state"],
        [" live prices "],
    )
    assert updated.description == "Durable notes."
    assert updated.use_for == ["facts", "project state"]
    assert updated.avoid_for == ["live prices"]
    assert svc.list_corpora()[0].description == "Durable notes."

    archive = svc.export_corpus("notes", tmp_path / "notes.tar.gz")
    svc.delete_corpus("notes")
    restored = svc.import_corpus(archive, rename="restored")
    assert restored.description == "Durable notes."
    assert restored.use_for == ["facts", "project state"]
    assert restored.avoid_for == ["live prices"]

    cleared = svc.set_corpus_context("restored")
    assert (cleared.description, cleared.use_for, cleared.avoid_for) == ("", [], [])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"description": "x" * 501}, "500"),
        ({"use_for": [str(i) for i in range(9)]}, "at most 8"),
        ({"avoid_for": ["x" * 241]}, "240"),
        ({"use_for": ["  "]}, "cannot be empty"),
    ],
)
def test_context_limits(svc, make_doc, kwargs, message) -> None:
    svc.index(str(make_doc("doc.md", "alpha")), "notes")
    with pytest.raises(CorpusError, match=message):
        svc.set_corpus_context("notes", **kwargs)
