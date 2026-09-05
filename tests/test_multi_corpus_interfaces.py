from __future__ import annotations

import json
from click.testing import CliRunner
from pydantic import ValidationError
import pytest

from vecgrep.backend.api.schemas import CorpusContextRequest, SearchRequest
from vecgrep.backend.api import routes
from vecgrep.backend.assembly import ResultStub
from vecgrep.cli import main as cli_main
from vecgrep.backend.service import SearchOutcome, SearchWarning
from vecgrep.mcp import server as mcp_server


def test_rest_search_scope_validation() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        SearchRequest(query="x", corpus="a", corpora=["b"])
    with pytest.raises(ValidationError, match="cannot be empty"):
        SearchRequest(query="x", corpora=[])


def test_rest_context_hint_validation() -> None:
    with pytest.raises(ValidationError, match="cannot be empty"):
        CorpusContextRequest(use_for=[" "])
    with pytest.raises(ValidationError, match="240"):
        CorpusContextRequest(avoid_for=["x" * 241])


def test_rest_route_passes_plural_scope_and_returns_warnings(monkeypatch) -> None:
    seen = {}

    class Service:
        def search_with_diagnostics(self, query, corpus, top_k, **kwargs):
            seen.update(query=query, corpus=corpus, top_k=top_k, **kwargs)
            return SearchOutcome(
                [],
                [SearchWarning("cli", "search_failed", "RuntimeError: corpus search failed")],
            )

        def calibration(self, corpus):
            return {
                "cosine_center": 0.5,
                "cosine_slope": 10.0,
                "bm25_top": 5.0,
                "bm25_floor": 0.0,
            }

    monkeypatch.setattr(routes, "_SERVICE", Service())
    response = routes.search(
        SearchRequest(query="needle", corpora=["chats", "cli"])
    )
    assert seen["corpus"] is None
    assert seen["corpus_names"] == ["chats", "cli"]
    assert response.warnings[0].corpus == "cli"


def test_cli_repeated_corpus_sends_plural_scope(monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(cli_main, "_api_alive", lambda: True)

    def post(path, payload):
        seen.update(path=path, payload=payload)
        return {"hits": [], "warnings": []}

    monkeypatch.setattr(cli_main, "_post", post)
    result = CliRunner().invoke(
        cli_main.cli,
        ["search", "needle", "--corpus", "chats", "--corpus", "cli"],
    )
    assert result.exit_code == 0, result.output
    assert seen["payload"]["corpus"] is None
    assert seen["payload"]["corpora"] == ["chats", "cli"]


def test_cli_singular_corpus_keeps_legacy_field(monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(cli_main, "_api_alive", lambda: True)

    def post(path, payload):
        seen.update(payload)
        return {"hits": []}

    monkeypatch.setattr(cli_main, "_post", post)
    result = CliRunner().invoke(
        cli_main.cli, ["search", "needle", "--corpus", "chats"]
    )
    assert result.exit_code == 0, result.output
    assert seen["corpus"] == "chats"
    assert seen["corpora"] is None


def test_cli_context_posts_routing_fields_and_clear(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cli_main, "_api_alive", lambda: True)

    def post(path, payload):
        calls.append((path, payload))
        return payload

    monkeypatch.setattr(cli_main, "_post", post)
    runner = CliRunner()
    result = runner.invoke(
        cli_main.cli,
        [
            "corpora", "context", "notes",
            "--description", "Durable notes",
            "--use-for", "facts",
            "--avoid-for", "live state",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls[-1] == (
        "/api/corpora/notes/context",
        {
            "description": "Durable notes",
            "use_for": ["facts"],
            "avoid_for": ["live state"],
        },
    )

    result = runner.invoke(cli_main.cli, ["corpora", "context", "notes", "--clear"])
    assert result.exit_code == 0, result.output
    assert calls[-1][1] == {"description": "", "use_for": [], "avoid_for": []}


def test_mcp_success_shape_stays_list_and_partial_shape_adds_warnings(monkeypatch) -> None:
    class Service:
        warning = False

        def search_with_diagnostics(self, query, **kwargs):
            assert kwargs["corpus_names"] == ["chats", "cli"]
            assert kwargs["explain"] is True
            warnings = (
                [SearchWarning("cli", "search_failed", "RuntimeError: corpus search failed")]
                if self.warning else []
            )
            return SearchOutcome([], warnings)

    service = Service()
    monkeypatch.setattr(mcp_server, "_svc", lambda: service)
    args = {
        "query": "needle",
        "corpora": ["chats", "cli"],
        "rerank": False,
        "budget": False,
    }
    assert json.loads(mcp_server._run_search(args)) == []

    service.warning = True
    partial = json.loads(mcp_server._run_search(args))
    assert partial["hits"] == []
    assert partial["warnings"][0]["corpus"] == "cli"


def test_mcp_budget_stub_exposes_relevance_and_raw_scores(monkeypatch) -> None:
    class Service:
        def search_budgeted_with_diagnostics(self, query, **kwargs):
            assert kwargs["explain"] is True
            return [], [ResultStub(
                chunk_id="cid-1",
                corpus="chats",
                source_id="cl-1/2026-08-30.md",
                doc_timestamp=1_788_000_000.0,
                snippet="needle in context",
                score=0.02,
                similarity_pct=82.5,
                relevance_pct=82.5,
                relevance_label="strong",
                matched_by=("vector", "bm25"),
                scores={"vector_cosine": 0.6888, "vector_rank": 1},
            )], []

    monkeypatch.setattr(mcp_server, "_svc", lambda: Service())
    payload = json.loads(mcp_server._run_search({
        "query": "needle",
        "corpus": "chats",
        "rerank": False,
        "budget": True,
    }))
    stub = payload["stubs"][0]
    assert stub["similarity_pct"] == stub["relevance_pct"] == 82.5
    assert stub["relevance_label"] == "strong"
    assert stub["matched_by"] == ["vector", "bm25"]
    assert stub["scores"] == {"vector_cosine": 0.6888, "vector_rank": 1}
