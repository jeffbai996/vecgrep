"""Human web-workspace contracts for vecgrep's mature retrieval surface."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vecgrep.backend.api import routes
from vecgrep.backend.api.schemas import BrowseRequest, IncidentRequest, SearchRequest


FRONTEND = Path(__file__).parent.parent / "vecgrep" / "frontend" / "src"


def test_search_request_supports_bounded_budget_breadth() -> None:
    req = SearchRequest(query="needle", budget=True, full_k=8, max_total=40)
    assert req.budget is True
    assert req.full_k == 8
    assert req.max_total == 40

    with pytest.raises(ValidationError):
        SearchRequest(query="needle", max_total=101)
    with pytest.raises(ValidationError):
        SearchRequest(query="needle", full_k=41, max_total=40)


def test_search_route_passes_budget_limit_to_service(monkeypatch) -> None:
    seen: dict = {}

    class FakeService:
        def search_budgeted(self, query, corpus, **kwargs):
            seen.update(query=query, corpus=corpus, **kwargs)
            return [], []

        def calibration(self, corpus):
            return {
                "cosine_center": 0.5,
                "cosine_slope": 10.0,
                "bm25_top": 5.0,
                "bm25_floor": 0.0,
            }

    monkeypatch.setattr(routes, "_SERVICE", FakeService())
    routes.search(
        SearchRequest(
            query="needle",
            corpus="notes",
            budget=True,
            full_k=6,
            max_total=40,
            explain=True,
        )
    )
    assert seen["full_k"] == 6
    assert seen["max_total"] == 40
    assert seen["explain"] is True


def test_browse_and_incident_routes_expose_service_tools(monkeypatch) -> None:
    seen: list[tuple[str, dict]] = []

    class FakeService:
        def browse(self, corpus, **kwargs):
            seen.append(("browse", {"corpus": corpus, **kwargs}))
            return []

        def incident(self, query, corpus, **kwargs):
            seen.append(("incident", {"query": query, "corpus": corpus, **kwargs}))
            return {"title": query, "confidence": "strong"}

    monkeypatch.setattr(routes, "_SERVICE", FakeService())

    assert routes.browse(
        BrowseRequest(corpus="cli", channel="discord", since="7d", tail=50)
    ) == []
    incident = routes.incident(
        IncidentRequest(query="what broke", corpus="cli", filters=["after:7d"])
    )
    assert incident["confidence"] == "strong"
    assert seen[0][1]["tail"] == 50
    assert seen[1][1]["filters"] == ["after:7d"]


def test_explorer_routes_expose_paginated_catalog_and_source_preview(monkeypatch) -> None:
    seen: list[tuple[str, dict]] = []

    class FakeService:
        def explore(self, corpus, **kwargs):
            seen.append(("explore", {"corpus": corpus, **kwargs}))
            return {"corpus": corpus, "folders": [], "documents": []}

        def explorer_source(self, corpus, source_id, *, max_chars):
            seen.append(
                (
                    "source",
                    {"corpus": corpus, "source_id": source_id, "max_chars": max_chars},
                )
            )
            return {"corpus": corpus, "source_id": source_id, "text": "hello"}

    monkeypatch.setattr(routes, "_SERVICE", FakeService())

    listing = routes.explore_corpus(
        "notes", path=["Memories"], q="launch", sort="newest", offset=10, limit=25
    )
    preview = routes.explorer_source("notes", source_id="memory-1", max_chars=5000)
    assert listing["documents"] == []
    assert preview["text"] == "hello"
    assert seen == [
        (
            "explore",
            {
                "corpus": "notes",
                "path": ["Memories"],
                "query": "launch",
                "sort": "newest",
                "offset": 10,
                "limit": 25,
            },
        ),
        (
            "source",
            {"corpus": "notes", "source_id": "memory-1", "max_chars": 5000},
        ),
    ]


def test_browse_request_requires_a_bounded_tail() -> None:
    assert BrowseRequest(corpus="cli", channel="discord").tail == 100

    with pytest.raises(ValidationError):
        BrowseRequest.model_validate(
            {"corpus": "cli", "source_path": "*", "tail": None}
        )


def test_web_ui_uses_dense_budgeted_results_and_insight_views() -> None:
    app = (FRONTEND / "App.tsx").read_text(encoding="utf-8")
    search = (FRONTEND / "components" / "SearchBar.tsx").read_text(encoding="utf-8")
    results = (FRONTEND / "components" / "ResultList.tsx").read_text(encoding="utf-8")
    api = (FRONTEND / "api.ts").read_text(encoding="utf-8")

    assert 'useState(40)' in search, "human search should show broad results by default"
    assert "budget: true" in api and "max_total" in api
    assert "response.stubs" in results, "stub tail must be visible, not discarded"
    assert "divide-y" in results and "space-y-3" not in results
    assert '"compare"' in app and '"browse"' in app
    assert "Incident" in (FRONTEND / "components" / "TimelinePanel.tsx").read_text(
        encoding="utf-8"
    )


def test_web_ui_browse_is_a_three_pane_explorer_with_search_reveal() -> None:
    app = (FRONTEND / "App.tsx").read_text(encoding="utf-8")
    browse = (FRONTEND / "components" / "BrowsePanel.tsx").read_text(encoding="utf-8")
    results = (FRONTEND / "components" / "ResultList.tsx").read_text(encoding="utf-8")
    api = (FRONTEND / "api.ts").read_text(encoding="utf-8")

    for pane in ("Collections", "Documents", "Preview"):
        assert pane in browse
    assert "api.explore" in browse and "api.explorerSource" in browse
    assert "onRevealSource" in results
    assert "revealSource" in app
    assert "explore:" in api and "explorerSource:" in api
    assert "Exact date" not in browse, "the old selector form must not survive"


def test_web_ui_browse_uses_the_shared_color_pill_language() -> None:
    browse = (FRONTEND / "components" / "BrowsePanel.tsx").read_text(encoding="utf-8")
    corpus_list = (FRONTEND / "components" / "CorpusList.tsx").read_text(encoding="utf-8")
    tones = (FRONTEND / "browseTones.ts").read_text(encoding="utf-8")

    for helper in ("corpusTone", "folderTone", "kindTone", "tagTone"):
        assert helper in browse
        assert helper in tones
    for color in ("violet", "sky", "emerald", "amber", "rose", "cyan"):
        assert color in tones
    assert "ColorPill" in browse
    assert "corpusTone" in corpus_list
    assert "Cataloging" in browse
    assert "Retry" in browse


def test_search_modes_share_the_result_badge_color_language() -> None:
    """Each mode button carries the retriever colour its result badge uses.

    ResultList tones a hit violet for VK, sky for V, emerald for K. The mode
    buttons must speak the same colour language so a violet result reads as
    "the mode I am on". Colour is the contract -- the literal VK/V/K letters
    were dropped from the buttons (2026-08-18) because a 9px badge span next
    to an 11px label cannot optically centre under items-center; the letters
    still appear on the result badges, where they have their own line box.
    """
    search = (FRONTEND / "components" / "SearchBar.tsx").read_text(encoding="utf-8")
    results = (FRONTEND / "components" / "ResultList.tsx").read_text(encoding="utf-8")

    for value, colour in (("hybrid", "violet"), ("vector", "sky"), ("bm25", "emerald")):
        block = search.split(f'value: "{value}"', 1)
        assert len(block) == 2, f"mode {value} missing from SearchBar"
        entry = block[1].split("},", 1)[0]
        assert colour in entry, f"mode {value} must use the {colour} retriever colour"

    # The colours are only a shared language if the result badges use them too.
    for colour in ("violet", "sky", "emerald"):
        assert colour in results, f"ResultList lost the {colour} badge tone"


def test_committed_web_bundle_has_no_private_companion_url() -> None:
    dist = FRONTEND.parent / "dist"
    built = "\n".join(
        path.read_text(encoding="utf-8")
        for path in dist.rglob("*")
        if path.is_file()
    )

    assert ".ts.net" not in built
