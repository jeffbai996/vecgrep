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


def test_corpus_dots_use_collision_free_high_contrast_tones() -> None:
    corpus_list = (FRONTEND / "components" / "CorpusList.tsx").read_text(
        encoding="utf-8"
    )
    tones = (FRONTEND / "browseTones.ts").read_text(encoding="utf-8")

    assert "corpusTone(c.name, corpusNames)" in corpus_list
    assert "h-2 w-2" in corpus_list
    assert "ring-white/20" in corpus_list
    assert "usedToneIndexes" in tones
    assert "CORPUS_PROBE_STEP" in tones


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


def test_search_idle_state_is_an_operations_dashboard() -> None:
    app = (FRONTEND / "App.tsx").read_text(encoding="utf-8")
    search = (FRONTEND / "components" / "SearchBar.tsx").read_text(encoding="utf-8")
    results = (FRONTEND / "components" / "ResultList.tsx").read_text(encoding="utf-8")
    dashboard = (FRONTEND / "components" / "SearchDashboard.tsx").read_text(
        encoding="utf-8"
    )

    assert "onPrimeQuery" not in app and "onPrimeQuery" not in results
    assert "SearchDashboard" in results
    for signal in (
        "Index overview",
        "API reachable",
        "Latest index activity",
        "Embedding inventory",
        "Corpus distribution",
    ):
        assert signal in dashboard
    for prompt in ("Find a decision", "Trace a change", "Recover context"):
        assert prompt not in results and prompt not in dashboard
    assert "Search results will appear here" not in results
    assert "h-[22px] w-[22px]" in search
    assert "⌕" not in search, "the tiny text glyph must not return"


def test_search_explainers_do_not_advertise_retired_models() -> None:
    sidebar = (FRONTEND / "components" / "HowSearchWorks.tsx").read_text(
        encoding="utf-8"
    )
    footer = (FRONTEND / "components" / "AboutFooter.tsx").read_text(
        encoding="utf-8"
    )
    copy = sidebar + footer

    for stale_literal in (
        "nomic-embed-text",
        "mxbai-embed-large",
        "text-embedding-3-small",
        "bge-reranker-base",
        "~127ms",
    ):
        assert stale_literal not in copy
    assert "active model inventory" in footer
    assert "configured cross-encoder" in sidebar


def test_score_tuning_names_its_controls_without_explaining_them() -> None:
    app = (FRONTEND / "App.tsx").read_text(encoding="utf-8")
    panel = (FRONTEND / "components" / "TuningPanel.tsx").read_text(
        encoding="utf-8"
    )
    tuning = (FRONTEND / "tuning.ts").read_text(encoding="utf-8")

    assert "Score tuning" in panel
    for section in ("Semantic scoring", "Keyword scoring", "Hybrid balance"):
        assert section in panel
    assert "Automatic" in panel and "Custom" in panel
    assert "onReset" in panel and "clearTuning" in app
    assert "localStorage.removeItem" in tuning
    # The panel explains itself by working: a named control, a live number and
    # the two words at each end of its track. The paragraphs that used to sit
    # above and under the sliders are tooltips now (Jeff 2026-09-09, and the
    # squad's no-explanatory-copy rule).
    assert "title={slider.help}" in panel, "the help text lost its tooltip home"
    for gone in ("Search retrieval stays unchanged",
                 "How raw matches become percentages",
                 "Changes apply instantly"):
        assert gone not in panel, f"explanatory copy is back: {gone!r}"
    # Two columns where there is room; the sliders used to stack one per row.
    assert panel.count("sm:grid-cols-2") >= 2, "the slider rows do not pair up"


def test_committed_web_bundle_has_no_private_companion_url() -> None:
    dist = FRONTEND.parent / "dist"
    # Text only: dist also carries the raster icons, and a PNG is not UTF-8.
    built = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in dist.rglob("*")
        if path.is_file() and path.suffix not in {".png", ".ico", ".woff", ".woff2"}
    )

    assert ".ts.net" not in built


def test_the_app_ships_a_favicon_and_serves_it() -> None:
    """vecgrep had no icon at all — no link in the head, nothing on disk — so
    every tab showed the browser's default globe. And only /assets is mounted,
    so a file at the root falls through to the SPA catch-all and the browser
    asking for /favicon.svg gets handed a page (Jeff 2026-09-09)."""
    head = (FRONTEND.parent / "index.html").read_text(encoding="utf-8")
    assert 'rel="icon"' in head and "favicon.svg" in head

    icon = FRONTEND.parent / "public" / "favicon.svg"
    assert icon.is_file(), "no favicon source"
    svg = icon.read_text(encoding="utf-8")
    assert svg.lstrip().startswith("<svg"), "favicon is not an svg"
    assert 'viewBox="0 0 512 512"' in svg, "off the squad's 512 icon grid"

    built = FRONTEND.parent / "dist" / "favicon.svg"
    assert built.is_file(), "favicon did not make it into dist/"

    main = (
        FRONTEND.parent.parent / "backend" / "main.py"
    ).read_text(encoding="utf-8")
    assert '"favicon.svg"' in main, (
        "nothing serves /favicon.svg, so it falls through to the SPA catch-all "
        "and the browser is handed index.html")


def test_instance_health_is_a_page_not_a_question() -> None:
    """Jeff 2026-09-09: "so that I can inspect vecgrep health at a glance
    without having to ask yall". A Health view, backed by an endpoint that is
    cheap enough to poll."""
    app = (FRONTEND / "App.tsx").read_text(encoding="utf-8")
    panel = (FRONTEND / "components" / "InstanceHealth.tsx").read_text(encoding="utf-8")
    client = (FRONTEND / "api.ts").read_text(encoding="utf-8")
    routes = (
        FRONTEND.parent.parent / "backend" / "api" / "routes.py"
    ).read_text(encoding="utf-8")

    assert '"health"' in app and "<InstanceHealth />" in app, "no Health view"
    assert '"/api/health/detail"' in client
    # NOT bolted onto /api/health: that one is the public, unauthenticated
    # liveness probe and is exempt from api_token, while this reports corpus
    # names, on-disk paths and process memory.
    assert '@router.get("/health/detail")' in routes
    assert '@public_router.get("/health")' in routes, "the public probe moved"

    # The checks that matter, and the numbers behind them.
    for signal in ("embedding model", "embed cache", "vector store", "search"):
        assert signal in routes, f"health lost its {signal!r} check"
    for shown in ("checks", "corpora", "cache", "process"):
        assert shown in panel, f"the page does not render {shown}"


def test_health_detail_avoids_the_expensive_stats_walk() -> None:
    """corpus_stats iterates every payload in a corpus. Fine on demand for one
    corpus, hopeless for a page you glance at, so the snapshot is built from
    counts the registry already holds."""
    routes = (
        FRONTEND.parent.parent / "backend" / "api" / "routes.py"
    ).read_text(encoding="utf-8")
    body = routes[routes.index('@router.get("/health/detail")'):]
    body = body[:body.index('@router.get("/stats/')]
    # The call, not the word: the docstring names corpus_stats to explain why
    # it is avoided.
    assert ".corpus_stats(" not in body, "the health page walks every payload"
    assert "list_corpora" in body


def test_the_view_lives_in_the_url() -> None:
    """A tab you cannot link to is a tab you have to describe over Discord.
    The view was component state, so /#health could not be sent, bookmarked or
    reloaded into (Jeff 2026-09-09, asking for health at a glance)."""
    app = (FRONTEND / "App.tsx").read_text(encoding="utf-8")
    assert "viewFromHash" in app, "the view does not read the URL"
    assert "hashchange" in app, "back/forward and a hand-edited hash do nothing"
    assert "window.location.hash = view" in app, "the URL never follows the view"


def test_search_reports_its_own_wall_time() -> None:
    """Jeff 2026-09-09: "it took like over a minute to load. we should include
    load time in seconds on the vecgrep results tbh so i can debug a bit
    better". Server-side, so a slow retrieval can be told apart from a slow
    round trip — which from a browser look identical."""
    schemas = (
        FRONTEND.parent.parent / "backend" / "api" / "schemas.py"
    ).read_text(encoding="utf-8")
    routes = (
        FRONTEND.parent.parent / "backend" / "api" / "routes.py"
    ).read_text(encoding="utf-8")
    client = (FRONTEND / "api.ts").read_text(encoding="utf-8")
    results = (FRONTEND / "components" / "ResultList.tsx").read_text(encoding="utf-8")

    assert "took_ms" in schemas, "the response cannot carry a time"
    # Both return paths: budget mode and the ordinary one.
    assert routes.count("took_ms=took()") >= 2, "a search path returns no time"
    assert "took_ms" in client and "took_ms" in results, "the page never shows it"


def test_every_icon_a_browser_asks_for_is_served() -> None:
    """One SVG was not enough. Safari ignores an SVG apple-touch-icon, and a
    browser falling back to /favicon.ico was being handed the SPA's index.html
    — so Jeff saw no icon at all (2026-09-09)."""
    public = FRONTEND.parent / "public"
    dist = FRONTEND.parent / "dist"
    for name in ("favicon.svg", "favicon.ico", "apple-touch-icon.png"):
        assert (public / name).is_file(), f"{name} missing from public/"
        assert (dist / name).is_file(), f"{name} did not reach dist/"

    head = (FRONTEND.parent / "index.html").read_text(encoding="utf-8")
    assert 'href="/favicon.ico"' in head
    assert 'rel="apple-touch-icon"' in head and 'apple-touch-icon.png' in head, \
        "apple-touch-icon must point at a PNG; Safari ignores an SVG here"

    main = (
        FRONTEND.parent.parent / "backend" / "main.py"
    ).read_text(encoding="utf-8")
    for name in ("favicon.svg", "favicon.ico", "apple-touch-icon.png"):
        assert f'"{name}"' in main, f"nothing serves /{name}"
    # A /{path} catch-all would swallow every unmatched GET in the app.
    assert '@app.get("/{icon:path}"' not in main


def test_the_svg_carries_explicit_dimensions() -> None:
    """A viewBox alone is enough for most renderers and not all of them."""
    svg = (FRONTEND.parent / "public" / "favicon.svg").read_text(encoding="utf-8")
    assert 'width="512"' in svg and 'height="512"' in svg
