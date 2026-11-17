from __future__ import annotations

import logging
import threading
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from vecgrep.backend.api import routes
from vecgrep.backend.api.schemas import BrowseRequest, IncidentRequest, SearchRequest

# ── test_api_background_recovery ──

def test_api_service_does_not_block_first_request_on_recovery(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    class FakeService:
        def __init__(self, *, recover_pending):
            assert recover_pending is False

        def recover_pending_mutations(self):
            entered.set()
            assert release.wait(2)
            return ["busy-corpus"]

    monkeypatch.setattr(routes, "VecgrepService", FakeService)
    monkeypatch.setattr(routes, "_SERVICE", None)
    try:
        service = routes._service()
        assert isinstance(service, FakeService)
        assert entered.wait(1)
    finally:
        release.set()
        routes._SERVICE = None

# ── test_bearer_token ──

class _Settings:
    def __init__(self, token):
        self.api_token = token


def test_trailing_newline_in_configured_token_still_authenticates(monkeypatch):
    monkeypatch.setattr(routes, "get_settings", lambda: _Settings("s3cr3t-token\n"))
    # Must NOT raise — the configured token's trailing newline is stripped.
    routes.require_token("Bearer s3cr3t-token")


def test_wrong_token_is_rejected(monkeypatch):
    monkeypatch.setattr(routes, "get_settings", lambda: _Settings("s3cr3t-token"))
    with pytest.raises(HTTPException) as ei:
        routes.require_token("Bearer not-the-token")
    assert ei.value.status_code == 403


def test_missing_token_is_401(monkeypatch):
    monkeypatch.setattr(routes, "get_settings", lambda: _Settings("s3cr3t-token"))
    with pytest.raises(HTTPException) as ei:
        routes.require_token(None)
    assert ei.value.status_code == 401


def test_no_configured_token_is_noop(monkeypatch):
    monkeypatch.setattr(routes, "get_settings", lambda: _Settings(None))
    routes.require_token(None)  # gate disabled → no-op

# ── test_search_stage_timings ──

STAGES = {"admit", "embed", "vector", "bm25", "retrieve", "total"}


def _index(svc, make_doc, name, corpus):
    svc.index(str(make_doc(name, "alpha beta gamma delta. epsilon zeta eta theta.")), corpus)


def test_a_hybrid_search_reports_every_retrieval_stage(svc, make_doc) -> None:
    _index(svc, make_doc, "a.md", "notes")
    outcome = svc.search_with_diagnostics("alpha beta", corpus_name="notes")
    t = outcome.timings_ms
    assert STAGES <= set(t)
    assert all(v >= 0 for v in t.values())
    assert t["total"] >= t["retrieve"]


def test_stage_times_sum_across_corpora_in_an_unscoped_search(svc, make_doc) -> None:
    _index(svc, make_doc, "a.md", "notes")
    _index(svc, make_doc, "b.md", "chats")
    t = svc.search_with_diagnostics("alpha beta").timings_ms
    assert STAGES <= set(t)
    assert t["corpora"] == 2


def test_rerank_time_is_reported_only_when_the_reranker_ran(svc, make_doc, monkeypatch) -> None:
    _index(svc, make_doc, "a.md", "notes")
    assert "rerank" not in svc.search_with_diagnostics("alpha", corpus_name="notes").timings_ms

    monkeypatch.setattr(svc, "_rerank_ready", lambda model=None: True)
    monkeypatch.setattr(
        "vecgrep.backend.rerank.rerank",
        lambda q, pairs, model: [(0.9, p[1]) for p in pairs],
    )
    t = svc.search_with_diagnostics("alpha", corpus_name="notes", rerank=True).timings_ms
    assert t["rerank"] >= 0


def test_waiting_for_the_reranker_is_its_own_stage(svc, make_doc, monkeypatch) -> None:
    """After a restart the readiness wait can take up to RERANK_WAIT_S; it must
    show up as a stage rather than as unexplained time inside `total`."""
    import time

    _index(svc, make_doc, "a.md", "notes")
    assert "rerank_wait" not in svc.search_with_diagnostics("alpha", corpus_name="notes").timings_ms

    def slow_not_ready(model=None):
        time.sleep(0.05)
        return False

    monkeypatch.setattr(svc, "_rerank_ready", slow_not_ready)
    t = svc.search_with_diagnostics("alpha", corpus_name="notes", rerank=True).timings_ms
    assert t["rerank_wait"] >= 40
    assert "rerank" not in t
    assert t["total"] >= t["rerank_wait"]


def test_the_rest_search_response_carries_the_timings(svc, make_doc, monkeypatch) -> None:
    from vecgrep.backend.api import routes
    from vecgrep.backend.api.schemas import SearchRequest

    _index(svc, make_doc, "a.md", "notes")
    monkeypatch.setattr(routes, "_service", lambda: svc)
    body = routes.search(
        SearchRequest(
            query="alpha",
            corpus="notes",
            top_k=None,
            full_k=8,
            max_total=100,
            token_ceiling=4000,
        )
    )
    assert STAGES <= set(body.timings_ms)


def test_per_request_http_client_logging_is_quiet() -> None:
    """httpx logs every qdrant call at INFO: ~18k journal lines per 6 h."""
    from vecgrep.backend import main  # noqa: F401 -- importing applies the level
    assert logging.getLogger("httpx").level >= logging.WARNING


def test_time_queued_behind_a_write_lock_is_reported_as_admit(svc, make_doc, monkeypatch) -> None:
    import contextlib
    import time as _time
    _index(svc, make_doc, "a.md", "notes")
    real_read = svc.locks.read

    @contextlib.contextmanager
    def slow_read(name, timeout=None):
        _time.sleep(0.05)
        with real_read(name, timeout=timeout):
            yield

    monkeypatch.setattr(svc.locks, "read", slow_read)
    t = svc.search_with_diagnostics("alpha", corpus_name="notes").timings_ms
    assert t["admit"] >= 50

# ── test_web_workspace ──

FRONTEND = Path(__file__).parent.parent / "vecgrep" / "frontend" / "src"


def test_search_request_supports_bounded_budget_breadth() -> None:
    req = SearchRequest(
        query="needle",
        budget=True,
        full_k=8,
        max_total=40,
        top_k=None,
        token_ceiling=4000,
    )
    assert req.budget is True
    assert req.full_k == 8
    assert req.max_total == 40

    with pytest.raises(ValidationError):
        SearchRequest(
            query="needle",
            max_total=101,
            top_k=None,
            full_k=8,
            token_ceiling=4000,
        )
    with pytest.raises(ValidationError):
        SearchRequest(
            query="needle",
            full_k=41,
            max_total=40,
            top_k=None,
            token_ceiling=4000,
        )


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
            top_k=None,
            token_ceiling=4000,
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
    assert incident is not None
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


















def test_committed_web_bundle_has_no_private_companion_url() -> None:
    dist = FRONTEND.parent / "dist"
    # Text only: dist also carries the raster icons, and a PNG is not UTF-8.
    built = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in dist.rglob("*")
        if path.is_file() and path.suffix not in {".png", ".ico", ".woff", ".woff2"}
    )

    assert ".ts.net" not in built














def test_the_ico_really_carries_more_than_one_size() -> None:
    """PIL derives each ICO entry from the base image it is handed, so building
    the file from a 16px render clamped all three entries to 16x16 and a 32px
    tab bar got an upscaled blur (found 2026-09-09). Render large once, then
    let it step down.

    The header is read directly rather than through PIL: this suite must not
    grow an image dependency to check a 674-byte file. An ICO is a 6-byte
    header, then one 16-byte directory entry per image whose first two bytes
    are width and height, with 0 meaning 256.
    """
    import struct

    raw = (FRONTEND.parent / "public" / "favicon.ico").read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", raw, 0)
    assert (reserved, kind) == (0, 1), "not an icon file"
    sizes = sorted((raw[6 + i * 16] or 256, raw[7 + i * 16] or 256)
                   for i in range(count))
    assert (32, 32) in sizes, f"only {sizes} in the ico"
    assert len(sizes) >= 2


def test_the_svg_carries_explicit_dimensions() -> None:
    """A viewBox alone is enough for most renderers and not all of them."""
    svg = (FRONTEND.parent / "public" / "favicon.svg").read_text(encoding="utf-8")
    assert 'width="512"' in svg and 'height="512"' in svg






def test_the_icon_links_carry_a_version() -> None:
    """A browser that once asked for /favicon.ico and was handed the SPA's
    index.html caches that miss, and it keeps caching it after the file is
    real: the icon shipped at 03:22 and the tab was still blank at 07:25
    (the bytes were on the wire the whole time). A version on the
    href is a different URL, so the stale entry cannot answer for it."""
    import re

    head = (FRONTEND.parent / "index.html").read_text(encoding="utf-8")
    hrefs = re.findall(r'<link[^>]*rel="(?:icon|apple-touch-icon)"[^>]*'
                       r'href="([^"]+)"', head)
    assert len(hrefs) == 3, f"expected three icon links, found {hrefs}"
    for href in hrefs:
        assert re.search(r"\?v=\d+$", href), f"{href} is not versioned"
    # The query is not part of the route, so the backend still serves these
    # by plain name — bumping the version must never need a server change.
    assert len({h.split("?")[1] for h in hrefs}) == 1, \
        "bump every icon together or the browser refetches only some"


def test_health_endpoints_auth_and_avoid_stats_walk(svc, make_doc, monkeypatch) -> None:
    import urllib.error
    import urllib.request

    from fastapi.testclient import TestClient

    from vecgrep.backend import config as cfg_mod
    from vecgrep.backend.main import create_app

    svc.index(str(make_doc("health.md", "health check marker")), "notes")

    def fail_stats(*_args, **_kwargs):
        raise AssertionError("instance_health must not call corpus_stats")

    def fail_urlopen(*_args, **_kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(svc, "corpus_stats", fail_stats)
    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)
    monkeypatch.setattr(routes, "_SERVICE", svc)
    monkeypatch.setenv("VECGREP_API_TOKEN", "test-tok" + "en-health")
    monkeypatch.setenv("VECGREP_QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setattr(cfg_mod, "_settings", None)

    client = TestClient(create_app(), base_url="http://127.0.0.1")
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/health/detail").status_code == 401
    resp = client.get(
        "/api/health/detail",
        headers={"Authorization": "Bearer test-tok" + "en-health"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    for key in ("checks", "corpora", "cache", "process"):
        assert key in payload
    check_names = {c["name"] for c in payload["checks"]}
    for expected in ("embedding model", "embed cache", "vector store", "search"):
        assert expected in check_names


def test_search_reports_its_own_wall_time(monkeypatch) -> None:
    class FakeService:
        def search_with_diagnostics(self, *_args, **_kwargs):
            from types import SimpleNamespace

            return SimpleNamespace(results=[], warnings=[], timings_ms={})

        def search_budgeted(self, *_args, **_kwargs):
            return [], []

        def calibration(self, _corpus):
            return {"cosine_center": 0.5, "cosine_slope": 10.0, "bm25_top": 5.0, "bm25_floor": 0.0}

    monkeypatch.setattr(routes, "_SERVICE", FakeService())
    normal = routes.search(SearchRequest.model_validate({"query": "needle"}))
    budgeted = routes.search(
        SearchRequest(query="needle", corpus="notes", budget=True, full_k=5, max_total=20, top_k=None, token_ceiling=2000)
    )
    assert normal.took_ms is not None and normal.took_ms >= 0
    assert budgeted.took_ms is not None and budgeted.took_ms >= 0


def test_every_icon_a_browser_asks_for_is_served() -> None:
    from fastapi.testclient import TestClient

    from vecgrep.backend.main import create_app

    public = FRONTEND.parent / "public"
    dist = FRONTEND.parent / "dist"
    for name in ("favicon.svg", "favicon.ico", "apple-touch-icon.png"):
        assert (public / name).is_file(), f"{name} missing from public/"
        assert (dist / name).is_file(), f"{name} did not reach dist/"

    head = (FRONTEND.parent / "index.html").read_text(encoding="utf-8")
    assert 'href="/favicon.ico' in head
    assert 'rel="apple-touch-icon"' in head and "apple-touch-icon.png" in head

    client = TestClient(create_app())
    for name, expected_type in (
        ("favicon.svg", "image/svg+xml"),
        ("favicon.ico", "image/x-icon"),
        ("apple-touch-icon.png", "image/png"),
    ):
        res = client.get(f"/{name}")
        assert res.status_code == 200
        assert expected_type in res.headers["content-type"]
    assert "text/html" in client.get("/not-an-icon.png").headers["content-type"]
