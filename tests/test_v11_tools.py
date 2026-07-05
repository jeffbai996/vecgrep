"""v1.1 tool batch — location-first reading + transparency + hygiene.

browse       — the query-less primitive: "show me channel X on date Y".
               Everything in v1.0 was query-first; the archivist workflow is
               often location-first. No ranking, no query — just the day's
               event sequence.
get_source   — whole source by source_id (a stub / corpus listing gives you
               the source_id; get_chunk needed a chunk_id you may not have).
list_aliases — read-only view of the active alias map, so an assistant can
               see why a query expanded (or didn't).
Plus hygiene pinned here: MCP handlers reuse ONE service per settings
generation (was: fresh VecgrepService per tool call), and the stdio
transport exposes the same propose_* write tools as HTTP.
"""
from __future__ import annotations

import json


# ── browse ───────────────────────────────────────────────────────────────────

def test_browse_by_channel_and_date(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    groups = svc.browse("evalchat", channel="channel-a", date="2026-01-15")
    assert len(groups) == 1
    g = groups[0]
    assert g["source_id"].endswith("channel-a/2026-01-15.md")
    events = g["events"]
    assert len(events) >= 20, "whole day, not a slice"
    assert events[0]["time"] <= events[-1]["time"], "chronological"
    assert any(e["speaker"] == "alice" for e in events)


def test_browse_channel_only_returns_all_days(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    groups = svc.browse("evalchat", channel="channel-a")
    assert {g["source_id"].split("/")[-1] for g in groups} == {
        "2026-01-08.md", "2026-01-15.md",
    }
    # Oldest first — it reads as a ledger.
    stamps = [g["doc_timestamp"] for g in groups]
    assert stamps == sorted(stamps)


def test_browse_date_only_spans_channels(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    groups = svc.browse("evalchat", date="2026-01-15")
    channels = {g["source_id"].split("/")[-2] for g in groups}
    assert channels == {"channel-a", "channel-b", "channel-c"}


def test_browse_source_path_glob(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    groups = svc.browse("evalchat", source_path="*channel-b*")
    assert len(groups) == 1
    assert "channel-b" in groups[0]["source_id"]


def test_browse_no_selector_rejected(svc) -> None:
    """No channel/date/glob = dump the whole corpus — refuse loudly instead
    of silently flooding the context."""
    import pytest

    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    with pytest.raises(ValueError):
        svc.browse("evalchat")


def test_browse_no_match_is_empty(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    assert svc.browse("evalchat", channel="channel-zz") == []


# ── get_source ───────────────────────────────────────────────────────────────

def test_get_source_returns_full_text_and_events(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    sid = next(
        s for s in svc.bm25._load("evalchat").by_source
        if s.endswith("channel-b/2026-01-15.md")
    )
    doc = svc.get_source("evalchat", sid)
    assert doc is not None
    assert doc["source_id"] == sid
    assert "root cause" in doc["text"]
    assert doc["doc_timestamp"] is not None
    assert len(doc["events"]) >= 8, "transcript parsed into events"


def test_get_source_unknown_id_is_none(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    assert svc.get_source("evalchat", "/no/such/file.md") is None


def test_get_source_non_transcript_has_no_events(svc, make_doc) -> None:
    p = make_doc("plain.md", "Just ordinary prose in a document. Second sentence here.")
    svc.index(str(p), "c")
    doc = svc.get_source("c", str(p.resolve()))
    assert doc is not None and doc["events"] == []
    assert "ordinary prose" in doc["text"]


# ── list_aliases ─────────────────────────────────────────────────────────────

def test_list_aliases_reflects_active_map(tmp_path, monkeypatch) -> None:
    from vecgrep.backend.aliases import describe_aliases

    p = tmp_path / "aliases.json"
    p.write_text(json.dumps({"alice": ["ali", "小艾"]}), encoding="utf-8")
    monkeypatch.setenv("VECGREP_ALIASES_FILE", str(p))
    d = describe_aliases()
    assert d["path"].endswith("aliases.json")
    assert d["entities"] == {"alice": ["ali", "小艾"]}
    assert d["active"] is True


def test_list_aliases_empty_when_no_map(tmp_path, monkeypatch) -> None:
    from vecgrep.backend.aliases import describe_aliases

    monkeypatch.setenv("VECGREP_ALIASES_FILE", str(tmp_path / "none.json"))
    d = describe_aliases()
    assert d["entities"] == {} and d["active"] is False


# ── MCP hygiene: service singleton + stdio parity ────────────────────────────

def test_mcp_service_is_reused_across_calls(vg_home, monkeypatch) -> None:
    from vecgrep.mcp import server as mcp_srv

    built = []
    real = mcp_srv.VecgrepService

    class Spy(real):  # type: ignore[misc]
        def __init__(self, *a, **k):
            built.append(1)
            super().__init__(*a, **k)

    monkeypatch.setattr(mcp_srv, "VecgrepService", Spy)
    mcp_srv._reset_service_cache()
    s1 = mcp_srv._svc()
    s2 = mcp_srv._svc()
    assert s1 is s2, "service must be cached across tool calls"
    assert len(built) == 1


def test_mcp_service_cache_resets_on_new_settings(vg_home, monkeypatch, tmp_path) -> None:
    from vecgrep.backend import config as cfg_mod
    from vecgrep.mcp import server as mcp_srv

    mcp_srv._reset_service_cache()
    s1 = mcp_srv._svc()
    # New settings generation (as tests / config reloads do) → new service.
    monkeypatch.setenv("VECGREP_HOME", str(tmp_path / "other"))
    cfg_mod._settings = None
    s2 = mcp_srv._svc()
    assert s1 is not s2, "stale service must not survive a settings reload"


def test_stdio_exposes_propose_tools_like_http() -> None:
    import asyncio

    import pytest

    pytest.importorskip("mcp")
    from mcp.types import ListToolsRequest

    from vecgrep.mcp.server import build_mcp_server

    server = build_mcp_server()
    handler = server.request_handlers[ListToolsRequest]
    result = asyncio.run(handler(ListToolsRequest(method="tools/list")))
    tools = {t.name for t in result.root.tools}
    assert {"propose_write", "propose_edit", "propose_delete"} <= tools
    assert {"browse", "get_source", "list_aliases"} <= tools
