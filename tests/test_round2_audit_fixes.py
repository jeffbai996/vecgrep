"""Regression tests for Round 2 scaling & accuracy audit fixes (Phases 5-7)."""
from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client.http import models as qm

from vecgrep.backend.ingestion.adapters import detect_adapter
from vecgrep.backend.service import (
    SearchResult,
    _build_qdrant_filter,
    _expand,
    _journal_corpus,
    _source_exists,
    _source_path_aliases,
)
from vecgrep.backend.store.corpora import Corpus
from vecgrep.backend.symbol_query import pin_definitions


def _make_result(
    *,
    chunk_id: str,
    score: float,
    similarity_pct: float,
    chunk: str = "class Foo:\n    pass",
    corpus: str = "c1",
    source_id: str = "a.py",
    metadata: dict | None = None,
) -> SearchResult:
    return SearchResult(
        score=score,
        similarity_pct=similarity_pct,
        chunk=chunk,
        chunk_start=0,
        chunk_end=len(chunk),
        context_before="",
        context_after="",
        source_id=source_id,
        corpus=corpus,
        metadata=metadata or {},
        chunk_id=chunk_id,
        matched_by=["vector"],
    )


def test_pin_definitions_tie_breaks_by_score_not_corpus_order() -> None:
    low_def = _make_result(
        chunk_id="low",
        score=0.12,
        similarity_pct=55.0,
        corpus="alpha_old",
        source_id="old.py",
        metadata={"qualname": "Foo", "symbol": "Foo"},
    )
    high_def = _make_result(
        chunk_id="high",
        score=0.88,
        similarity_pct=92.0,
        corpus="zeta_new",
        source_id="new.py",
        metadata={"qualname": "Foo", "symbol": "Foo"},
    )
    mention = _make_result(
        chunk_id="mention",
        score=0.95,
        similarity_pct=96.0,
        chunk="x = Foo()",
        corpus="alpha_old",
        source_id="caller.py",
        metadata={"qualname": "caller", "symbol": "caller"},
    )
    pool = [low_def, mention, high_def]
    pinned = pin_definitions([mention, low_def], pool, "Foo", top_k=2)
    assert [r.chunk_id for r in pinned] == ["high", "low"]


def test_build_qdrant_filter_uses_match_except_for_active_status() -> None:
    flt, has_unpushed = _build_qdrant_filter(["meta.status=active"])
    assert has_unpushed is False
    assert isinstance(flt, qm.Filter)
    assert flt.must is None
    assert flt.must_not is not None and len(flt.must_not) == 1
    cond = flt.must_not[0]
    assert isinstance(cond, qm.FieldCondition)
    assert cond.key == "metadata.status"
    assert isinstance(cond.match, qm.MatchExcept)
    assert list(cond.match.except_) == ["active"]


def test_build_qdrant_filter_pushes_negated_after_and_before() -> None:
    flt, has_unpushed = _build_qdrant_filter(
        ["-after:2026-01-01", "-before:2026-02-01"]
    )
    assert has_unpushed is False
    assert isinstance(flt, qm.Filter)
    assert flt.must_not is not None and len(flt.must_not) == 2


def test_expand_prunes_ignored_walk_dirs(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config.md").write_text("# git internal\n", encoding="utf-8")
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "readme.md").write_text(
        "# dep readme\n", encoding="utf-8"
    )
    (tmp_path / "src").mkdir()
    real_doc = tmp_path / "src" / "guide.md"
    real_doc.write_text("# Real Guide\nHello world.\n", encoding="utf-8")

    docs = _expand(str(tmp_path), adapter=None)
    assert [Path(d.source_id).name for d in docs] == ["guide.md"]


def test_source_exists_and_aliases_handle_fragment_ids(tmp_path: Path) -> None:
    export_file = tmp_path / "chats.json"
    export_file.write_text("{}", encoding="utf-8")
    s1 = f"{export_file}#conv-1"
    s2 = f"{export_file}#conv-2"
    assert _source_exists(s1) is True
    assert _source_exists(f"{tmp_path / 'missing.json'}#conv-1") is False
    assert _source_path_aliases([s1, s2]) == []


def test_journal_corpus_omits_sources_without_mutating_lists() -> None:
    c = Corpus(
        name="demo",
        embed_backend="ollama",
        embed_model="bge-m3",
        dim=1024,
        chunker="markdown",
        created_at=1.0,
        updated_at=2.0,
        sources=["a.md", "b.md"],
        source_hashes={"a.md": "h1", "b.md": "h2"},
        use_for=["hint1"],
    )
    journaled = _journal_corpus(c)
    assert journaled is not None
    assert journaled["sources"] == []
    assert journaled["source_hashes"] == {}
    assert journaled["use_for"] == ["hint1"]
    journaled["use_for"].append("mutated")
    assert c.use_for == ["hint1"]


def test_throttle_client_key_ignores_spoofed_xff_from_untrusted_peer() -> None:
    from vecgrep.backend.auth import throttle

    k1 = throttle.client_key(
        {
            "type": "http",
            "headers": [(b"x-forwarded-for", b"1.1.1.1")],
            "client": ("10.0.0.99", 1234),
        }
    )
    k2 = throttle.client_key(
        {
            "type": "http",
            "headers": [(b"x-forwarded-for", b"2.2.2.2")],
            "client": ("10.0.0.99", 5678),
        }
    )
    assert k1 == k2


def test_bm25_sqlite_pushes_down_over_200_source_ids_via_json_each(
    tmp_path: Path,
) -> None:
    from vecgrep.backend.store.bm25_sqlite import BM25SqliteStore

    store = BM25SqliteStore(tmp_path)
    ids = [f"id-{i}" for i in range(250)]
    texts = [f"needle token in doc {i}" for i in range(250)]
    payloads = [
        {"corpus": "big", "source_id": f"src-{i}.md", "text": texts[i]}
        for i in range(250)
    ]
    store.upsert("big", ids, texts, payloads)

    allowed = {f"src-{i}.md" for i in range(210)}
    hits = store.search("big", "needle", top_k=10, source_ids=allowed)
    assert len(hits) == 10
    assert all(pl["source_id"] in allowed for _, _, pl in hits)


def test_qdrant_payload_index_fields_include_common_meta_keys() -> None:
    from vecgrep.backend.store.qdrant_store import _PAYLOAD_INDEX_FIELDS

    indexed = {name for name, _ in _PAYLOAD_INDEX_FIELDS}
    for expected in (
        "metadata.source_kind",
        "metadata.tier",
        "metadata.adapter",
        "metadata.tags",
    ):
        assert expected in indexed


def test_mcp_read_only_tools_use_concurrent_read_gate_while_writes_stay_serialized() -> None:
    import asyncio
    import time
    from mcp.server.fastmcp import FastMCP
    from vecgrep.mcp import server as mcp_server

    fmcp = FastMCP("test-gates")
    mcp_server._offload_sync_tools(fmcp)
    read_live = {"now": 0, "peak": 0}
    write_live = {"now": 0, "peak": 0}

    @fmcp.tool(description="read tool")
    def search() -> str:
        read_live["now"] += 1
        read_live["peak"] = max(read_live["peak"], read_live["now"])
        time.sleep(0.08)
        read_live["now"] -= 1
        return "ok"

    @fmcp.tool(description="write tool")
    def write() -> str:
        write_live["now"] += 1
        write_live["peak"] = max(write_live["peak"], write_live["now"])
        time.sleep(0.08)
        write_live["now"] -= 1
        return "ok"

    async def drive() -> None:
        await asyncio.gather(*(fmcp._tool_manager.call_tool("search", {}) for _ in range(4)))
        await asyncio.gather(*(fmcp._tool_manager.call_tool("write", {}) for _ in range(3)))

    asyncio.run(drive())
    assert read_live["peak"] >= 2, "read-only MCP tools should share the wider read gate"
    assert write_live["peak"] == 1, "write MCP tools must stay serialized"


def test_incident_includes_primary_slice_for_non_transcript_sources(
    svc: VecgrepService, make_doc
) -> None:
    p = make_doc(
        "runbook.md",
        "# Gateway Outage Postmortem\n\n"
        "The gateway failed when the upstream TLS certificate expired at midnight.\n"
        "Rotating the certificate restored traffic across all regions.\n",
    )
    svc.index(str(p), "docs")
    inc = svc.incident("gateway TLS certificate outage", "docs")
    assert inc is not None
    assert inc["primary_timeline"] == []
    assert "TLS certificate" in inc["primary_slice"]


def test_pin_definitions_orders_reranked_candidates_ahead_of_unscored_tail() -> None:
    from vecgrep.backend.service import SearchResult
    from vecgrep.backend.symbol_query import pin_definitions

    def _mk(cid: str, score: float, matched_by: list[str]) -> SearchResult:
        return SearchResult(
            score=score,
            similarity_pct=50.0,
            chunk=f"def target_fn(): pass #{cid}",
            chunk_start=0,
            chunk_end=20,
            context_before="",
            context_after="",
            source_id=f"{cid}.py",
            corpus="code",
            metadata={"symbol": "target_fn", "qualname": f"Mod.{cid}.target_fn"},
            chunk_id=cid,
            matched_by=matched_by,
        )

    reranked_def = _mk("reranked", 0.01, ["bm25", "rerank"])
    unscored_tail_def = _mk("tail", 0.035, ["bm25"])
    out = pin_definitions(
        [reranked_def],
        [unscored_tail_def, reranked_def],
        "target_fn",
        top_k=2,
    )
    assert [r.chunk_id for r in out] == ["reranked", "tail"]


def test_payload_to_filter_stub_matches_passes_filters_without_uuid(monkeypatch) -> None:
    import vecgrep.backend.service as svc_mod
    from vecgrep.backend.service import _passes_filters, _payload_to_filter_stub

    def _forbid_chunk_id(*args, **kwargs):
        raise AssertionError("_chunk_id must not be called for filter stubs")

    monkeypatch.setattr(svc_mod, "_chunk_id", _forbid_chunk_id)
    payload = {
        "source_id": "notes/alpha.md",
        "corpus": "notes",
        "doc_timestamp": 1767225600.0,
        "metadata": {"status": "active", "project": "vecgrep"},
    }
    stub = _payload_to_filter_stub(payload)
    assert _passes_filters(
        stub,
        ["source:notes/*.md", "corpus:notes", "meta.project=vecgrep"],
        default_active=True,
    )
    assert not _passes_filters(
        stub,
        ["meta.project=other"],
        default_active=True,
    )


def test_eval_run_config_clears_rerank_score_cache(
    svc: VecgrepService, make_doc
) -> None:
    from vecgrep.backend import rerank as rerank_mod
    from vecgrep.eval import harness
    from vecgrep.eval.gold import GoldCase

    p = make_doc("doc.md", "Alpha beta gamma delta epsilon.")
    svc.index(str(p), "evaltest")
    rerank_mod._score_cache[("model", "q", "chunk")] = 0.99
    assert len(rerank_mod._score_cache) == 1

    cfg = harness.RunConfig(name="test", top_k=3, rerank=False)
    cases = [GoldCase(id="c1", corpus="evaltest", query="alpha", want=[str(p)])]
    harness.run_config(svc, cfg, cases, warmup=False, repeat=1)
    assert len(rerank_mod._score_cache) == 0

