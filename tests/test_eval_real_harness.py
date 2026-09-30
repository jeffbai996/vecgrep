"""vecgrep.eval — the side-by-side retrieval harness.

Hermetic (StubEmbed, embedded qdrant): these pin the harness's contract, not
retrieval quality. Quality numbers come from running it against real corpora
with a real gold set outside the repo.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vecgrep.eval import harness
from vecgrep.eval.gold import GoldCase, load_gold
from vecgrep.eval.metrics import score_case, summarize


# ── gold loading ─────────────────────────────────────────────────────────────

def test_gold_loads_native_and_legacy_shapes(tmp_path: Path) -> None:
    p = tmp_path / "gold.json"
    p.write_text(json.dumps({"cases": [
        {"id": "a", "corpus": "chats", "query": "x", "want": ["2026-08-07"], "tags": ["incident"]},
        {"q": "legacy ints", "want": [105, "12"], "note": "legacy style"},
        {"q": "legacy substr", "want_substr": ["2026-01-01"]},
        {"id": "neg", "corpus": "chats", "query": "nothing", "negative": True},
        "_comment string is ignored",
    ]}), encoding="utf-8")
    cases = load_gold(p, default_corpus="notes")
    assert [c.id for c in cases] == ["a", "notes-1", "notes-2", "neg"]
    assert cases[0].corpus == "chats" and cases[1].corpus == "notes"
    # bare ints expand to both entry kinds
    assert set(cases[1].want) == {"memory-105", "journal-105", "memory-12", "journal-12"}
    assert cases[2].want == ("2026-01-01",)
    assert cases[3].negative and cases[3].want == ()


def test_gold_rejects_case_with_neither_want_nor_negative(tmp_path: Path) -> None:
    p = tmp_path / "gold.json"
    p.write_text(json.dumps([{"query": "x", "corpus": "c"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_gold(p)


def test_entry_id_patterns_match_whole_stem_only() -> None:
    c = GoldCase(id="x", corpus="notes", query="q", want=("memory-1",))
    assert c.matches_want("/seed/notes/memory-1.md")
    assert not c.matches_want("/seed/notes/memory-105.md")
    assert not c.matches_want("/seed/notes/journal-1.md")
    d = GoldCase(id="y", corpus="chats", query="q", want=("2026-08-07",))
    assert d.matches_want("/transcripts/cl-3/2026-08-07.md")


# ── metrics ──────────────────────────────────────────────────────────────────

def _hit(sid: str, pct: float) -> dict:
    return {"source_id": sid, "similarity_pct": pct}


def test_scoring_is_source_level_and_dedups_chunks() -> None:
    c = GoldCase(id="a", corpus="chats", query="q", want=("day-2",))
    # day-1 appears as three chunks first; the wanted doc is the SECOND source
    hits = [_hit("day-1", 90), _hit("day-1", 88), _hit("day-1", 80), _hit("day-2", 70), _hit("day-3", 60)]
    r = score_case(c, hits, 0.05)
    assert r.sources == ["day-1", "day-2", "day-3"]
    assert r.first_hit_rank == 2
    assert r.hit_at(1) is False and r.hit_at(3) is True
    assert r.precision_at(3) == pytest.approx(1 / 3)
    assert r.top_pct == 90


def test_summary_rows_carry_negatives_and_latency() -> None:
    g = GoldCase(id="g", corpus="c", query="q", want=("want",))
    n = GoldCase(id="n", corpus="c", query="q", negative=True)
    f = GoldCase(id="f", corpus="c", query="q", want=("want",), forbid=("bad",))
    results = [
        score_case(g, [_hit("want", 80), _hit("x", 50), _hit("y", 40)], 0.010),
        score_case(n, [_hit("junk", 65)], 0.020),        # above the 60 floor => false positive
        score_case(f, [_hit("bad", 90), _hit("want", 70)], 0.030),
    ]
    s = summarize(results)
    assert s["n_graded"] == 2 and s["n_negative"] == 1
    assert s["hit@1"] == 50.0 and s["hit@3"] == 100.0
    assert s["mrr"] == pytest.approx(round((1 + 0.5) / 2, 3))
    assert s["neg_fp_rate"] == 100.0 and s["neg_top_pct_mean"] == 65.0
    assert s["forbid_leaks"] == 1
    assert s["misses"] == []
    assert s["lat_p50_ms"] == 20.0 and s["lat_p95_ms"] == 30.0
    assert s["spread_1v3"] == 40.0   # only g has >=3 sources: 80-40


# ── build / run / drop against a tiny live corpus ────────────────────────────

def _seed(tmp_path: Path) -> Path:
    root = tmp_path / "src"
    root.mkdir()
    (root / "2026-01-01.md").write_text(
        "---\nchannel: a\ndate: 2026-01-01\n---\n**alice** · 09:00\nthe relay service failed with ERR-0451 "
        "and we restarted it after the certificate renewal broke.\n", encoding="utf-8")
    (root / "2026-01-02.md").write_text(
        "---\nchannel: a\ndate: 2026-01-02\n---\n**bob** · 10:00\nplanning the sourdough bake, hydration at "
        "seventy five percent, long cold proof overnight.\n", encoding="utf-8")
    return root


def test_build_side_by_side_copies_live_corpus_and_pins_model(svc, tmp_path: Path) -> None:
    root = _seed(tmp_path)
    svc.index(str(root), "chats")
    live = svc.registry.get("chats")
    rep = harness.build_side_by_side(svc, "chats", "eval-chats-a")
    dst = svc.registry.get("eval-chats-a")
    assert rep.docs == 2 and rep.points == svc.store.count("vecgrep__eval-chats-a")
    assert dst.embed_model == live.embed_model and dst.chunker == live.chunker
    assert dst.decay_half_life_days == live.decay_half_life_days
    # the live corpus is untouched
    assert svc.registry.get("chats").chunk_count == live.chunk_count
    with pytest.raises(ValueError):
        harness.build_side_by_side(svc, "chats", "eval-chats-a")   # exists, no --force
    with pytest.raises(ValueError):
        harness.build_side_by_side(svc, "chats", "chats-copy")     # not eval-*


def test_run_config_maps_logical_corpus_and_reports(svc, tmp_path: Path) -> None:
    root = _seed(tmp_path)
    svc.index(str(root), "chats")
    harness.build_side_by_side(svc, "chats", "eval-chats-a")
    cases = [
        GoldCase(id="relay", corpus="chats", query="ERR-0451 relay failure", want=("2026-01-01",)),
        GoldCase(id="bread", corpus="chats", query="sourdough hydration", want=("2026-01-02",)),
        GoldCase(id="neg", corpus="chats", query="quarterly tax filing deadline", negative=True),
    ]
    cfg = harness.RunConfig(name="t", corpora={"chats": "eval-chats-a"}, mode="hybrid", top_k=5,
                            bm25_weight=2.0, rrf_k=30)
    from vecgrep.backend import service as sm
    before = (sm.BM25_WEIGHT, sm.RRF_K)
    r = harness.run_config(svc, cfg, cases, warmup=False)
    assert (sm.BM25_WEIGHT, sm.RRF_K) == before, "ranking constants must be restored"
    assert r["summary"]["n_graded"] == 2 and r["summary"]["n_negative"] == 1
    assert r["summary"]["hit@3"] == 100.0
    assert "eval-chats-a" in r["storage"]
    assert r["storage"]["eval-chats-a"]["points"] > 0
    ids = {c["id"] for c in r["cases"]}
    assert ids == {"relay", "bread", "neg"}
    md = harness.table([r])
    assert md.startswith("| config |") and "| t |" in md


def test_drop_only_touches_eval_corpora(svc, tmp_path: Path) -> None:
    root = _seed(tmp_path)
    svc.index(str(root), "chats")
    harness.build_side_by_side(svc, "chats", "eval-chats-a")
    harness.build_side_by_side(svc, "chats", "eval-chats-b")
    dropped = harness.drop_eval_corpora(svc, ["eval-chats-b"])
    assert dropped == ["eval-chats-b"]
    assert svc.registry.has("eval-chats-a") and svc.registry.has("chats")
    assert harness.drop_eval_corpora(svc) == ["eval-chats-a"]
    assert svc.registry.has("chats")


# ── store options the harness can build with ────────────────────────────────

def test_ensure_collection_pins_datatype_and_quantization(svc) -> None:
    from qdrant_client import models as qm
    svc.store.ensure_collection("vecgrep__eval-f16", 8, datatype="float16", quantization="int8")
    info = svc.store.client.get_collection("vecgrep__eval-f16")
    assert info.config.params.vectors.datatype == qm.Datatype.FLOAT16
    # idempotent: a second call with different options must NOT recreate
    svc.store.ensure_collection("vecgrep__eval-f16", 8)
    assert svc.store.client.get_collection("vecgrep__eval-f16").config.params.vectors.datatype == qm.Datatype.FLOAT16
    with pytest.raises(ValueError):
        svc.store.ensure_collection("vecgrep__eval-bad", 8, datatype="bfloat16")
    with pytest.raises(ValueError):
        svc.store.ensure_collection("vecgrep__eval-bad2", 8, quantization="pq")


def test_build_side_by_side_passes_datatype_through(svc, tmp_path: Path) -> None:
    from qdrant_client import models as qm
    root = _seed(tmp_path)
    svc.index(str(root), "chats")
    rep = harness.build_side_by_side(svc, "chats", "eval-chats-f16", datatype="float16")
    assert rep.options["datatype"] == "float16"
    info = svc.store.client.get_collection("vecgrep__eval-chats-f16")
    assert info.config.params.vectors.datatype == qm.Datatype.FLOAT16
    assert rep.points > 0


def test_unscoped_run_searches_every_corpus_not_the_mapped_one(svc, tmp_path: Path) -> None:
    """`unscoped=True` measures the DEFAULT bot call: no corpus argument, so
    the service fans out and fuses. Without it the harness can only ever score
    a scoped search, which is not the path the MCP rerank gate governs."""
    root = _seed(tmp_path)
    svc.index(str(root), "chats")
    other = tmp_path / "other"
    other.mkdir()
    (other / "note.md").write_text(
        "---\nchannel: b\ndate: 2026-01-03\n---\n**carol** · 09:00\n"
        "unrelated filler about garden mulch and compost turning.\n", encoding="utf-8")
    svc.index(str(other), "notes")

    cases = [GoldCase(id="relay", corpus="chats", query="ERR-0451 relay failure",
                      want=("2026-01-01",))]
    seen: list[dict] = []
    real_search = svc.search

    def spy(query, corpus_name=None, **kw):
        seen.append({"corpus_name": corpus_name, "kw": kw})
        return real_search(query, corpus_name, **kw)

    svc.search = spy  # type: ignore[method-assign]
    try:
        cfg = harness.RunConfig(name="u", mode="hybrid", top_k=5, unscoped=True)
        r = harness.run_config(svc, cfg, cases, warmup=False)
    finally:
        svc.search = real_search  # type: ignore[method-assign]

    assert seen, "harness never called search"
    assert all(s["corpus_name"] is None for s in seen), \
        "unscoped run must pass corpus_name=None so the service fans out"
    assert r["summary"]["n_graded"] == 1
    # storage is still reported, keyed by the corpora actually searched
    assert "chats" in r["storage"] and "notes" in r["storage"]


def test_scoped_run_still_passes_the_mapped_corpus(svc, tmp_path: Path) -> None:
    root = _seed(tmp_path)
    svc.index(str(root), "chats")
    cases = [GoldCase(id="relay", corpus="chats", query="ERR-0451 relay failure",
                      want=("2026-01-01",))]
    seen: list[str | None] = []
    real_search = svc.search

    def spy(query, corpus_name=None, **kw):
        seen.append(corpus_name)
        return real_search(query, corpus_name, **kw)

    svc.search = spy  # type: ignore[method-assign]
    try:
        harness.run_config(svc, harness.RunConfig(name="s", top_k=5), cases, warmup=False)
    finally:
        svc.search = real_search  # type: ignore[method-assign]
    assert seen == ["chats"]
