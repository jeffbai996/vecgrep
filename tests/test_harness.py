from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests import eval_harness as eh
from vecgrep.eval import harness
from vecgrep.eval.gold import GoldCase, load_gold
from vecgrep.eval.metrics import score_case, summarize

# ── test_eval_real_harness ──

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

    svc.search = spy
    try:
        cfg = harness.RunConfig(name="u", mode="hybrid", top_k=5, unscoped=True)
        r = harness.run_config(svc, cfg, cases, warmup=False)
    finally:
        svc.search = real_search

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

    svc.search = spy
    try:
        harness.run_config(svc, harness.RunConfig(name="s", top_k=5), cases, warmup=False)
    finally:
        svc.search = real_search
    assert seen == ["chats"]

# ── test_eval_harness ──

FIXTURES = Path(__file__).parent / "fixtures" / "transcripts"
BASELINE = Path(__file__).parent / "fixtures" / "baselines" / "v0_7_0.json"


# ── fixtures: shape and hygiene ──────────────────────────────────────────────

def test_fixture_files_exist_and_mimic_transcript_shape() -> None:
    files = sorted(FIXTURES.rglob("*.md"))
    assert len(files) >= 4, "need at least 4 transcript fixtures"
    for f in files:
        text = f.read_text(encoding="utf-8")
        # Real transcript shadows carry frontmatter with channel + date; the
        # markdown adapter lifts these into metadata and doc_timestamp.
        assert text.startswith("---"), f"{f.name}: missing frontmatter"
        assert "channel:" in text.splitlines()[1], f"{f.name}: no channel key"
        assert "date:" in text, f"{f.name}: no date key"
        # Speaker lines in the archiver's format: **name** · HH:MM
        assert "** · " in text, f"{f.name}: no speaker/timestamp lines"


def test_fixtures_are_bilingual_and_alias_bearing() -> None:
    corpus_text = "".join(
        f.read_text(encoding="utf-8") for f in FIXTURES.rglob("*.md")
    )
    # Bilingual: CJK content present.
    assert any("一" <= ch <= "鿿" for ch in corpus_text), "no CJK text"
    # The alias cluster (one person, three surface forms) is present so the
    # Phase 6 alias-expansion eval has something to find.
    for form in ("alice", "ali", "小艾"):
        assert form in corpus_text, f"alias form {form!r} missing"
    # Exact-keyword anchors for keyword-precision queries.
    assert "ERR-0451" in corpus_text
    assert "TimeoutError in worker-3" in corpus_text


def test_fixtures_contain_no_private_data() -> None:
    """Public-mirror guard: fixtures must stay generic. Structural checks only
    — naming real private tokens here would itself leak them into the public
    mirror (the pre-commit PII hook is the real gate and rightly blocks that).
    Operators can point VECGREP_EVAL_DENYLIST at an out-of-repo file with one
    forbidden token per line for a personalized sweep."""
    import os

    corpus_text = "".join(
        f.read_text(encoding="utf-8").lower() for f in FIXTURES.rglob("*.md")
    )
    import re

    assert not re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", corpus_text), "email in fixtures"
    assert not re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", corpus_text), "IP address in fixtures"
    assert not re.search(r"\b[a-f0-9]{32,}\b", corpus_text), "token-like hex in fixtures"

    denylist_path = os.environ.get("VECGREP_EVAL_DENYLIST", "")
    if denylist_path and Path(denylist_path).is_file():
        for line in Path(denylist_path).read_text(encoding="utf-8").splitlines():
            token = line.strip().lower()
            if token and not token.startswith("#"):
                assert token not in corpus_text, "denylisted token in fixtures"


# ── harness: indexing + golden queries ───────────────────────────────────────

def test_build_eval_corpus_indexes_all_fixtures(svc) -> None:
    stats = eh.build_eval_corpus(svc)
    assert stats["files"] == len(list(FIXTURES.rglob("*.md")))
    assert stats["chunks"] > stats["files"], "each file should chunk"


def test_golden_queries_have_required_fields() -> None:
    assert len(eh.GOLDEN_QUERIES) >= 6
    ids = [q.id for q in eh.GOLDEN_QUERIES]
    assert len(ids) == len(set(ids)), "duplicate query ids"
    for q in eh.GOLDEN_QUERIES:
        assert q.query, f"{q.id}: empty query"
        assert q.expected_sources or q.kind == "probe", (
            f"{q.id}: non-probe queries need expected sources"
        )


def test_run_eval_produces_metrics_for_every_query(svc) -> None:
    eh.build_eval_corpus(svc)
    report = eh.run_eval(svc)
    assert set(report["queries"].keys()) == {q.id for q in eh.GOLDEN_QUERIES}
    for qid, m in report["queries"].items():
        assert m["n_results"] >= 0
        assert 0.0 <= m["redundancy"] <= 1.0, f"{qid}: redundancy out of range"
        assert 0.0 <= m["expected_recall"] <= 1.0


# ── the baseline captures the PROBLEMS the release fixes ─────────────────────

def test_v070_baseline_recorded_the_duplicate_problem() -> None:
    """Historical record: the committed v0.7.0 baseline captured near-identical
    repeated content surviving span-overlap dedup (the Phase 1 target). Live
    behavior has since improved — tests/test_mmr.py asserts the improvement;
    this asserts the problem was real and measured."""
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert data["queries"]["dup_stress"]["redundancy"] > 0.0


def test_baseline_shows_date_leakage_today(svc) -> None:
    """The 'today' incident query drags in the week-old trap file at baseline
    (no date filters yet). in_window_fraction < 1.0 proves the leak Phase 3
    closes."""
    eh.build_eval_corpus(svc)
    report = eh.run_eval(svc)
    m = report["queries"]["incident_today"]
    assert m["forbidden_hits"] > 0, "trap file should leak at baseline"


def test_baseline_file_exists_and_matches_schema() -> None:
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert data["version"] == "0.7.0"
    assert "queries" in data
    for qid, m in data["queries"].items():
        assert "n_results" in m and "redundancy" in m, f"{qid}: bad schema"


def test_build_side_by_side_rejects_unknown_decay_sentinel(svc, make_doc) -> None:
    from typing import Any, cast

    import pytest

    from vecgrep.eval.harness import build_side_by_side

    p = make_doc("doc.md", "Alpha beta gamma.")
    svc.index(str(p), "live")
    with pytest.raises(ValueError, match="decay_half_life_days"):
        build_side_by_side(
            svc, "live", "eval-bad-decay", decay_half_life_days=cast(Any, "none")
        )
