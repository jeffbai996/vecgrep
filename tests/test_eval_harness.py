"""Phase 0 — eval harness self-tests.

The eval harness (tests/eval_harness.py) is the measuring stick for the whole
memory-v1 release: synthetic Discord-style transcript fixtures + golden
benchmark queries + assembly-quality metrics. These tests pin the harness's
own contract so later phases can trust its numbers.

Everything is hermetic: StubEmbed vectors + BM25, no live embedder. That means
the harness measures RESULT ASSEMBLY (dedup, filters, chronology, budget) —
not embedding quality, which a stub can't represent.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests import eval_harness as eh

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
