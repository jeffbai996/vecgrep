"""Cap what the cross-encoder scores, not what the search returns.

`_apply_rerank` used to score every fused candidate. Scoped to one corpus that
is CANDIDATE_POOL=50 pairs; unscoped it is 50 PER CORPUS, so a seven-corpus
fan-out sent ~350 pairs through the model and unscoped p50 went 873 -> 2936 ms
(round 4, 2026-09-10). The reranker only ever surfaces top_k of them, and a
candidate sitting 200th on fusion score is not a plausible winner -- scoring it
is latency spent on an answer nobody gets.

So: rerank the top RERANK_POOL_MAX by FUSION score, keep the rest below the
reranked head in fusion order. The tail is still returned (a filtered/deduped
result set must not shrink because of a latency cap); it just cannot jump the
queue on a score it was never given.
"""
from __future__ import annotations

import pytest

from vecgrep.backend import service as svc_mod
from vecgrep.backend.service import SearchResult


def _result(i: int, pct: float) -> SearchResult:
    return SearchResult(
        score=pct / 100.0, similarity_pct=pct, chunk=f"chunk text {i}",
        chunk_start=0, chunk_end=len(f"chunk text {i}"),
        context_before="", context_after="", source_id=f"s{i}", corpus="c",
        metadata={}, chunk_id=f"c{i}", matched_by=["vector"],
    )


@pytest.fixture
def candidates():
    # 120 candidates, fusion score DESCENDING with index, deliberately shuffled
    # into the arrival order a fan-out produces (corpus by corpus, not global).
    pool = [_result(i, 100.0 - i) for i in range(120)]
    return pool[60:] + pool[:60]


def test_only_the_top_pool_by_fusion_score_reaches_the_cross_encoder(
        svc, candidates, monkeypatch):
    seen: list[list[str]] = []

    def fake_rerank(query, pairs, model):
        seen.append([p[1].chunk_id for p in pairs])
        # score in reverse of arrival, so rerank order is observably not fusion
        return [(float(i), p[1]) for i, p in enumerate(pairs)]

    monkeypatch.setattr(svc_mod, "RERANK_POOL_MAX", 50)
    monkeypatch.setattr("vecgrep.backend.rerank.rerank", fake_rerank)
    svc._apply_rerank("q", list(candidates), 10, None)

    assert len(seen) == 1, "the cross-encoder must run exactly once"
    scored = seen[0]
    assert len(scored) == 50
    # the 50 highest fusion scores are c0..c49, whatever order they arrived in
    assert set(scored) == {f"c{i}" for i in range(50)}


def test_the_unscored_tail_is_still_returned_below_the_reranked_head(
        svc, candidates, monkeypatch):
    def fake_rerank(query, pairs, model):
        return [(float(i), p[1]) for i, p in enumerate(pairs)]

    monkeypatch.setattr(svc_mod, "RERANK_POOL_MAX", 50)
    monkeypatch.setattr("vecgrep.backend.rerank.rerank", fake_rerank)
    out = svc._apply_rerank("q", list(candidates), 80, None)

    ids = [r.chunk_id for r in out]
    # A latency cap must not shrink the result set the caller asked for.
    assert len(ids) == 80
    head, tail = ids[:50], ids[50:]
    assert set(head) <= {f"c{i}" for i in range(50)}, "head is the reranked pool"
    assert all(t not in head for t in tail)
    # the tail keeps fusion order: c50, c51, ... (descending fusion score)
    assert tail == [f"c{50 + i}" for i in range(len(tail))]


def test_the_tail_is_not_marked_reranked_and_keeps_its_fusion_score(
        svc, candidates, monkeypatch):
    def fake_rerank(query, pairs, model):
        return [(0.9, p[1]) for p in pairs]

    monkeypatch.setattr(svc_mod, "RERANK_POOL_MAX", 50)
    monkeypatch.setattr("vecgrep.backend.rerank.rerank", fake_rerank)
    out = svc._apply_rerank("q", list(candidates), 80, None)

    by_id = {r.chunk_id: r for r in out}
    tail = by_id["c60"]
    assert "rerank" not in tail.matched_by, \
        "a chunk the model never scored must not claim it was reranked"
    assert tail.similarity_pct == pytest.approx(40.0), \
        "unscored tail keeps its fusion score, not a rerank-calibrated one"
    assert "rerank_score" not in (tail.explain or {})
    head = by_id["c0"]
    assert "rerank" in head.matched_by


def test_a_pool_smaller_than_the_cap_is_reranked_whole(svc, monkeypatch):
    small = [_result(i, 100.0 - i) for i in range(12)]
    seen: list[int] = []

    def fake_rerank(query, pairs, model):
        seen.append(len(pairs))
        return [(float(len(pairs) - i), p[1]) for i, p in enumerate(pairs)]

    monkeypatch.setattr(svc_mod, "RERANK_POOL_MAX", 50)
    monkeypatch.setattr("vecgrep.backend.rerank.rerank", fake_rerank)
    out = svc._apply_rerank("q", small, 10, None)
    assert seen == [12], "no truncation when the pool already fits"
    assert all("rerank" in r.matched_by for r in out)


def test_cap_is_env_tunable_and_defaults_to_the_candidate_pool(svc):
    assert svc_mod.RERANK_POOL_MAX == svc_mod.CANDIDATE_POOL == 50


def test_a_zero_or_negative_cap_disables_truncation(svc, candidates, monkeypatch):
    """An operator turning the cap off must get the old whole-pool behaviour,
    not an empty rerank."""
    seen: list[int] = []

    def fake_rerank(query, pairs, model):
        seen.append(len(pairs))
        return [(float(i), p[1]) for i, p in enumerate(pairs)]

    monkeypatch.setattr(svc_mod, "RERANK_POOL_MAX", 0)
    monkeypatch.setattr("vecgrep.backend.rerank.rerank", fake_rerank)
    svc._apply_rerank("q", list(candidates), 10, None)
    assert seen == [120]
