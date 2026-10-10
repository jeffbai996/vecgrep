"""Corpus rank weight and recency decay survive the reranker when asked to.

Both were applied only to the fused score, which picks the reranker's pool. The
cross-encoder then re-sorted on text alone, so `squad-store` at 1.25 and a
45-day decay on `chats` changed which candidates were scored and nothing about
the order the caller saw. On the 2026-10-10 eval, 9 of 11 squad-store misses
were the curated memory losing to an old transcript of the conversation that
produced it.

With settings.rerank_prior on, the final order is the calibrated reranker
probability times the same weight x recency multiplier. The displayed
similarity_pct stays the reranker's own number: downstream relevance floors
were tuned on it.
"""
from __future__ import annotations

import time

import pytest

from vecgrep.backend import service as svc_mod
from vecgrep.backend.service import SearchResult

DAY = 86_400.0


def _hit(cid: str, corpus: str, ts: float | None = None) -> SearchResult:
    return SearchResult(
        score=0.5, similarity_pct=50.0, chunk=f"text {cid}", chunk_start=0,
        chunk_end=6, context_before="", context_after="", source_id=f"/{corpus}/{cid}.md",
        corpus=corpus, metadata={}, chunk_id=cid, matched_by=["vector"], doc_timestamp=ts,
    )


@pytest.fixture
def corpora(svc, make_doc):
    for name in ("notes", "chats"):
        svc.index(str(make_doc(f"{name}.md", "alpha beta gamma.")), name)
    svc.set_rank_weight("notes", 1.25)
    svc.set_decay("chats", 45.0)
    return svc


def _scores(by_id: dict[str, float]):
    """A fake cross-encoder that returns raw scores, sorted like the real one."""
    def fake(query, pairs, model):
        scored = [(by_id[p[1].chunk_id], p[1]) for p in pairs]
        return sorted(scored, key=lambda s: s[0], reverse=True)
    return fake


def _order(svc, hits, by_id, monkeypatch):
    monkeypatch.setattr("vecgrep.backend.rerank.rerank", _scores(by_id))
    return [r.chunk_id for r in svc._apply_rerank("q", hits, 10, None)]


def test_the_prior_is_off_by_default(corpora, monkeypatch):
    assert corpora.settings.rerank_prior is False
    hits = [_hit("chat", "chats", time.time()), _hit("note", "notes")]
    assert _order(corpora, hits, {"chat": 0.60, "note": 0.58}, monkeypatch) == ["chat", "note"]


def test_a_weighted_corpus_wins_a_close_call(corpora, monkeypatch):
    monkeypatch.setattr(corpora.settings, "rerank_prior", True)
    hits = [_hit("chat", "chats", time.time()), _hit("note", "notes")]
    assert _order(corpora, hits, {"chat": 0.60, "note": 0.58}, monkeypatch) == ["note", "chat"]


def test_the_weight_does_not_overturn_a_clear_reranker_verdict(corpora, monkeypatch):
    monkeypatch.setattr(corpora.settings, "rerank_prior", True)
    hits = [_hit("chat", "chats", time.time()), _hit("note", "notes")]
    assert _order(corpora, hits, {"chat": 0.80, "note": 0.40}, monkeypatch) == ["chat", "note"]


def test_an_old_transcript_yields_to_a_fresh_one_of_equal_relevance(corpora, monkeypatch):
    monkeypatch.setattr(corpora.settings, "rerank_prior", True)
    now = time.time()
    hits = [_hit("old", "chats", now - 120 * DAY), _hit("new", "chats", now - 1 * DAY)]
    assert _order(corpora, hits, {"old": 0.60, "new": 0.59}, monkeypatch) == ["new", "old"]


def test_the_displayed_percent_stays_the_reranker_value(corpora, monkeypatch):
    monkeypatch.setattr(corpora.settings, "rerank_prior", True)
    monkeypatch.setattr("vecgrep.backend.rerank.rerank", _scores({"note": 0.58}))
    (r,) = corpora._apply_rerank("q", [_hit("note", "notes")], 10, None)
    assert r.similarity_pct == pytest.approx(svc_mod._rerank_to_pct(0.58))
    assert r.explain["rerank_prior"] == pytest.approx(1.25)


def test_rerank_prior_is_an_editable_setting():
    from vecgrep.backend import config
    assert "rerank_prior" in config.EDITABLE_FIELDS
