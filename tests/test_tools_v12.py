"""v1.2 tools — related / compare / corpus_stats / summarize_corpus.

Service-layer contracts; the API/CLI/MCP registrations are thin wrappers over
these. Stub embeddings are deterministic-but-not-semantic, so `related`
asserts structural properties (self excluded, bounded, in-corpus), not
semantic neighbourliness.
"""
from __future__ import annotations

from tests import eval_harness as eh


def _built(svc):
    eh.build_eval_corpus(svc)
    return svc


# ── related ──────────────────────────────────────────────────────────────────

def test_related_excludes_self_and_bounds(svc) -> None:
    svc = _built(svc)
    seed = svc.search("relay-service", "evalchat", top_k=1)[0]
    rel = svc.related(seed.chunk_id, "evalchat", top_k=5)
    assert rel, "neighbours exist in a 55-chunk corpus"
    assert len(rel) <= 5
    assert all(r.chunk_id != seed.chunk_id for r in rel)
    assert all(r.corpus == "evalchat" for r in rel)


def test_related_unknown_chunk_raises(svc) -> None:
    svc = _built(svc)
    import pytest

    from vecgrep.backend.service import CorpusError
    with pytest.raises(CorpusError):
        svc.related("nope-not-a-chunk", "evalchat")


# ── compare ──────────────────────────────────────────────────────────────────

def test_compare_two_windows(svc) -> None:
    """Fixtures span 2026-01-08 and 2026-01-15 — split the windows between
    them and both sides populate, with per-side source sets and the delta."""
    svc = _built(svc)
    out = svc.compare(
        "relay-service", "evalchat",
        a_after="2026-01-01", a_before="2026-01-12",
        b_after="2026-01-12", b_before="2026-01-31",
        top_k=8,
    )
    assert out["a"]["results"] and out["b"]["results"]
    a_srcs = set(out["a"]["sources"])
    b_srcs = set(out["b"]["sources"])
    assert a_srcs and b_srcs and a_srcs.isdisjoint(b_srcs)
    assert set(out["only_in_b"]) == b_srcs - a_srcs
    assert out["windows"]["a"]["after"] == "2026-01-01"


def test_compare_empty_window_is_explicit(svc) -> None:
    svc = _built(svc)
    out = svc.compare(
        "relay-service", "evalchat",
        a_after="2020-01-01", a_before="2020-02-01",   # nothing there
        b_after="2026-01-01", b_before="2026-01-31",
    )
    assert out["a"]["results"] == []
    assert out["b"]["results"]


# ── corpus_stats ─────────────────────────────────────────────────────────────

def test_corpus_stats_counts_and_span(svc) -> None:
    svc = _built(svc)
    st = svc.corpus_stats("evalchat")
    assert st["corpus"] == "evalchat"
    assert st["chunks"] == 55
    assert st["docs"] == 4
    assert st["date_span"]["first"].startswith("2026-01-")
    assert st["date_span"]["last"].startswith("2026-01-")
    # fixtures are 01-08 and 01-15 per channel — the days between are gaps
    assert st["gap_days"] >= 1
    assert st["sources"]  # per-source chunk counts present


def test_corpus_stats_unknown_corpus_raises(svc) -> None:
    import pytest

    from vecgrep.backend.service import CorpusError
    with pytest.raises(CorpusError):
        svc.corpus_stats("nope")


# ── summarize_corpus ─────────────────────────────────────────────────────────

def test_summarize_corpus_rollup(svc) -> None:
    svc = _built(svc)
    out = svc.summarize_corpus("evalchat", sample=10)
    assert out["corpus"] == "evalchat"
    assert out["chunks"] == 55
    # enrichment-powered speaker tally (fixtures have alice/bob/ops-bot)
    speakers = dict(out["top_speakers"])
    assert "alice" in speakers
    assert out["date_span"]["first"]
    # sampling is explicit, never silent
    assert out["sampled"] is True
    assert len(out["sample_chunks"]) == 10
    assert all(isinstance(c, dict) and c.get("text") for c in out["sample_chunks"])


def test_summarize_corpus_window_filters(svc) -> None:
    svc = _built(svc)
    out = svc.summarize_corpus("evalchat", after="2026-01-12", sample=10)
    # only the 01-15 docs → fewer chunks than the whole corpus
    assert 0 < out["chunks"] < 55
