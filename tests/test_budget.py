"""Phase 2 — result budget + stub tier.

Breadth without blowing the context budget: the top `full_k` results keep
their context windows and full scores; everything after them degrades to a
one-line stub (source + timestamp + snippet + score + chunk_id) emitted until
a token ceiling is hit. The assistant re-queries any stub's full context via
its chunk_id (`/api/chunk`, MCP `get_chunk`).
"""
from __future__ import annotations

from pathlib import Path

from vecgrep.backend.assembly import (
    ResultStub,
    estimate_tokens,
    split_full_and_stubs,
)
from vecgrep.backend.service import SearchResult

LONG_CHUNK = (
    "this chunk is long enough to cost a meaningful number of tokens in the "
    "stub budget accounting so ceilings actually bite during the test run"
)


def _hit(i: int, score: float = 0.5, chunk: str = LONG_CHUNK) -> SearchResult:
    return SearchResult(
        score=score,
        similarity_pct=score * 100,
        chunk=f"[{i}] {chunk}",
        chunk_start=i * 200,
        chunk_end=i * 200 + 150,
        context_before="ctx-before " * 30,
        context_after="ctx-after " * 30,
        source_id=f"channel-x/2026-01-{(i % 28) + 1:02d}.md",
        corpus="c",
        metadata={},
        chunk_id=f"cid-{i}",
        matched_by=["vector"],
        doc_timestamp=1_700_000_000.0 + i,
    )


# ── token estimation ─────────────────────────────────────────────────────────

def test_estimate_tokens_scales_with_length() -> None:
    assert estimate_tokens("") == 0
    short, long = estimate_tokens("word " * 10), estimate_tokens("word " * 1000)
    assert 0 < short < long
    # ~4 chars/token heuristic: 5000 chars ≈ 1250 tokens, sanity band only.
    assert 800 < long < 2000


# ── split_full_and_stubs ─────────────────────────────────────────────────────

def test_split_gives_full_then_stubs_capped_at_max_total() -> None:
    results = [_hit(i, score=1.0 - i * 0.01) for i in range(100)]
    full, stubs = split_full_and_stubs(results, full_k=8, max_total=80)
    assert len(full) == 8
    assert len(stubs) <= 72
    assert len(full) + len(stubs) <= 80
    # Order preserved: full tier is the head of the ranking.
    assert [f.chunk_id for f in full] == [r.chunk_id for r in results[:8]]


def test_split_stub_shape() -> None:
    results = [_hit(i) for i in range(12)]
    _full, stubs = split_full_and_stubs(results, full_k=8, max_total=80)
    s: ResultStub = stubs[0]
    assert s.chunk_id == "cid-8"
    assert s.source_id.endswith(".md")
    assert s.doc_timestamp is not None
    assert "\n" not in s.snippet, "stub snippet must be one line"
    assert len(s.snippet) <= 200
    assert s.score > 0 and s.similarity_pct > 0
    # Stubs must NOT carry context windows — that's the whole point.
    assert not hasattr(s, "context_before")


def test_split_token_ceiling_bites() -> None:
    results = [_hit(i) for i in range(80)]
    _f_uncapped, stubs_uncapped = split_full_and_stubs(
        results, full_k=8, max_total=80, token_ceiling=10_000_000
    )
    _f_capped, stubs_capped = split_full_and_stubs(
        results, full_k=8, max_total=80, token_ceiling=200
    )
    assert len(stubs_capped) < len(stubs_uncapped)
    assert stubs_capped, "a small ceiling should still emit some stubs"
    total = sum(estimate_tokens(s.snippet + s.source_id) for s in stubs_capped)
    assert total <= 200 + estimate_tokens(
        stubs_capped[-1].snippet + stubs_capped[-1].source_id
    ), "ceiling may be crossed by at most the final stub"


def test_split_fewer_results_than_full_k() -> None:
    results = [_hit(i) for i in range(3)]
    full, stubs = split_full_and_stubs(results, full_k=8, max_total=80)
    assert len(full) == 3 and stubs == []


def test_split_empty() -> None:
    assert split_full_and_stubs([], full_k=8, max_total=80) == ([], [])


# ── doc_timestamp now rides on SearchResult ──────────────────────────────────

def test_search_results_carry_doc_timestamp(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    hits = svc.search("relay-service", "evalchat", top_k=5)
    assert hits, "fixtures must match"
    dated = [h for h in hits if h.doc_timestamp is not None]
    assert dated, "frontmatter date: must populate doc_timestamp"
    # 2026-01-xx fixture dates land in a sane epoch range.
    assert all(1_700_000_000 < h.doc_timestamp < 1_900_000_000 for h in dated)


# ── service.search_budgeted end-to-end ───────────────────────────────────────

def test_search_budgeted_returns_full_and_stub_tiers(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    full, stubs = svc.search_budgeted("relay-service ERR-0451", "evalchat", full_k=8)
    assert 0 < len(full) <= 8
    assert isinstance(stubs, list)
    if stubs:
        assert isinstance(stubs[0], ResultStub)
        # The stub tail continues the ranking below the full tier.
        assert stubs[0].score <= full[-1].score
    # 55 fixture chunks exist; breadth means we surface well beyond top-10.
    assert len(full) + len(stubs) > 10


def test_search_budgeted_widens_candidate_pool(svc, monkeypatch) -> None:
    """A budgeted call must not be silently capped by CANDIDATE_POOL=50 when
    max_total exceeds it: the per-corpus pool must widen to top_k. (Retrieval
    still only returns what actually MATCHES — the widened pool is a ceiling,
    not a corpus dump — so this pins the mechanism, not result counts.)"""
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    seen: list[int] = []
    orig = type(svc)._search_one

    def spy(self, corpus, query, top_k, mode, explain=False):
        seen.append(top_k)
        return orig(self, corpus, query, top_k, mode, explain=explain)

    monkeypatch.setattr(type(svc), "_search_one", spy)
    svc.search_budgeted("relay-service", "evalchat", max_total=80)
    assert seen and seen[0] >= 80, f"pool not widened: {seen}"
    seen.clear()
    svc.search("relay-service", "evalchat", top_k=5)
    assert seen and seen[0] == 50, "normal searches keep the standard pool"


def test_search_budgeted_stub_chunk_ids_resolve(svc) -> None:
    """Every stub's chunk_id must round-trip through get_chunk_window — the
    re-query path an assistant uses to expand a stub."""
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    _full, stubs = svc.search_budgeted("relay-service ERR-0451", "evalchat")
    assert stubs
    win = svc.get_chunk_window("evalchat", stubs[0].chunk_id, 400)
    assert win and (win.get("chunk") or win.get("text"))


# ── web UI default ───────────────────────────────────────────────────────────

def test_web_ui_default_hits_is_25() -> None:
    # 20 → 25 with the 2026-07-05 budget bump (10 full + 90 stub cap).
    src = (
        Path(__file__).parent.parent
        / "vecgrep" / "frontend" / "src" / "components" / "SearchBar.tsx"
    ).read_text(encoding="utf-8")
    assert "useState(25)" in src, "web UI default hit count must be 25"
    assert "useState(5)" not in src


def test_default_budget_is_10_full_100_total():
    """Jeff's 2026-07-05 bump: 10 full + up to 90 stubs. max_total stays a CAP
    — token_ceiling is the real limit, so a dense set legitimately yields
    fewer stubs. These defaults are what the API/CLI/MCP inherit."""
    from vecgrep.backend.assembly import DEFAULT_FULL_K, DEFAULT_MAX_TOTAL
    assert DEFAULT_FULL_K == 10
    assert DEFAULT_MAX_TOTAL == 100


def test_default_split_head_is_10():
    results = [_hit(i, score=1.0 - i * 0.01) for i in range(30)]
    full, stubs = split_full_and_stubs(results)  # defaults
    assert len(full) == 10
    assert len(full) + len(stubs) <= 100
