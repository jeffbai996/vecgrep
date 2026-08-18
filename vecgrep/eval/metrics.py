"""Scoring. Source-level: a corpus answer is a document (a transcript day, a
memory file, a source file), so results are deduped to their first source
occurrence before ranking — a document shattered into 400 chunks must not get
400 shots at the top-k."""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .gold import GoldCase

KS = (1, 3, 5, 10)


@dataclass
class CaseResult:
    case: GoldCase
    sources: list[str]            # deduped, best first
    top_pct: float                # similarity_pct of the best hit (0 if none)
    pcts: list[float]             # per deduped source, best first
    latency_s: float
    first_hit_rank: int | None = None
    forbid_hits: int = 0

    def hit_at(self, k: int) -> bool:
        return self.first_hit_rank is not None and self.first_hit_rank <= k

    def precision_at(self, k: int) -> float:
        top = self.sources[:k]
        if not top:
            return 0.0
        return sum(1 for s in top if self.case.matches_want(s)) / len(top)


def score_case(case: GoldCase, hits: list[dict], latency_s: float) -> CaseResult:
    """`hits` are search results as dicts with source_id + similarity_pct."""
    seen: list[str] = []
    pcts: list[float] = []
    for h in hits:
        sid = h.get("source_id") or ""
        if sid and sid not in seen:
            seen.append(sid)
            pcts.append(float(h.get("similarity_pct") or 0.0))
    r = CaseResult(case=case, sources=seen, top_pct=(pcts[0] if pcts else 0.0),
                   pcts=pcts, latency_s=latency_s)
    for i, s in enumerate(seen, start=1):
        if case.matches_want(s):
            r.first_hit_rank = i
            break
    r.forbid_hits = sum(1 for s in seen if case.matches_forbid(s))
    return r


def _pct(x: float) -> float:
    return round(100.0 * x, 1)


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[idx]


def summarize(results: list[CaseResult], neg_floor_pct: float = 60.0) -> dict:
    """One row. Graded cases drive hit@k / MRR / P@k; negatives drive the
    false-positive columns (a should-match-nothing query whose top hit still
    clears the recall floor is junk that would be injected into a turn)."""
    graded = [r for r in results if not r.case.negative]
    negs = [r for r in results if r.case.negative]
    lat = [r.latency_s for r in results]
    row: dict = {"n_graded": len(graded), "n_negative": len(negs)}
    for k in KS:
        row[f"hit@{k}"] = _pct(sum(r.hit_at(k) for r in graded) / len(graded)) if graded else None
    row["mrr"] = round(
        sum((1.0 / r.first_hit_rank) for r in graded if r.first_hit_rank) / len(graded), 3
    ) if graded else None
    row["p@3"] = _pct(statistics.mean(r.precision_at(3) for r in graded)) if graded else None
    row["p@5"] = _pct(statistics.mean(r.precision_at(5) for r in graded)) if graded else None
    row["misses"] = [r.case.id for r in graded if r.first_hit_rank is None]
    row["forbid_leaks"] = sum(1 for r in graded if r.forbid_hits)
    if negs:
        row["neg_fp_rate"] = _pct(sum(1 for r in negs if r.top_pct >= neg_floor_pct) / len(negs))
        row["neg_top_pct_mean"] = round(statistics.mean(r.top_pct for r in negs), 1)
    else:
        row["neg_fp_rate"] = None
        row["neg_top_pct_mean"] = None
    # separability: how far the best hit stands above the third-best, averaged
    # over graded cases that returned >=3 sources — the "can the reader tell a
    # bullseye from a tangent" number the 2026-06-23 investigation tracked
    spreads = [r.pcts[0] - r.pcts[2] for r in graded if len(r.pcts) >= 3]
    row["spread_1v3"] = round(statistics.mean(spreads), 1) if spreads else None
    row["lat_p50_ms"] = round(1000 * _quantile(lat, 0.5), 1)
    row["lat_p95_ms"] = round(1000 * _quantile(lat, 0.95), 1)
    return row


def per_case_rows(results: list[CaseResult]) -> list[dict]:
    return [{
        "id": r.case.id,
        "corpus": r.case.corpus,
        "negative": r.case.negative,
        "rank": r.first_hit_rank,
        "top_pct": round(r.top_pct, 1),
        "forbid_hits": r.forbid_hits,
        "latency_ms": round(1000 * r.latency_s, 1),
        "top3": r.sources[:3],
    } for r in results]
