// Score calibration controls.
//
// The server computes `similarity_pct` with default calibration, but always
// emits the raw underlying scores (`explain.vector_cosine`, `bm25_score`,
// `bm25_max`). This module reads those raw scores and re-derives the display
// percentage under user-tunable parameters, so the user can drag a slider
// and see scores update without a network round trip.
//
// Defaults mirror the server-side constants. If the user hasn't opened the
// tuning panel, display percentages will match the server-rendered ones.

import { SearchHit } from "./api";

export type Tuning = {
  cosineCenter: number; // sigmoid inflection point (0.0 - 1.0)
  cosineSlope: number;  // sigmoid steepness (larger = sharper)
  bm25Top: number;      // display pct ceiling for top BM25 hit (per query)
  bm25Floor: number;    // display pct floor for weakest BM25 hit (per query)
  bm25Bias: number;     // when both retrievers fire, +N% added to whichever
                        // signal includes BM25. 0 = no bias, 100 = BM25 dominates.
};

export const DEFAULT_TUNING: Tuning = {
  cosineCenter: 0.66,
  cosineSlope: 12,
  bm25Top: 100,
  bm25Floor: 25,
  bm25Bias: 0,
};

const STORAGE_KEY = "vecgrep_tuning_v1";

export function loadTuning(): Tuning {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_TUNING };
    const parsed = JSON.parse(raw) as Partial<Tuning>;
    return { ...DEFAULT_TUNING, ...parsed };
  } catch {
    return { ...DEFAULT_TUNING };
  }
}

export function saveTuning(t: Tuning) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(t));
  } catch {
    // localStorage full / disabled — silently fall back.
  }
}

function cosineToPct(cos: number, t: Tuning): number {
  const x = t.cosineSlope * (cos - t.cosineCenter);
  // Guard against overflow for extreme x.
  if (x > 60) return 100;
  if (x < -60) return 0;
  return 100 / (1 + Math.exp(-x));
}

function bm25ToPct(score: number, max: number, t: Tuning): number {
  if (max <= 0) return t.bm25Floor;
  const ratio = Math.max(0, Math.min(1, score / max));
  return t.bm25Floor + (t.bm25Top - t.bm25Floor) * ratio;
}

/**
 * Re-derive display percentage from raw retriever scores under the given
 * tuning. Falls back to server-rendered similarity_pct when raw scores are
 * missing (older API responses, rerank-only mode, etc.).
 */
export function pctOf(hit: SearchHit, t: Tuning): number {
  const e = hit.explain;
  if (!e) return hit.similarity_pct;

  // If rerank fired, use rerank_score * 100 — already in [0,1].
  if (typeof e.rerank_score === "number") {
    return Math.max(0, Math.min(100, e.rerank_score * 100));
  }

  const cosPct =
    typeof e.vector_cosine === "number"
      ? cosineToPct(e.vector_cosine, t)
      : null;
  const bmPct =
    typeof e.bm25_score === "number" && typeof e.bm25_max === "number"
      ? bm25ToPct(e.bm25_score, e.bm25_max, t)
      : null;

  if (cosPct !== null && bmPct !== null) {
    // Both retrievers fired. Apply user's BM25 bias preference: nudges
    // toward BM25 when bias > 0, toward vector when bias < 0.
    const biased = bmPct + t.bm25Bias / 2; // half-effect since we take max
    return Math.max(cosPct, Math.min(100, biased));
  }
  if (cosPct !== null) return cosPct;
  if (bmPct !== null) return bmPct;
  return hit.similarity_pct;
}

/**
 * Re-rank a hit list by recomputed pct. Stable for ties (preserves
 * server-side order when pcts are equal).
 */
export function rerankByTuning(hits: SearchHit[], t: Tuning): SearchHit[] {
  return [...hits]
    .map((h, i) => ({ h, i, p: pctOf(h, t) }))
    .sort((a, b) => b.p - a.p || a.i - b.i)
    .map((x) => x.h);
}
