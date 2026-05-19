import { SearchHit } from "../api";
import { pctOf, rerankByTuning, Tuning } from "../tuning";

type Props = { hits: SearchHit[] | null; searching: boolean; tuning: Tuning };

// Confidence tiers, calibrated against the new sigmoid scoring.
// Under the calibrated map: noise floor sits at ~10-25%, weak relevant
// 30-60%, clearly relevant 60-80%, strong 80%+. We tier on (pct, matched_by)
// jointly so a "both retrievers agreed" hit at 70% still reads high.
type Tier = "high" | "med" | "none";

function confidenceTier(pct: number, matchedBy: string[] | undefined): Tier {
  const set = new Set(matchedBy || []);
  const hasBoth = set.has("vector") && set.has("bm25");
  if (pct >= 75 || hasBoth) return "high";
  if (pct >= 45) return "med";
  return "none";
}

const TIER_PCT_CLASS: Record<Tier, string> = {
  high: "text-emerald-400",
  med: "text-amber-400",
  none: "text-zinc-500",
};

const TIER_BORDER_CLASS: Record<Tier, string> = {
  high: "border-emerald-900/60 hover:border-emerald-700",
  med: "border-amber-900/50 hover:border-amber-700",
  none: "border-zinc-800 hover:border-zinc-700",
};

const TIER_LABEL: Record<Tier, string> = {
  high: "high",
  med: "soft",
  none: "weak",
};

const TIER_LABEL_CLASS: Record<Tier, string> = {
  high: "text-emerald-500/80",
  med: "text-amber-500/80",
  none: "text-zinc-600",
};

function shortSource(id: string) {
  if (id.startsWith("http")) return id;
  const parts = id.split("/");
  return parts.slice(-2).join("/");
}

function MatchBadge({ matchedBy }: { matchedBy: string[] | undefined }) {
  const set = new Set(matchedBy || []);
  const hasV = set.has("vector");
  const hasK = set.has("bm25");
  if (!hasV && !hasK) return null;
  // Color encoding: K (keyword/literal) = emerald, V (vector/semantic) = sky,
  // VK (both) = violet. The hue carries meaning; the tooltip explains it.
  const tone = hasV && hasK
    ? "bg-violet-900/40 border-violet-700/60 text-violet-300"
    : hasK
    ? "bg-emerald-900/40 border-emerald-700/60 text-emerald-300"
    : "bg-sky-900/40 border-sky-700/60 text-sky-300";
  const label = hasV && hasK ? "VK" : hasV ? "V" : "K";
  const title = hasV && hasK
    ? "vector + bm25 — both retrievers fired"
    : hasV
    ? "vector — semantic similarity match"
    : "bm25 — exact-keyword match (literal token in source)";
  return (
    <span
      title={title}
      className={`text-[10px] font-mono font-bold border rounded px-1.5 py-px ${tone}`}
    >
      {label}
    </span>
  );
}

export default function ResultList({ hits, searching, tuning }: Props) {
  if (searching && !hits) {
    return (
      <div className="text-zinc-500 font-mono text-sm">searching...</div>
    );
  }
  if (hits === null) {
    return (
      <div className="text-zinc-600 font-mono text-sm">
        results show up here.
      </div>
    );
  }
  if (hits.length === 0) {
    return <div className="text-zinc-500 font-mono text-sm">no matches.</div>;
  }
  // Re-rank under user tuning so dragging the bm25-bias slider actually
  // reorders the list, not just relabels it.
  const ordered = rerankByTuning(hits, tuning);
  return (
    <div className="space-y-3">
      {ordered.map((h, i) => {
        const displayPct = pctOf(h, tuning);
        const tier = confidenceTier(displayPct, h.matched_by);
        return (
          <article
            key={i}
            className={`border rounded p-4 transition-colors ${TIER_BORDER_CLASS[tier]}`}
          >
            <header className="flex items-baseline justify-between mb-2 gap-3">
              <div className="text-xs font-mono text-zinc-500 truncate flex items-center gap-2 min-w-0">
                <span className="text-zinc-400">{h.corpus}</span>
                <span className="text-zinc-700">·</span>
                <span className="truncate">{shortSource(h.source_id)}</span>
                <MatchBadge matchedBy={h.matched_by} />
              </div>
              <div className="flex items-baseline gap-2 shrink-0">
                <span
                  className={`text-[10px] font-mono uppercase tracking-wider ${TIER_LABEL_CLASS[tier]}`}
                >
                  {TIER_LABEL[tier]}
                </span>
                <span
                  className={`text-sm font-mono font-semibold ${TIER_PCT_CLASS[tier]}`}
                  title={`raw scores: ${formatRaw(h)}`}
                >
                  {displayPct.toFixed(1)}%
                </span>
              </div>
            </header>
            <div className="font-mono text-sm leading-relaxed whitespace-pre-wrap">
              {h.context_before && (
                <span className="text-zinc-600">
                  {trimTo(h.context_before, 200, "start")}
                </span>
              )}
              <mark className="bg-yellow-500/20 text-yellow-100 not-italic">
                {h.chunk}
              </mark>
              {h.context_after && (
                <span className="text-zinc-600">
                  {" "}
                  {trimTo(h.context_after, 200, "end")}
                </span>
              )}
            </div>
          </article>
        );
      })}
    </div>
  );
}

function trimTo(s: string, n: number, side: "start" | "end") {
  if (s.length <= n) return s;
  return side === "start" ? "... " + s.slice(-n) : s.slice(0, n) + " ...";
}

function formatRaw(h: SearchHit): string {
  const e = h.explain;
  if (!e) return "n/a";
  const parts: string[] = [];
  if (typeof e.vector_cosine === "number")
    parts.push(`cos=${e.vector_cosine.toFixed(3)}`);
  if (typeof e.bm25_score === "number")
    parts.push(`bm25=${e.bm25_score.toFixed(2)}`);
  if (typeof e.rrf === "number") parts.push(`rrf=${e.rrf.toFixed(4)}`);
  return parts.join(" · ") || "n/a";
}
