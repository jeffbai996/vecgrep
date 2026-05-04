import { SearchHit } from "../api";

type Props = { hits: SearchHit[] | null; searching: boolean };

// Confidence tiers — must match bot logic in cc-context discord_handler.py.
// Pct alone is unreliable: vector embeddings floor around 70-75% on noise,
// and BM25 hits get crushed by RRF into single-digit pcts. So we tier on
// (pct, matched_by) jointly: a BM25 hit is "high" even at 1.6%.
type Tier = "high" | "med" | "none";

function confidenceTier(pct: number, matchedBy: string[] | undefined): Tier {
  const set = new Set(matchedBy || []);
  const hasBm25 = set.has("bm25");
  const hasVector = set.has("vector");
  if (pct >= 85 || hasBm25) return "high";
  if (pct >= 78 || (hasVector && pct >= 75)) return "med";
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

export default function ResultList({ hits, searching }: Props) {
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
  return (
    <div className="space-y-3">
      {hits.map((h, i) => {
        const tier = confidenceTier(h.similarity_pct, h.matched_by);
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
                >
                  {h.similarity_pct.toFixed(1)}%
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
