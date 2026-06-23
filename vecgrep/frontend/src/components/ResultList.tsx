import { useEffect, useRef, useState } from "react";
import { api, ChunkWindow, SearchHit } from "../api";
import { pctOf, rerankByTuning, Tuning } from "../tuning";

type Props = { hits: SearchHit[] | null; searching: boolean; tuning: Tuning };

// Per-result expansion state. Keyed by chunk_id so it survives re-renders
// when the parent list changes order under tuning.
type ExpandState = {
  data: ChunkWindow | null;       // null while loading or before first fetch
  loading: boolean;
  error: string | null;
};

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

// Brighter borders for open results — visually anchors the user's eye to
// what's currently expanded, especially helpful with multiple results open.
const TIER_BORDER_CLASS_OPEN: Record<Tier, string> = {
  high: "border-emerald-600/80",
  med: "border-amber-600/80",
  none: "border-zinc-600",
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
  // Expansion state lives at the list level so the same chunk stays open
  // across tuning-driven reorders (which only swap the array, not identity).
  const [expanded, setExpanded] = useState<Record<string, ExpandState>>({});

  // ESC collapses the most recently opened result. Tracking "most recent" by
  // insertion order in the expanded map.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key !== "Escape") return;
      setExpanded((s) => {
        const keys = Object.keys(s);
        if (!keys.length) return s;
        const last = keys[keys.length - 1];
        const next = { ...s };
        delete next[last];
        return next;
      });
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  async function toggleExpand(h: SearchHit) {
    const cur = expanded[h.chunk_id];
    if (cur) {
      setExpanded((e) => {
        const next = { ...e };
        delete next[h.chunk_id];
        return next;
      });
      return;
    }
    setExpanded((e) => ({
      ...e,
      [h.chunk_id]: { data: null, loading: true, error: null },
    }));
    try {
      const data = await api.getChunk(h.corpus, h.chunk_id, 2000);
      setExpanded((e) => ({
        ...e,
        [h.chunk_id]: { data, loading: false, error: null },
      }));
    } catch (err) {
      setExpanded((e) => ({
        ...e,
        [h.chunk_id]: {
          data: null,
          loading: false,
          error: err instanceof Error ? err.message : "failed to fetch",
        },
      }));
    }
  }

  if (searching && !hits) {
    return (
      <div className="space-y-3" aria-busy="true" aria-label="searching">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="border border-zinc-800 rounded p-3 animate-pulse"
          >
            <div className="flex items-center gap-2 mb-2">
              <div className="h-3 w-10 bg-zinc-800 rounded" />
              <div className="h-3 w-32 bg-zinc-800/70 rounded" />
            </div>
            <div className="h-3 w-full bg-zinc-800/60 rounded mb-1.5" />
            <div className="h-3 w-11/12 bg-zinc-800/60 rounded mb-1.5" />
            <div className="h-3 w-4/6 bg-zinc-800/50 rounded" />
          </div>
        ))}
      </div>
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
        const exp = h.chunk_id ? expanded[h.chunk_id] : undefined;
        const isOpen = !!exp;
        return (
          <article
            key={h.chunk_id || i}
            className={`border rounded p-4 transition-colors ${
              isOpen ? TIER_BORDER_CLASS_OPEN[tier] : TIER_BORDER_CLASS[tier]
            }`}
          >
            <header className="flex items-baseline justify-between mb-2 gap-3">
              <div className="text-xs font-mono text-zinc-500 truncate flex items-center gap-2 min-w-0">
                {/* Chevron — visible affordance that the row is clickable.
                    Rotates 90° on expand so state is unambiguous. */}
                {h.chunk_id && (
                  <span
                    className={`text-zinc-500 select-none transition-transform inline-block w-3 ${
                      isOpen ? "rotate-90 text-zinc-300" : ""
                    }`}
                    aria-hidden="true"
                  >
                    ▸
                  </span>
                )}
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
            {/* Body — clickable to expand. Keep the existing inline preview
                until the user opts in; expanded view replaces it. div+role
                instead of <button> because the expanded view nests its own
                buttons (nested buttons are invalid HTML). */}
            <div
              role={h.chunk_id ? "button" : undefined}
              tabIndex={h.chunk_id ? 0 : -1}
              onClick={() => h.chunk_id && toggleExpand(h)}
              onKeyDown={(e) => {
                if (h.chunk_id && (e.key === "Enter" || e.key === " ")) {
                  e.preventDefault();
                  toggleExpand(h);
                }
              }}
              className={`font-mono text-sm leading-relaxed whitespace-pre-wrap break-words -mx-1 px-1 py-0.5 rounded transition-colors ${
                h.chunk_id ? "cursor-pointer hover:bg-zinc-900/30" : ""
              } ${isOpen ? "bg-zinc-900/20" : ""}`}
              title={h.chunk_id ? (isOpen ? "click to collapse (Esc)" : "click to expand context") : ""}
            >
              {isOpen ? (
                <ExpandedView
                  exp={exp!}
                  fallback={
                    <>
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
                    </>
                  }
                />
              ) : (
                <>
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
                </>
              )}
            </div>
          </article>
        );
      })}
    </div>
  );
}

function ExpandedView({
  exp,
  fallback,
}: {
  exp: ExpandState;
  // Fallback content (the inline preview from the search hit) is shown while
  // the wider window is in flight, so the user never sees the chunk disappear.
  fallback: React.ReactNode;
}) {
  const markRef = useRef<HTMLElement | null>(null);

  // Once data lands, scroll the highlighted chunk into view inside the
  // scrollable container. Without this, a chunk at chunk_start=5000 ends up
  // off-screen after a ±2000 expansion, forcing the user to hunt for it.
  useEffect(() => {
    if (!exp.data || !markRef.current) return;
    markRef.current.scrollIntoView({ block: "center", behavior: "auto" });
  }, [exp.data]);

  if (exp.error) {
    return (
      <div className="text-red-400 text-xs py-1">error: {exp.error}</div>
    );
  }
  if (exp.loading && !exp.data) {
    return (
      <>
        <div className="opacity-60">{fallback}</div>
        <div className="mt-2 text-[10px] font-mono text-zinc-500 flex items-center gap-2">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-zinc-500 animate-pulse" />
          loading wider context…
        </div>
      </>
    );
  }
  const d = exp.data!;
  const coveredChars = d.before.length + d.chunk.length + d.after.length;
  return (
    <>
      <div className="flex items-center justify-between mb-2 text-[10px] font-mono text-zinc-500">
        <span>
          showing {coveredChars.toLocaleString()} / {d.source_length.toLocaleString()} chars
        </span>
        <span className="text-zinc-600">click row or Esc to collapse</span>
      </div>
      <div className="max-h-[600px] overflow-y-auto pr-2 border-l-2 border-zinc-800 pl-3 whitespace-pre-wrap break-words">
        {d.before && <span className="text-zinc-500">{d.before}</span>}
        <mark
          ref={markRef}
          className="bg-yellow-500/25 text-yellow-100 not-italic"
        >
          {d.chunk}
        </mark>
        {d.after && <span className="text-zinc-500">{d.after}</span>}
      </div>
    </>
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
