import { useEffect, useRef, useState } from "react";
import { api, ChunkWindow, Corpus, SearchHit, SearchResponse, SearchStub } from "../api";
import { pctOf, rerankByTuning, Tuning } from "../tuning";
import SearchDashboard from "./SearchDashboard";

type Props = {
  response: SearchResponse | null;
  searching: boolean;
  tuning: Tuning;
  corpus: string | null;
  corpora: Corpus[];
  corporaLoading: boolean;
  corporaError: string | null;
  onRevealSource: (corpus: string, sourceId: string) => void;
};

type ExpandState = {
  data: ChunkWindow | null;
  loading: boolean;
  error: string | null;
};

type ResultRow =
  | { kind: "hit"; hit: SearchHit }
  | { kind: "stub"; stub: SearchStub };

function rowRef(row: ResultRow) {
  return row.kind === "hit" ? row.hit : row.stub;
}

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
  const tone = hasV && hasK
    ? "border-violet-800/70 text-violet-400"
    : hasK
    ? "border-emerald-800/70 text-emerald-400"
    : "border-sky-800/70 text-sky-400";
  const label = hasV && hasK ? "VK" : hasV ? "V" : "K";
  const title = hasV && hasK
    ? "Found by semantic and keyword retrieval"
    : hasV
    ? "Found by semantic retrieval"
    : "Found by exact-keyword retrieval";
  return (
    <span title={title} className={`text-[9px] font-mono border rounded px-1 py-px ${tone}`}>
      {label}
    </span>
  );
}

export default function ResultList({
  response,
  searching,
  tuning,
  corpus,
  corpora,
  corporaLoading,
  corporaError,
  onRevealSource,
}: Props) {
  const [expanded, setExpanded] = useState<Record<string, ExpandState>>({});

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key !== "Escape") return;
      setExpanded((current) => {
        const keys = Object.keys(current);
        if (!keys.length) return current;
        const next = { ...current };
        delete next[keys[keys.length - 1]];
        return next;
      });
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  async function toggleExpand(row: ResultRow) {
    const ref = rowRef(row);
    const key = `${ref.corpus}:${ref.chunk_id}`;
    if (expanded[key]) {
      setExpanded((current) => {
        const next = { ...current };
        delete next[key];
        return next;
      });
      return;
    }
    setExpanded((current) => ({
      ...current,
      [key]: { data: null, loading: true, error: null },
    }));
    try {
      const data = await api.getChunk(ref.corpus, ref.chunk_id, 3000);
      setExpanded((current) => ({
        ...current,
        [key]: { data, loading: false, error: null },
      }));
    } catch (error) {
      setExpanded((current) => ({
        ...current,
        [key]: {
          data: null,
          loading: false,
          error: error instanceof Error ? error.message : "Failed to fetch context",
        },
      }));
    }
  }

  if (searching && !response) {
    return (
      <div className="border border-zinc-800 rounded-xl overflow-hidden" aria-busy="true" aria-label="Searching">
        <div className="divide-y divide-zinc-800/70">
          {[0, 1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="px-3 py-2.5 animate-pulse flex gap-3">
              <div className="h-3 w-6 bg-zinc-800 rounded mt-1" />
              <div className="flex-1">
                <div className="h-3 w-2/5 bg-zinc-800 rounded mb-2" />
                <div className="h-3 w-5/6 bg-zinc-800/70 rounded" />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }
  if (response === null) {
    return (
      <SearchDashboard
        corpora={corpora}
        selectedCorpus={corpus}
        loading={corporaLoading}
        error={corporaError}
      />
    );
  }

  const orderedHits = rerankByTuning(response.hits, tuning);
  const rows: ResultRow[] = [
    ...orderedHits.map((hit): ResultRow => ({ kind: "hit", hit })),
    ...response.stubs.map((stub): ResultRow => ({ kind: "stub", stub })),
  ];

  if (!rows.length) {
    return <div className="text-zinc-500 text-sm py-8 text-center">No matches.</div>;
  }

  return (
    <section data-testid="result-list" className="border border-zinc-800 rounded-xl overflow-hidden bg-zinc-950/20">
      <header className="px-3 py-2 border-b border-zinc-800 flex items-center justify-between gap-3 text-[11px] font-mono">
        <span className="text-zinc-400">
          {rows.length} result{rows.length === 1 ? "" : "s"}
          <span className="text-zinc-700"> · </span>
          <span className="text-zinc-600">distinct evidence, relevance order</span>
        </span>
        <span className="text-zinc-600 shrink-0">
          {searching ? "refreshing" : `${response.hits.length} rich · ${response.stubs.length} compact`}
        </span>
      </header>

      <ol className="divide-y divide-zinc-800/70">
        {rows.map((row, index) => {
          const ref = rowRef(row);
          const key = `${ref.corpus}:${ref.chunk_id}`;
          const exp = expanded[key];
          const isOpen = Boolean(exp);
          const hit = row.kind === "hit" ? row.hit : null;
          const pct = row.kind === "hit" ? pctOf(row.hit, tuning) : row.stub.similarity_pct;
          const snippet = row.kind === "hit" ? row.hit.chunk : row.stub.snippet;
          const label = hit?.relevance_label || relevanceLabel(pct);
          return (
            <li data-testid="result-row" key={key} className={isOpen ? "bg-zinc-900/45" : "hover:bg-zinc-900/25"}>
              <div
                role="button"
                tabIndex={0}
                aria-expanded={isOpen}
                onClick={() => toggleExpand(row)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    toggleExpand(row);
                  }
                }}
                className="grid grid-cols-[2rem_minmax(0,1fr)_auto] gap-2.5 px-3 py-2.5 cursor-pointer focus:outline-none focus:bg-zinc-900/60"
              >
                <span className="font-mono text-[11px] text-zinc-700 pt-0.5 tabular-nums">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <div className="min-w-0">
                  <div className="flex items-center gap-1.5 min-w-0 text-[10px] font-mono mb-1">
                    <span className={`inline-block transition-transform text-zinc-600 ${isOpen ? "rotate-90" : ""}`}>▸</span>
                    <span className="text-zinc-400 shrink-0">{ref.corpus}</span>
                    <span className="text-zinc-700">/</span>
                    <span className="text-zinc-500 truncate" title={ref.source_id}>{shortSource(ref.source_id)}</span>
                    {hit && <MatchBadge matchedBy={hit.matched_by} />}
                    {hit?.anchor && (
                      <span className="text-zinc-700 truncate hidden lg:inline" title={hit.anchor}>
                        {lineAnchor(hit)}
                      </span>
                    )}
                    <button
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        onRevealSource(ref.corpus, ref.source_id);
                      }}
                      className="ml-1 text-zinc-700 hover:text-zinc-300"
                    >
                      Reveal
                    </button>
                    {ref.doc_timestamp && (
                      <time className="text-zinc-700 ml-auto shrink-0 hidden sm:inline">
                        {new Date(ref.doc_timestamp * 1000).toISOString().slice(0, 10)}
                      </time>
                    )}
                  </div>
                  <p className="result-snippet text-[13px] leading-[1.35rem] text-zinc-300 font-mono break-words">
                    {snippet}
                  </p>
                  {hit && <MetadataHints metadata={hit.metadata} />}
                </div>
                <div className="pl-2 text-right font-mono tabular-nums self-start">
                  <div className={scoreTone(pct)}>{pct.toFixed(0)}%</div>
                  <div className="text-[9px] uppercase tracking-wide text-zinc-600">{label}</div>
                </div>
              </div>

              {isOpen && (
                <ExpandedView
                  state={exp}
                  row={row}
                  onCollapse={() => toggleExpand(row)}
                />
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function ExpandedView({
  state,
  row,
  onCollapse,
}: {
  state: ExpandState;
  row: ResultRow;
  onCollapse: () => void;
}) {
  const markRef = useRef<HTMLElement | null>(null);
  const ref = rowRef(row);

  useEffect(() => {
    if (state.data && markRef.current) {
      markRef.current.scrollIntoView({ block: "nearest", behavior: "auto" });
    }
  }, [state.data]);

  if (state.error) {
    return <div className="border-t border-zinc-800 px-12 py-3 text-xs text-red-400">{state.error}</div>;
  }
  if (state.loading || !state.data) {
    return (
      <div className="border-t border-zinc-800 px-12 py-3 text-[11px] font-mono text-zinc-500">
        Loading source context...
      </div>
    );
  }

  const data = state.data;
  const shown = data.before.length + data.chunk.length + data.after.length;
  return (
    <div className="border-t border-zinc-800 px-3 sm:px-12 py-3" onClick={(e) => e.stopPropagation()}>
      <div className="flex items-center justify-between gap-3 text-[10px] font-mono text-zinc-600 mb-2">
        <span>{shown.toLocaleString()} / {data.source_length.toLocaleString()} characters</span>
        <button type="button" onClick={onCollapse} className="hover:text-zinc-300">Collapse · Esc</button>
      </div>
      <div className="max-h-[560px] overflow-y-auto border-l-2 border-zinc-800 pl-3 pr-2 text-sm font-mono leading-relaxed whitespace-pre-wrap break-words">
        {data.before && <span className="text-zinc-500">{data.before}</span>}
        <mark ref={markRef} className="bg-yellow-500/20 text-yellow-100 not-italic">{data.chunk}</mark>
        {data.after && <span className="text-zinc-500">{data.after}</span>}
      </div>
      {row.kind === "hit" && <ScoreBreakdown hit={row.hit} />}
      <RelatedChunks corpus={ref.corpus} chunkId={ref.chunk_id} />
    </div>
  );
}

function RelatedChunks({ corpus, chunkId }: { corpus: string; chunkId: string }) {
  const [state, setState] = useState<
    | { kind: "idle" }
    | { kind: "loading" }
    | { kind: "error"; message: string }
    | { kind: "loaded"; hits: SearchHit[] }
  >({ kind: "idle" });

  async function load() {
    setState({ kind: "loading" });
    try {
      const response = await api.related(corpus, chunkId);
      setState({ kind: "loaded", hits: response.hits.filter((hit) => hit.chunk_id !== chunkId) });
    } catch (error) {
      setState({ kind: "error", message: error instanceof Error ? error.message : "Failed to fetch" });
    }
  }

  if (state.kind === "idle") {
    return <button type="button" onClick={load} className="mt-3 text-[10px] font-mono uppercase tracking-wider text-zinc-600 hover:text-zinc-300">Find related evidence ▸</button>;
  }
  if (state.kind === "loading") return <div className="mt-3 text-[10px] font-mono text-zinc-600">Finding related evidence...</div>;
  if (state.kind === "error") return <div className="mt-3 text-[10px] font-mono text-red-400">{state.message}</div>;
  if (!state.hits.length) return <div className="mt-3 text-[10px] font-mono text-zinc-600">No related chunks.</div>;
  return (
    <div className="mt-3 border-t border-zinc-800 pt-2">
      <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-600 mb-1">Related evidence</div>
      <div className="divide-y divide-zinc-800/60">
        {state.hits.slice(0, 6).map((hit) => (
          <div key={hit.chunk_id} className="py-1.5 grid grid-cols-[2.5rem_minmax(0,1fr)] gap-2 text-xs font-mono">
            <span className="text-zinc-600 tabular-nums">{hit.similarity_pct.toFixed(0)}%</span>
            <span className="text-zinc-400 truncate" title={hit.source_id}>{shortSource(hit.source_id)} · {hit.chunk}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function MetadataHints({ metadata }: { metadata: Record<string, unknown> }) {
  const values: string[] = [];
  const speakers = Array.isArray(metadata.speakers) ? metadata.speakers.map(String) : [];
  if (speakers.length) values.push(speakers.slice(0, 3).join(", "));
  if (metadata.has_code === true) values.push("code");
  if (metadata.has_table === true) values.push("table");
  if (metadata.has_link === true) values.push("link");
  if (!values.length) return null;
  return <div className="mt-1 text-[9px] font-mono text-zinc-700 truncate">{values.join(" · ")}</div>;
}

function ScoreBreakdown({ hit }: { hit: SearchHit }) {
  const explain = hit.explain || {};
  const parts: string[] = [];
  if (typeof explain.vector_cosine === "number") parts.push(`vector ${explain.vector_cosine.toFixed(3)}${explain.vector_rank ? ` #${explain.vector_rank}` : ""}`);
  if (typeof explain.bm25_score === "number") parts.push(`keyword ${explain.bm25_score.toFixed(2)}${explain.bm25_rank ? ` #${explain.bm25_rank}` : ""}`);
  if (typeof explain.rrf === "number") parts.push(`fusion ${explain.rrf.toFixed(4)}`);
  if (typeof explain.rerank_score === "number") parts.push(`rerank ${explain.rerank_score.toFixed(3)}`);
  if (!parts.length) return null;
  return <div className="mt-2 text-[10px] font-mono text-zinc-700">{parts.join(" · ")}</div>;
}

function lineAnchor(hit: SearchHit) {
  if (!hit.line_start) return "";
  return hit.line_end && hit.line_end !== hit.line_start
    ? `L${hit.line_start}-${hit.line_end}`
    : `L${hit.line_start}`;
}

function relevanceLabel(pct: number) {
  if (pct >= 85) return "exact";
  if (pct >= 70) return "strong";
  if (pct >= 45) return "related";
  return "weak";
}

function scoreTone(pct: number) {
  if (pct >= 75) return "text-sm text-emerald-400 font-semibold";
  if (pct >= 45) return "text-sm text-amber-400 font-semibold";
  return "text-sm text-zinc-500 font-semibold";
}
