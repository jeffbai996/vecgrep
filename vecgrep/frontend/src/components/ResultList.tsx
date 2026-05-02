import { SearchHit } from "../api";

type Props = { hits: SearchHit[] | null; searching: boolean };

function shortSource(id: string) {
  if (id.startsWith("http")) return id;
  const parts = id.split("/");
  return parts.slice(-2).join("/");
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
      {hits.map((h, i) => (
        <article
          key={i}
          className="border border-zinc-800 rounded p-4 hover:border-zinc-700 transition-colors"
        >
          <header className="flex items-baseline justify-between mb-2 gap-3">
            <div className="text-xs font-mono text-zinc-500 truncate">
              <span className="text-zinc-400">{h.corpus}</span>
              <span className="mx-2 text-zinc-700">·</span>
              {shortSource(h.source_id)}
            </div>
            <div className="text-sm font-mono text-zinc-300 shrink-0">
              {h.similarity_pct.toFixed(1)}%
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
      ))}
    </div>
  );
}

function trimTo(s: string, n: number, side: "start" | "end") {
  if (s.length <= n) return s;
  return side === "start" ? "... " + s.slice(-n) : s.slice(0, n) + " ...";
}
