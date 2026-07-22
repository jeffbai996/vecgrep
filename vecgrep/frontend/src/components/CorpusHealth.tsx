import { useEffect, useState } from "react";
import { api, CorpusStats } from "../api";

type Props = { corpus: string };

// Sidebar health snapshot for the selected corpus. Same data as
// `vecgrep stats` — coverage span, gap days (a broken archiver shows up as a
// growing gap), and the heaviest sources.
export default function CorpusHealth({ corpus }: Props) {
  const [stats, setStats] = useState<CorpusStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setStats(null);
    setError(null);
    api
      .stats(corpus)
      .then((s) => { if (!cancelled) setStats(s); })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : String(e)); });
    return () => { cancelled = true; };
  }, [corpus]);

  if (error) {
    return (
      <div className="border border-zinc-800 rounded p-3 text-xs font-mono text-red-400">
        stats: {error}
      </div>
    );
  }
  if (!stats) {
    return (
      <div className="border border-zinc-800 rounded p-3 text-xs font-mono text-zinc-600 animate-pulse">
        loading stats...
      </div>
    );
  }

  const topSources = Object.entries(stats.sources).slice(0, 5);
  const dated = stats.date_span.first && stats.date_span.last;

  return (
    <div className="border border-zinc-800 rounded p-3 space-y-2 text-xs font-mono">
      <div className="text-zinc-500 uppercase tracking-wider">health · {stats.corpus}</div>

      <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-zinc-400">
        <span className="text-zinc-600">docs</span>
        <span className="text-right text-zinc-300">{stats.docs.toLocaleString()}</span>
        <span className="text-zinc-600">chunks</span>
        <span className="text-right text-zinc-300">{stats.chunks.toLocaleString()}</span>
        {dated && (
          <>
            <span className="text-zinc-600">span</span>
            <span className="text-right text-zinc-300">
              {stats.date_span.first} → {stats.date_span.last}
            </span>
            <span className="text-zinc-600">days covered</span>
            <span className="text-right text-zinc-300">{stats.days_covered}</span>
            <span className="text-zinc-600">gap days</span>
            <span className={`text-right ${stats.gap_days > 0 ? "text-amber-400" : "text-emerald-400"}`}>
              {stats.gap_days}
            </span>
          </>
        )}
      </div>

      {topSources.length > 0 && (
        <div className="space-y-1 pt-1 border-t border-zinc-800/70">
          <div className="text-[10px] uppercase tracking-wider text-zinc-600">top sources</div>
          {topSources.map(([sid, count]) => (
            <div key={sid} className="flex items-baseline justify-between gap-2">
              <span className="text-zinc-400 truncate" title={sid}>{shortSource(sid)}</span>
              <span className="text-zinc-600 shrink-0">{count}</span>
            </div>
          ))}
          {stats.sources_truncated && (
            <div className="text-[10px] text-zinc-600">…more sources not shown</div>
          )}
        </div>
      )}
    </div>
  );
}

function shortSource(id: string) {
  if (id.startsWith("http")) return id;
  const parts = id.split("/");
  return parts.slice(-2).join("/");
}
