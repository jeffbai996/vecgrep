import { useState } from "react";
import { api, TimelineGroup } from "../api";

type Props = { corpus: string | null; corpusCount: number };

// "What happened?" mode. One query -> contiguous chronological slices grouped
// by source, transcript slices parsed into (speaker, time, text) events.
// Mirrors `vecgrep timeline` on the CLI.
export default function TimelinePanel({ corpus, corpusCount }: Props) {
  const [query, setQuery] = useState("");
  const [filtersRaw, setFiltersRaw] = useState("");
  const [maxGroups, setMaxGroups] = useState(4);
  const [groups, setGroups] = useState<TimelineGroup[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || corpusCount === 0) return;
    setBusy(true);
    setError(null);
    try {
      const filters = filtersRaw.trim() ? filtersRaw.trim().split(/\s+/) : [];
      setGroups(await api.timeline(query.trim(), corpus, maxGroups, filters));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setGroups([]);
    } finally {
      setBusy(false);
    }
  };

  const placeholder =
    corpusCount === 0
      ? "index a corpus first..."
      : corpus
      ? `what happened... (in ${corpus})`
      : "what happened... (all corpora)";

  return (
    <div className="space-y-4">
      <form onSubmit={onSubmit} className="space-y-2">
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={placeholder}
            disabled={busy || corpusCount === 0}
            className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-3 py-2 font-mono text-sm focus:outline-none focus:border-zinc-400 disabled:opacity-50"
          />
          <input
            type="number"
            min={1}
            max={12}
            value={maxGroups}
            onChange={(e) => setMaxGroups(parseInt(e.target.value || "4", 10))}
            title="max groups"
            className="w-16 bg-zinc-900 border border-zinc-700 rounded px-2 py-2 font-mono text-sm text-center focus:outline-none focus:border-zinc-400"
          />
          <button
            type="submit"
            disabled={busy || corpusCount === 0 || !query.trim()}
            className="bg-zinc-100 text-zinc-900 px-4 py-2 rounded font-mono text-sm font-semibold hover:bg-white disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {busy ? "..." : "reconstruct"}
          </button>
        </div>
        <input
          type="text"
          value={filtersRaw}
          onChange={(e) => setFiltersRaw(e.target.value)}
          placeholder="filters (optional): after:2026-07-01 before:2026-07-15 speaker:NAME source:GLOB has:code"
          className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-1.5 font-mono text-xs text-zinc-400 placeholder:text-zinc-700 focus:outline-none focus:border-zinc-600"
        />
      </form>

      {error && (
        <div className="text-sm text-red-400 font-mono whitespace-pre-wrap border border-red-800 bg-red-950/40 rounded p-3">
          {error}
        </div>
      )}

      {groups === null && !error && (
        <div className="text-zinc-600 font-mono text-sm">
          chronological reconstruction shows up here — best on transcript-style
          corpora (chat logs, journals) where chunks carry timestamps.
        </div>
      )}
      {groups !== null && groups.length === 0 && !error && (
        <div className="text-zinc-500 font-mono text-sm">nothing found for that window.</div>
      )}

      {groups && groups.length > 0 && (
        <ol className="relative border-l border-zinc-800 ml-2 space-y-6">
          {groups.map((g, i) => (
            <li key={`${g.source_id}-${i}`} className="ml-4">
              <span className="absolute -left-[5px] mt-1.5 h-2.5 w-2.5 rounded-full border border-zinc-600 bg-zinc-900" />
              <header className="flex items-baseline gap-2 text-xs font-mono mb-2 flex-wrap">
                {g.doc_timestamp && (
                  <span className="text-zinc-200 font-semibold">
                    {new Date(g.doc_timestamp * 1000).toISOString().slice(0, 10)}
                  </span>
                )}
                <span className="text-zinc-500 truncate">{shortSource(g.source_id)}</span>
                <span className="text-zinc-700">·</span>
                <span className="text-zinc-600">{g.corpus}</span>
              </header>
              {g.events.length > 0 ? (
                <div className="border border-zinc-800 rounded divide-y divide-zinc-800/60">
                  {g.events.map((ev, j) => (
                    <div key={j} className="px-3 py-1.5 text-sm font-mono flex gap-3">
                      <span className="text-zinc-600 text-xs shrink-0 w-12 pt-0.5">{ev.time}</span>
                      <div className="min-w-0">
                        <span className="text-sky-300 text-xs">{ev.speaker}</span>
                        <span className="text-zinc-300 whitespace-pre-wrap break-words block">
                          {ev.text}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="border border-zinc-800 rounded p-3 text-sm font-mono text-zinc-300 whitespace-pre-wrap break-words">
                  {g.slice_text}
                </div>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

function shortSource(id: string) {
  if (id.startsWith("http")) return id;
  const parts = id.split("/");
  return parts.slice(-2).join("/");
}
