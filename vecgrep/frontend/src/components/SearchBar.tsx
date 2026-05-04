import { useState } from "react";
import { SearchMode } from "../api";

type Props = {
  onSearch: (query: string, topK: number, mode: SearchMode, rerank: boolean) => void;
  disabled: boolean;
  corpus: string | null;
  corpusCount: number;
};

const MODES: SearchMode[] = ["hybrid", "vector", "bm25"];

// Subdued tints that match the legend/result palette but sit one shade
// darker so the mode toggle doesn't shout — V (sky), K/bm25 (emerald),
// hybrid (violet, the VK "both" hue). Each mode has an "active" state
// (filled) and an "idle" state (outline-only, very muted).
const MODE_STYLES: Record<SearchMode, { active: string; idle: string }> = {
  hybrid: {
    active: "bg-violet-950/60 border-violet-800/70 text-violet-300",
    idle: "border-violet-950/60 text-violet-900 hover:text-violet-400 hover:border-violet-800/60",
  },
  vector: {
    active: "bg-sky-950/60 border-sky-800/70 text-sky-300",
    idle: "border-sky-950/60 text-sky-900 hover:text-sky-400 hover:border-sky-800/60",
  },
  bm25: {
    active: "bg-emerald-950/60 border-emerald-800/70 text-emerald-300",
    idle: "border-emerald-950/60 text-emerald-900 hover:text-emerald-400 hover:border-emerald-800/60",
  },
};

export default function SearchBar({ onSearch, disabled, corpus, corpusCount }: Props) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [rerank, setRerank] = useState(false);

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || corpusCount === 0) return;
    onSearch(query.trim(), topK, mode, rerank);
  };

  const placeholder =
    corpusCount === 0
      ? "index a corpus first..."
      : corpus
      ? `search in ${corpus}...`
      : "search across all corpora...";

  return (
    <form onSubmit={onSubmit} className="space-y-2">
      <div className="flex items-center gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={placeholder}
          disabled={disabled || corpusCount === 0}
          className="flex-1 bg-zinc-900 border border-zinc-700 rounded px-3 py-2 font-mono text-sm focus:outline-none focus:border-zinc-400 disabled:opacity-50"
        />
        <input
          type="number"
          min={1}
          max={50}
          value={topK}
          onChange={(e) => setTopK(parseInt(e.target.value || "5", 10))}
          title="top-k"
          className="w-16 bg-zinc-900 border border-zinc-700 rounded px-2 py-2 font-mono text-sm text-center focus:outline-none focus:border-zinc-400"
        />
        <button
          type="submit"
          disabled={disabled || corpusCount === 0 || !query.trim()}
          className="bg-zinc-100 text-zinc-900 px-4 py-2 rounded font-mono text-sm font-semibold hover:bg-white disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {disabled ? "..." : "search"}
        </button>
      </div>
      <div className="flex items-center gap-3 text-[10px] font-mono">
        <div className="flex items-center gap-1">
          <span className="text-zinc-600 mr-1 uppercase tracking-wider">mode</span>
          {MODES.map((m) => {
            const styles = MODE_STYLES[m];
            return (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={`px-2 py-0.5 rounded border transition-colors ${
                  mode === m ? styles.active : styles.idle
                }`}
              >
                {m}
              </button>
            );
          })}
        </div>
        <label className="flex items-center gap-1 cursor-pointer text-zinc-500 hover:text-zinc-300">
          <input
            type="checkbox"
            checked={rerank}
            onChange={(e) => setRerank(e.target.checked)}
            className="accent-zinc-100"
          />
          <span className="uppercase tracking-wider">rerank</span>
        </label>
      </div>
    </form>
  );
}
