import { useState } from "react";
import { SearchMode, SearchOptions } from "../api";

type Props = {
  onSearch: (query: string, options: SearchOptions) => void;
  disabled: boolean;
  corpus: string | null;
  corpusCount: number;
};

const MODES: {
  value: SearchMode;
  label: string;
  hint: string;
  active: string;
  idle: string;
}[] = [
  {
    value: "hybrid",
    label: "Hybrid",
    hint: "meaning + exact terms",
    active: "border-violet-700/80 bg-violet-950/70 text-violet-300",
    idle: "border-transparent text-violet-500/70 hover:bg-violet-950/35 hover:text-violet-300",
  },
  {
    value: "vector",
    label: "Semantic",
    hint: "meaning only",
    active: "border-sky-700/80 bg-sky-950/70 text-sky-300",
    idle: "border-transparent text-sky-500/70 hover:bg-sky-950/35 hover:text-sky-300",
  },
  {
    value: "bm25",
    label: "Keyword",
    hint: "exact terms only",
    active: "border-emerald-700/80 bg-emerald-950/70 text-emerald-300",
    idle: "border-transparent text-emerald-500/70 hover:bg-emerald-950/35 hover:text-emerald-300",
  },
];

const QUICK_FILTERS = [
  { token: "date:today", label: "Today" },
  { token: "after:7d", label: "7 days" },
  { token: "after:30d", label: "30 days" },
  { token: "bot:false", label: "Humans" },
  { token: "has:code", label: "Code" },
  { token: "has:link", label: "Links" },
  { token: "has:table", label: "Tables" },
];

export default function SearchBar({ onSearch, disabled, corpus, corpusCount }: Props) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(40);
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [rerank, setRerank] = useState(false);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [quickFilters, setQuickFilters] = useState<string[]>([]);
  const [filtersRaw, setFiltersRaw] = useState("");

  const filters = [
    ...quickFilters,
    ...(filtersRaw.trim() ? filtersRaw.trim().split(/\s+/) : []),
  ];

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || corpusCount === 0) return;
    onSearch(query.trim(), { topK, mode, rerank, filters });
  };

  const toggleFilter = (token: string) => {
    setQuickFilters((current) =>
      current.includes(token)
        ? current.filter((value) => value !== token)
        : [...current, token]
    );
  };

  const placeholder =
    corpusCount === 0
      ? "Index a corpus first"
      : corpus
      ? `Search ${corpus}`
      : "Search everything";

  return (
    <form onSubmit={onSubmit} className="border border-zinc-800 bg-zinc-950/40 rounded-xl overflow-hidden">
      <div className="flex items-center gap-2 p-2">
        <span className="text-zinc-600 pl-2" aria-hidden="true">⌕</span>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={placeholder}
          disabled={disabled || corpusCount === 0}
          autoFocus
          className="min-w-0 flex-1 bg-transparent px-1 py-2.5 text-base focus:outline-none disabled:opacity-50 placeholder:text-zinc-600"
        />
        <button
          type="submit"
          disabled={disabled || corpusCount === 0 || !query.trim()}
          className="bg-zinc-100 text-zinc-950 px-4 py-2 rounded-lg font-mono text-xs font-semibold hover:bg-white disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {disabled ? "Searching" : "Search"}
        </button>
      </div>

      <div className="border-t border-zinc-800/80 px-3 py-2 flex items-center gap-2 flex-wrap text-[11px] font-mono">
        <div className="flex items-center rounded-lg bg-zinc-900/70 p-0.5">
          {MODES.map((item) => (
            <button
              key={item.value}
              type="button"
              onClick={() => setMode(item.value)}
              title={item.hint}
              aria-pressed={mode === item.value}
              className={`border px-2.5 py-1 rounded-md leading-none transition-colors ${
                mode === item.value ? item.active : item.idle
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>

        <label className="flex items-center gap-1.5 text-zinc-500 hover:text-zinc-300 cursor-pointer px-1">
          <input
            type="checkbox"
            checked={rerank}
            onChange={(e) => setRerank(e.target.checked)}
            className="accent-violet-400 h-3 w-3 shrink-0"
          />
          Deep rerank
        </label>

        <label className="flex items-center gap-1.5 text-zinc-600 ml-auto">
          <span>Show</span>
          <select
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="bg-zinc-900 border border-zinc-800 rounded-md px-2 py-1 text-zinc-300 focus:outline-none focus:border-zinc-600"
            aria-label="Result count"
          >
            <option value={25}>25</option>
            <option value={40}>40</option>
            <option value={60}>60</option>
            <option value={100}>100</option>
          </select>
        </label>

        <button
          type="button"
          onClick={() => setFiltersOpen((value) => !value)}
          className={`px-2 py-1 rounded-md transition-colors ${
            filtersOpen || filters.length
              ? "bg-zinc-800 text-zinc-200"
              : "text-zinc-500 hover:text-zinc-300"
          }`}
        >
          Filters{filters.length ? ` · ${filters.length}` : ""} {filtersOpen ? "▴" : "▾"}
        </button>
      </div>

      {filtersOpen && (
        <div className="border-t border-zinc-800/80 p-3 space-y-3 bg-zinc-950/70">
          <div>
            <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-600 mb-2">
              Quick filters
            </div>
            <div className="flex flex-wrap gap-1.5">
              {QUICK_FILTERS.map((item) => {
                const active = quickFilters.includes(item.token);
                return (
                  <button
                    key={item.token}
                    type="button"
                    onClick={() => toggleFilter(item.token)}
                    className={`px-2.5 py-1 rounded-full border text-xs transition-colors ${
                      active
                        ? "border-violet-700 bg-violet-950/60 text-violet-200"
                        : "border-zinc-800 text-zinc-500 hover:border-zinc-600 hover:text-zinc-300"
                    }`}
                  >
                    {item.label}
                  </button>
                );
              })}
            </div>
          </div>
          <label className="block">
            <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-600">
              Precise filters
            </span>
            <input
              type="text"
              value={filtersRaw}
              onChange={(e) => setFiltersRaw(e.target.value)}
              placeholder="speaker:alice source:*journal* -corpus:scratch meta.kind=decision"
              className="mt-1.5 w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 font-mono text-xs text-zinc-300 placeholder:text-zinc-700 focus:outline-none focus:border-zinc-600"
            />
          </label>
          <p className="text-[10px] font-mono text-zinc-600">
            Also accepts date:, before:, after:, channel:, source_path:, speaker:, bot:, has:, meta.KEY=VALUE; prefix any filter with - to exclude.
          </p>
        </div>
      )}
    </form>
  );
}
