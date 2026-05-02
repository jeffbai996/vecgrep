import { useState } from "react";

type Props = {
  onSearch: (query: string, topK: number) => void;
  disabled: boolean;
  corpus: string | null;
  corpusCount: number;
};

export default function SearchBar({ onSearch, disabled, corpus, corpusCount }: Props) {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || corpusCount === 0) return;
    onSearch(query.trim(), topK);
  };

  const placeholder =
    corpusCount === 0
      ? "index a corpus first..."
      : corpus
      ? `search in ${corpus}...`
      : "search across all corpora...";

  return (
    <form onSubmit={onSubmit} className="flex items-center gap-2">
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
    </form>
  );
}
