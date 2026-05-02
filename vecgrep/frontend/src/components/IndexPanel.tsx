import { useState } from "react";
import { api } from "../api";

type Props = { onIndexed: () => void };

export default function IndexPanel({ onIndexed }: Props) {
  const [source, setSource] = useState("");
  const [corpus, setCorpus] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!source.trim() || !corpus.trim()) return;
    setBusy(true);
    setMsg(null);
    try {
      const r = await api.index(source.trim(), corpus.trim());
      setMsg(`indexed ${r.docs} doc(s), ${r.chunks} chunk(s).`);
      setSource("");
      onIndexed();
    } catch (e) {
      setMsg(`error: ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="border border-zinc-800 rounded p-3 space-y-2">
      <div className="text-xs text-zinc-500 font-mono uppercase tracking-wider">
        index
      </div>
      <form onSubmit={onSubmit} className="space-y-2">
        <input
          type="text"
          value={source}
          onChange={(e) => setSource(e.target.value)}
          placeholder="path or URL"
          className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 font-mono text-xs focus:outline-none focus:border-zinc-400"
        />
        <input
          type="text"
          value={corpus}
          onChange={(e) => setCorpus(e.target.value)}
          placeholder="corpus name"
          className="w-full bg-zinc-900 border border-zinc-700 rounded px-2 py-1.5 font-mono text-xs focus:outline-none focus:border-zinc-400"
        />
        <button
          type="submit"
          disabled={busy || !source.trim() || !corpus.trim()}
          className="w-full bg-zinc-100 text-zinc-900 px-3 py-1.5 rounded font-mono text-xs font-semibold hover:bg-white disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {busy ? "indexing..." : "index"}
        </button>
      </form>
      {msg && (
        <div className="text-xs font-mono text-zinc-400 whitespace-pre-wrap break-words">
          {msg}
        </div>
      )}
    </div>
  );
}
