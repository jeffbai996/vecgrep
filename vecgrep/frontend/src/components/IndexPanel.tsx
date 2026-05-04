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
      <div className="flex items-center justify-between">
        <div className="text-xs text-zinc-500 font-mono uppercase tracking-wider">
          index
        </div>
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

      <details className="text-xs font-mono text-zinc-500 group">
        <summary className="cursor-pointer select-none hover:text-zinc-300 list-none flex items-center gap-1">
          <span className="inline-block transition-transform group-open:rotate-90">▸</span>
          <span>how do I add stuff?</span>
        </summary>
        <div className="mt-2 pl-4 space-y-2 text-zinc-400 leading-relaxed">
          <p>
            <span className="text-zinc-300">source</span> can be:
          </p>
          <ul className="space-y-1 pl-3 list-disc list-outside marker:text-zinc-700">
            <li>
              an absolute file path (<code className="text-zinc-300">/home/you/notes.md</code>)
              &mdash; .md, .txt, .pdf are auto-detected
            </li>
            <li>
              a directory (<code className="text-zinc-300">/home/you/repo</code>) &mdash;
              walks recursively, dispatches per file extension
            </li>
            <li>
              a URL (<code className="text-zinc-300">https://...</code>) &mdash;
              fetches and indexes the page text
            </li>
          </ul>
          <p>
            <span className="text-zinc-300">corpus name</span> is just a label.
            New name → new corpus. Existing name → adds/updates docs in place.
          </p>
          <p>
            re-indexing the same source is a no-op (content-hashed). edit the
            file and re-run to refresh just that doc. corpora pin their embed
            model &mdash; mixing models inside one corpus is rejected.
          </p>
          <p className="text-zinc-500">
            CLI does the same:{" "}
            <code className="text-zinc-400">vecgrep index &lt;src&gt; --corpus &lt;name&gt;</code>
          </p>
        </div>
      </details>
    </div>
  );
}
