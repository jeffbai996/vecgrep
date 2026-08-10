import { useState } from "react";
import { api, CompareResponse, SearchHit } from "../api";

type Props = { corpus: string | null };

export default function ComparePanel({ corpus }: Props) {
  const [query, setQuery] = useState("");
  const [aAfter, setAAfter] = useState("");
  const [aBefore, setABefore] = useState("30d");
  const [bAfter, setBAfter] = useState("30d");
  const [bBefore, setBBefore] = useState("");
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!corpus || !query.trim()) return;
    setBusy(true);
    setError(null);
    try {
      setResult(await api.compare(query.trim(), corpus, {
        a_after: aAfter || undefined,
        a_before: aBefore || undefined,
        b_after: bAfter || undefined,
        b_before: bBefore || undefined,
      }, 12));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  if (!corpus) {
    return <Empty title="Choose one corpus" body="Compare needs one corpus so both windows use the same evidence base." />;
  }

  return (
    <div className="space-y-4">
      <form onSubmit={submit} className="border border-zinc-800 rounded-xl overflow-hidden bg-zinc-950/30">
        <div className="flex items-center gap-2 p-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={`Compare how ${corpus} changed around...`}
            className="min-w-0 flex-1 bg-transparent px-2 py-2.5 focus:outline-none placeholder:text-zinc-600"
          />
          <button disabled={busy || !query.trim()} className="bg-zinc-100 text-zinc-950 px-4 py-2 rounded-lg font-mono text-xs font-semibold disabled:opacity-40">
            {busy ? "Comparing" : "Compare"}
          </button>
        </div>
        <div className="grid md:grid-cols-2 border-t border-zinc-800 divide-y md:divide-y-0 md:divide-x divide-zinc-800">
          <WindowFields label="Earlier window" after={aAfter} before={aBefore} setAfter={setAAfter} setBefore={setABefore} />
          <WindowFields label="Later window" after={bAfter} before={bBefore} setAfter={setBAfter} setBefore={setBBefore} />
        </div>
      </form>

      {error && <div className="border border-red-900 bg-red-950/30 rounded-lg p-3 text-sm text-red-400 font-mono">{error}</div>}
      {!result && !error && <Empty title="Temporal diff" body="Run the same query through two date windows to see what appeared, disappeared, or persisted." />}
      {result && (
        <>
          <div className="grid grid-cols-3 gap-2">
            <DeltaStat label="Earlier only" value={result.only_in_a.length} tone="text-amber-400" />
            <DeltaStat label="Both" value={result.in_both.length} tone="text-zinc-200" />
            <DeltaStat label="Later only" value={result.only_in_b.length} tone="text-sky-400" />
          </div>
          <div className="grid xl:grid-cols-2 gap-4 items-start">
            <CompareSide title="Earlier" hits={result.a.results} accent="border-amber-900/60" />
            <CompareSide title="Later" hits={result.b.results} accent="border-sky-900/60" />
          </div>
        </>
      )}
    </div>
  );
}

function WindowFields({ label, after, before, setAfter, setBefore }: {
  label: string;
  after: string;
  before: string;
  setAfter: (value: string) => void;
  setBefore: (value: string) => void;
}) {
  return (
    <fieldset className="p-3">
      <legend className="text-[10px] font-mono uppercase tracking-wider text-zinc-500 px-1">{label}</legend>
      <div className="grid grid-cols-2 gap-2 mt-1">
        <label className="text-[10px] font-mono text-zinc-600">
          After
          <input value={after} onChange={(e) => setAfter(e.target.value)} placeholder="YYYY-MM-DD" className="mt-1 w-full bg-zinc-900 border border-zinc-800 rounded-md px-2 py-1.5 text-xs text-zinc-300 focus:outline-none focus:border-zinc-600" />
        </label>
        <label className="text-[10px] font-mono text-zinc-600">
          Before
          <input value={before} onChange={(e) => setBefore(e.target.value)} placeholder="30d or date" className="mt-1 w-full bg-zinc-900 border border-zinc-800 rounded-md px-2 py-1.5 text-xs text-zinc-300 focus:outline-none focus:border-zinc-600" />
        </label>
      </div>
    </fieldset>
  );
}

function CompareSide({ title, hits, accent }: { title: string; hits: SearchHit[]; accent: string }) {
  return (
    <section className={`border ${accent} rounded-xl overflow-hidden`}>
      <header className="px-3 py-2 border-b border-zinc-800 flex justify-between text-xs font-mono">
        <span className="text-zinc-300">{title}</span><span className="text-zinc-600">{hits.length} results</span>
      </header>
      {hits.length ? (
        <ol className="divide-y divide-zinc-800/70">
          {hits.map((hit, index) => (
            <li key={hit.chunk_id} className="grid grid-cols-[1.5rem_minmax(0,1fr)_3rem] gap-2 px-3 py-2.5">
              <span className="text-[10px] font-mono text-zinc-700">{index + 1}</span>
              <div className="min-w-0">
                <div className="text-[10px] font-mono text-zinc-600 truncate mb-1" title={hit.source_id}>{shortSource(hit.source_id)}</div>
                <p className="result-snippet text-xs leading-relaxed font-mono text-zinc-300">{hit.chunk}</p>
              </div>
              <span className="text-xs text-right font-mono text-zinc-500">{hit.similarity_pct.toFixed(0)}%</span>
            </li>
          ))}
        </ol>
      ) : <div className="p-5 text-sm text-zinc-600 text-center">No evidence in this window.</div>}
    </section>
  );
}

function DeltaStat({ label, value, tone }: { label: string; value: number; tone: string }) {
  return <div className="border border-zinc-800 rounded-lg px-3 py-2 text-center"><div className={`text-lg font-mono ${tone}`}>{value}</div><div className="text-[10px] font-mono text-zinc-600 uppercase tracking-wide">{label}</div></div>;
}

function Empty({ title, body }: { title: string; body: string }) {
  return <div className="py-12 text-center"><div className="text-sm text-zinc-400 mb-1">{title}</div><div className="text-xs text-zinc-600 max-w-md mx-auto">{body}</div></div>;
}

function shortSource(id: string) {
  const parts = id.split("/");
  return id.startsWith("http") ? id : parts.slice(-2).join("/");
}
