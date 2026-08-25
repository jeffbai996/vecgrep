import { Corpus } from "../api";
import { corpusTone } from "../browseTones";

type Props = {
  corpora: Corpus[];
  selectedCorpus: string | null;
  loading: boolean;
  error: string | null;
};

type ModelSummary = {
  backend: string;
  model: string;
  corpusCount: number;
  docCount: number;
};

export default function SearchDashboard({
  corpora,
  selectedCorpus,
  loading,
  error,
}: Props) {
  const corpusNames = corpora.map((corpus) => corpus.name);
  const scopedCorpora = selectedCorpus
    ? corpora.filter((corpus) => corpus.name === selectedCorpus)
    : corpora;
  const totalDocs = sum(scopedCorpora, "doc_count");
  const totalChunks = sum(scopedCorpora, "chunk_count");
  const latestUpdate = Math.max(0, ...scopedCorpora.map((corpus) => corpus.updated_at));
  const recent = [...scopedCorpora]
    .sort((left, right) => right.updated_at - left.updated_at)
    .slice(0, 5);
  const models = modelInventory(scopedCorpora);
  const maxChunks = Math.max(1, ...scopedCorpora.map((corpus) => corpus.chunk_count));
  const largest = [...scopedCorpora].sort(
    (left, right) => right.chunk_count - left.chunk_count
  )[0];
  const emptyCount = scopedCorpora.filter((corpus) => corpus.doc_count === 0).length;
  const scopeLabel = selectedCorpus || "All corpora";

  if (loading && corpora.length === 0) {
    return <DashboardSkeleton />;
  }

  return (
    <section
      aria-label="Index overview"
      className="border border-zinc-800/80 bg-zinc-950/20 rounded-xl overflow-hidden"
    >
      <header className="px-4 py-3 border-b border-zinc-800/80 flex items-center justify-between gap-3">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-[0.16em] text-zinc-600">
            Index overview
          </div>
          <h2 className="mt-1 text-sm font-medium text-zinc-300">{scopeLabel}</h2>
        </div>
        <div
          className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-1 text-[10px] font-mono ${
            error
              ? "border-rose-900/70 bg-rose-950/30 text-rose-300"
              : "border-emerald-900/70 bg-emerald-950/30 text-emerald-300"
          }`}
        >
          <span
            className={`h-1.5 w-1.5 rounded-full ${error ? "bg-rose-400" : "bg-emerald-400"}`}
          />
          {error ? "Registry unavailable" : "API reachable"}
        </div>
      </header>

      {error && (
        <div className="border-b border-rose-900/50 bg-rose-950/20 px-4 py-2 text-[11px] font-mono text-rose-300">
          {error}
        </div>
      )}

      <div className="grid grid-cols-2 lg:grid-cols-4 border-b border-zinc-800/80">
        <Metric label="Corpora" value={scopedCorpora.length.toLocaleString()} />
        <Metric label="Documents" value={totalDocs.toLocaleString()} />
        <Metric label="Chunks" value={totalChunks.toLocaleString()} />
        <Metric
          label="Last indexed"
          value={latestUpdate ? relativeAge(latestUpdate) : "No activity"}
          title={latestUpdate ? absoluteTime(latestUpdate) : undefined}
        />
      </div>

      {scopedCorpora.length === 0 ? (
        <div className="px-4 py-10 text-center font-mono text-xs text-zinc-600">
          No indexed corpora in this scope.
        </div>
      ) : (
        <div className="grid lg:grid-cols-[1.25fr_1fr]">
          <div className="border-b lg:border-b-0 lg:border-r border-zinc-800/80">
            <DashboardSection title="Latest index activity">
              <div className="space-y-1">
                {recent.map((corpus) => {
                  const tone = corpusTone(corpus.name, corpusNames);
                  return (
                    <div
                      key={corpus.name}
                      className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 rounded-lg px-2 py-1.5 hover:bg-zinc-900/45"
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <span className={`h-2 w-2 rounded-full shrink-0 ring-1 ring-white/20 ${tone.dot}`} />
                        <span className={`truncate text-xs font-mono ${tone.text}`}>
                          {corpus.name}
                        </span>
                        <span className="truncate text-[10px] font-mono text-zinc-700">
                          {corpus.doc_count.toLocaleString()} docs
                        </span>
                      </div>
                      <time
                        dateTime={new Date(corpus.updated_at * 1000).toISOString()}
                        title={absoluteTime(corpus.updated_at)}
                        className="text-[10px] font-mono text-zinc-500 tabular-nums"
                      >
                        {relativeAge(corpus.updated_at)}
                      </time>
                    </div>
                  );
                })}
              </div>
            </DashboardSection>

            <DashboardSection title="Corpus distribution" borderTop>
              <div className="space-y-2.5">
                {scopedCorpora.map((corpus) => {
                  const tone = corpusTone(corpus.name, corpusNames);
                  const width = Math.max(2, (corpus.chunk_count / maxChunks) * 100);
                  return (
                    <div key={corpus.name} className="grid grid-cols-[6rem_minmax(0,1fr)_auto] items-center gap-2">
                      <span className="truncate text-[10px] font-mono text-zinc-500" title={corpus.name}>
                        {corpus.name}
                      </span>
                      <div className="h-1.5 overflow-hidden rounded-full bg-zinc-900">
                        <div className={`h-full rounded-full ${tone.dot}`} style={{ width: `${width}%` }} />
                      </div>
                      <span className="w-14 text-right text-[10px] font-mono tabular-nums text-zinc-600">
                        {compactNumber(corpus.chunk_count)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </DashboardSection>
          </div>

          <div>
            <DashboardSection title="Embedding inventory">
              <div className="space-y-2">
                {models.map((entry) => (
                  <div key={`${entry.backend}:${entry.model}`} className="rounded-lg border border-sky-900/40 bg-sky-950/15 px-3 py-2">
                    <div className="flex items-center justify-between gap-3">
                      <span className="truncate text-xs font-mono text-sky-300" title={entry.model}>
                        {entry.model}
                      </span>
                      <span className="shrink-0 rounded border border-zinc-800 px-1.5 py-0.5 text-[9px] font-mono uppercase text-zinc-500">
                        {entry.backend}
                      </span>
                    </div>
                    <div className="mt-1 text-[10px] font-mono text-zinc-600">
                      {entry.corpusCount} corp{entry.corpusCount === 1 ? "us" : "ora"} · {entry.docCount.toLocaleString()} docs
                    </div>
                  </div>
                ))}
              </div>
            </DashboardSection>

            <DashboardSection title="Things to know" borderTop>
              <ul className="space-y-2 text-[11px] font-mono leading-relaxed text-zinc-500">
                <li className="flex gap-2">
                  <span className="text-violet-400">●</span>
                  <span>
                    {models.length === 1
                      ? `All ${scopedCorpora.length} corpora use one active embedding model.`
                      : `${models.length} embedding models are active across this scope.`}
                  </span>
                </li>
                {largest && (
                  <li className="flex gap-2">
                    <span className="text-sky-400">●</span>
                    <span>
                      <span className="text-zinc-300">{largest.name}</span> is the largest corpus at {largest.chunk_count.toLocaleString()} chunks.
                    </span>
                  </li>
                )}
                <li className="flex gap-2">
                  <span className={emptyCount ? "text-amber-400" : "text-emerald-400"}>●</span>
                  <span>
                    {emptyCount
                      ? `${emptyCount} corp${emptyCount === 1 ? "us has" : "ora have"} no indexed documents.`
                      : "Every listed corpus has indexed documents."}
                  </span>
                </li>
              </ul>
            </DashboardSection>
          </div>
        </div>
      )}
    </section>
  );
}

function Metric({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div className="border-r border-b last:border-r-0 lg:border-b-0 border-zinc-800/70 px-4 py-3">
      <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-700">{label}</div>
      <div className="mt-1 text-base font-mono tabular-nums text-zinc-300" title={title}>
        {value}
      </div>
    </div>
  );
}

function DashboardSection({
  title,
  borderTop = false,
  children,
}: {
  title: string;
  borderTop?: boolean;
  children: React.ReactNode;
}) {
  return (
    <section className={`${borderTop ? "border-t border-zinc-800/80" : ""} p-4`}>
      <h3 className="mb-3 text-[10px] font-mono uppercase tracking-[0.14em] text-zinc-600">
        {title}
      </h3>
      {children}
    </section>
  );
}

function DashboardSkeleton() {
  return (
    <section
      aria-label="Loading index overview"
      aria-busy="true"
      className="min-h-[360px] animate-pulse rounded-xl border border-zinc-800/80 bg-zinc-950/20 p-4"
    >
      <div className="h-3 w-28 rounded bg-zinc-800" />
      <div className="mt-5 grid grid-cols-2 gap-2 lg:grid-cols-4">
        {[0, 1, 2, 3].map((item) => (
          <div key={item} className="h-16 rounded-lg bg-zinc-900" />
        ))}
      </div>
      <div className="mt-4 grid gap-3 lg:grid-cols-2">
        <div className="h-48 rounded-lg bg-zinc-900" />
        <div className="h-48 rounded-lg bg-zinc-900" />
      </div>
    </section>
  );
}

function modelInventory(corpora: Corpus[]): ModelSummary[] {
  const models = new Map<string, ModelSummary>();
  for (const corpus of corpora) {
    const key = `${corpus.embed_backend}:${corpus.embed_model}`;
    const current = models.get(key) || {
      backend: corpus.embed_backend,
      model: corpus.embed_model,
      corpusCount: 0,
      docCount: 0,
    };
    current.corpusCount += 1;
    current.docCount += corpus.doc_count;
    models.set(key, current);
  }
  return [...models.values()].sort((left, right) => right.corpusCount - left.corpusCount);
}

function sum(corpora: Corpus[], field: "doc_count" | "chunk_count") {
  return corpora.reduce((total, corpus) => total + corpus[field], 0);
}

function relativeAge(timestamp: number) {
  const seconds = Math.max(0, Math.floor(Date.now() / 1000 - timestamp));
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 60) return `${days}d ago`;
  return `${Math.floor(days / 30)}mo ago`;
}

function absoluteTime(timestamp: number) {
  return new Date(timestamp * 1000).toLocaleString();
}

function compactNumber(value: number) {
  return new Intl.NumberFormat(undefined, {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}
