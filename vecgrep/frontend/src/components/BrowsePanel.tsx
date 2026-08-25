import { useEffect, useMemo, useState } from "react";
import {
  api,
  ExplorerDocument,
  ExplorerListing,
  ExplorerSort,
  ExplorerSource,
} from "../api";
import {
  BrowseTone,
  corpusTone,
  folderTone,
  kindTone,
  neutralTone,
  schemeTone,
  tagTone,
} from "../browseTones";

type RevealTarget = { corpus: string; sourceId: string; nonce: number };
type Props = {
  corpus: string | null;
  revealSource: RevealTarget | null;
  onRevealHandled: () => void;
};

const PAGE_SIZE = 50;

export default function BrowsePanel({ corpus, revealSource, onRevealHandled }: Props) {
  const [path, setPath] = useState<string[]>([]);
  const [queryInput, setQueryInput] = useState("");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<ExplorerSort>("name");
  const [offset, setOffset] = useState(0);
  const [listing, setListing] = useState<ExplorerListing | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [preview, setPreview] = useState<ExplorerSource | null>(null);
  const [busy, setBusy] = useState(false);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    setPath([]);
    setQueryInput("");
    setQuery("");
    setSort("name");
    setOffset(0);
    setListing(null);
    setSelectedId(null);
    setPreview(null);
    setError(null);
  }, [corpus]);

  useEffect(() => {
    if (!corpus) return;
    let current = true;
    setBusy(true);
    setError(null);
    api.explore(corpus, { path, query, sort, offset, limit: PAGE_SIZE })
      .then((response) => {
        if (current) setListing(response);
      })
      .catch((reason) => {
        if (current) {
          setListing(null);
          setError(reason instanceof Error ? reason.message : String(reason));
        }
      })
      .finally(() => {
        if (current) setBusy(false);
      });
    return () => {
      current = false;
    };
  }, [corpus, path, query, sort, offset, reloadKey]);

  useEffect(() => {
    if (!corpus || !revealSource || revealSource.corpus !== corpus) return;
    void openSource(revealSource.sourceId, true).finally(onRevealHandled);
    // nonce deliberately retriggers a reveal of the same source.
  }, [corpus, revealSource?.nonce]);

  async function openSource(sourceId: string, reveal = false) {
    if (!corpus) return;
    setSelectedId(sourceId);
    setPreviewBusy(true);
    setPreviewError(null);
    try {
      const source = await api.explorerSource(corpus, sourceId);
      setPreview(source);
      if (reveal) {
        setPath(source.parent_path);
        setOffset(0);
        setQueryInput("");
        setQuery("");
      }
    } catch (reason) {
      setPreview(null);
      setPreviewError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setPreviewBusy(false);
    }
  }

  function openFolder(nextPath: string[]) {
    setPath(nextPath);
    setOffset(0);
    setSelectedId(null);
    setPreview(null);
    setPreviewError(null);
  }

  function submitFilter(event: React.FormEvent) {
    event.preventDefault();
    setOffset(0);
    setQuery(queryInput.trim());
  }

  if (!corpus) {
    return (
      <div className="py-16 text-center text-sm text-zinc-500">
        Choose one corpus to open its document explorer.
      </div>
    );
  }

  const usingRecent = Boolean(
    listing && !query && path.length === 0 && listing.documents.length === 0
  );
  const documents = listing
    ? usingRecent
      ? listing.recent_documents
      : listing.documents
    : [];
  const activeCorpusTone = corpusTone(corpus);
  const catalogCount = listing
    ? Object.values(listing.facets.kinds).reduce((total, count) => total + count, 0)
    : 0;

  return (
    <div className="space-y-3">
      <header className={`relative overflow-hidden border rounded-xl bg-zinc-950/40 px-3 py-3 flex flex-col xl:flex-row xl:items-center gap-3 ${activeCorpusTone.border}`}>
        <div className={`absolute inset-y-0 left-0 w-1 ${activeCorpusTone.dot}`} />
        <div className="min-w-0 xl:mr-auto pl-1 flex flex-wrap items-center gap-2">
          <ColorPill label={corpus} tone={activeCorpusTone} />
          {listing ? (
            <>
              <ColorPill label={schemeLabel(listing.scheme)} tone={schemeTone(listing.scheme)} dot={false} />
              <span className="text-[10px] font-mono text-zinc-600 tabular-nums">
                {catalogCount.toLocaleString()} source{catalogCount === 1 ? "" : "s"}
              </span>
            </>
          ) : (
            <span
              aria-live="polite"
              className="inline-flex items-center gap-1.5 rounded-full border border-sky-800/70 bg-sky-950/35 px-2 py-1 text-[10px] font-mono text-sky-300"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-sky-400 animate-pulse" />
              Cataloging
            </span>
          )}
          {busy && listing && (
            <span className="text-[9px] font-mono text-sky-400 animate-pulse">updating</span>
          )}
        </div>
        <form onSubmit={submitFilter} className="flex min-w-0 flex-1 xl:max-w-xl gap-2">
          <div className="relative min-w-0 flex-1">
            <span className={`absolute left-3 top-1/2 -translate-y-1/2 h-1.5 w-1.5 rounded-full ${query ? activeCorpusTone.dot : "bg-zinc-700"}`} />
            <input
              value={queryInput}
              onChange={(event) => setQueryInput(event.target.value)}
              placeholder="Filter names, paths, or tags"
              className="w-full bg-zinc-900/80 border border-zinc-800 rounded-lg pl-7 pr-3 py-2 text-xs text-zinc-200 placeholder:text-zinc-700 focus:outline-none focus:border-sky-700"
            />
          </div>
          <button
            type="submit"
            className={`border rounded-lg px-3 text-[10px] font-mono transition-colors ${query ? `${activeCorpusTone.border} ${activeCorpusTone.bg} ${activeCorpusTone.text}` : "border-zinc-700 text-zinc-400 hover:text-zinc-100 hover:border-zinc-500"}`}
          >
            Filter
          </button>
        </form>
        <select
          aria-label="Document sort"
          value={sort}
          onChange={(event) => {
            setSort(event.target.value as ExplorerSort);
            setOffset(0);
          }}
          className="bg-zinc-900/80 border border-zinc-800 rounded-lg px-2 py-2 text-[10px] font-mono text-zinc-400 focus:outline-none focus:border-sky-700"
        >
          <option value="name">Name</option>
          <option value="newest">Newest</option>
          <option value="oldest">Oldest</option>
        </select>
      </header>

      {error && (
        <div className="border border-rose-900/80 bg-rose-950/30 rounded-lg px-3 py-2.5 flex items-center gap-3 text-xs text-rose-300 font-mono">
          <span className="h-2 w-2 rounded-full bg-rose-400 shrink-0" />
          <span className="min-w-0 flex-1 break-words">{error}</span>
          <button
            type="button"
            onClick={() => setReloadKey((value) => value + 1)}
            className="shrink-0 rounded-full border border-rose-800 bg-rose-950/50 px-2.5 py-1 text-[10px] text-rose-200 hover:border-rose-600"
          >
            Retry
          </button>
        </div>
      )}

      <div
        aria-busy={busy}
        className={`grid lg:grid-cols-[minmax(190px,0.7fr)_minmax(260px,1fr)_minmax(340px,1.45fr)] border border-zinc-800 rounded-xl overflow-hidden bg-zinc-950/20 min-h-[640px] transition-opacity ${busy && listing ? "opacity-75" : "opacity-100"}`}
      >
        <CollectionsPane
          path={path}
          listing={listing}
          busy={busy}
          onOpen={openFolder}
        />
        <DocumentsPane
          documents={documents}
          listing={listing}
          query={query}
          usingRecent={usingRecent}
          selectedId={selectedId}
          busy={busy}
          onOpen={openSource}
          onPage={setOffset}
        />
        <PreviewPane
          preview={preview}
          listing={listing}
          catalogBusy={busy && !listing}
          busy={previewBusy}
          error={previewError}
        />
      </div>
    </div>
  );
}

function CollectionsPane({
  path,
  listing,
  busy,
  onOpen,
}: {
  path: string[];
  listing: ExplorerListing | null;
  busy: boolean;
  onOpen: (path: string[]) => void;
}) {
  return (
    <section className="border-b lg:border-b-0 lg:border-r border-zinc-800 min-w-0">
      <PaneHeading title="Collections" detail={listing ? `${listing.folders.length} here` : ""} />
      <div className="px-2 py-2 border-b border-zinc-800/70 flex flex-wrap items-center gap-1.5 text-[10px] font-mono bg-zinc-950/30">
        <button
          onClick={() => onOpen([])}
          className={`rounded-full border px-2 py-0.5 transition-colors ${path.length === 0 ? "border-violet-800/70 bg-violet-950/35 text-violet-300" : "border-zinc-800 text-zinc-600 hover:text-zinc-200"}`}
        >
          Root
        </button>
        {path.map((segment, index) => (
          <span key={`${segment}-${index}`} className="min-w-0 flex items-center gap-1">
            <span className="text-zinc-700">/</span>
            <button
              onClick={() => onOpen(path.slice(0, index + 1))}
              className={`rounded-full border px-2 py-0.5 truncate max-w-[150px] transition-colors ${index === path.length - 1 ? `${folderTone(path.slice(0, index + 1)).border} ${folderTone(path.slice(0, index + 1)).bg} ${folderTone(path.slice(0, index + 1)).text}` : "border-zinc-800 text-zinc-500 hover:text-zinc-200"}`}
              title={segment}
            >
              {segment}
            </button>
          </span>
        ))}
      </div>
      <div className="max-h-[330px] lg:max-h-[570px] overflow-y-auto p-1.5">
        {path.length > 0 && (
          <button
            onClick={() => onOpen(path.slice(0, -1))}
            className="w-full rounded-md px-2 py-2 text-left text-xs font-mono text-zinc-600 hover:bg-zinc-900 hover:text-zinc-300 flex items-center gap-2"
          >
            <span className="h-5 w-5 rounded-md border border-zinc-800 bg-zinc-900 flex items-center justify-center text-[10px]">..</span>
            Parent
          </button>
        )}
        {listing?.folders.map((folder) => {
          const tone = folderTone(folder.path);
          return (
            <button
              key={folder.path.join("\u0000")}
              onClick={() => onOpen(folder.path)}
              className="w-full grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 rounded-lg px-2 py-2 text-left hover:bg-zinc-900/80 group"
            >
              <span className={`h-6 w-6 rounded-md border flex items-center justify-center ${tone.border} ${tone.bg}`}>
                <span className={`h-2 w-2 rounded-sm ${tone.dot}`} />
              </span>
              <span className={`truncate text-xs ${tone.text} group-hover:text-zinc-100`} title={folder.name}>
                {folder.name}
              </span>
              <span className={`rounded-full border px-1.5 py-px text-[9px] font-mono tabular-nums ${tone.border} ${tone.bg} ${tone.text}`}>
                {folder.document_count.toLocaleString()}
              </span>
            </button>
          );
        })}
        {!busy && listing && !listing.folders.length && (
          <div className="px-2 py-8 text-center text-xs text-zinc-700">
            <span className="inline-flex rounded-full border border-zinc-800 bg-zinc-900/50 px-2.5 py-1">No subfolders</span>
          </div>
        )}
        {busy && !listing && <LoadingRows count={5} />}
      </div>
    </section>
  );
}

function DocumentsPane({
  documents,
  listing,
  query,
  usingRecent,
  selectedId,
  busy,
  onOpen,
  onPage,
}: {
  documents: ExplorerDocument[];
  listing: ExplorerListing | null;
  query: string;
  usingRecent: boolean;
  selectedId: string | null;
  busy: boolean;
  onOpen: (sourceId: string) => void;
  onPage: (offset: number) => void;
}) {
  const title = query ? "Matches" : usingRecent ? "Recent documents" : "Documents";
  const detail = usingRecent
    ? `${documents.length} latest`
    : listing
    ? `${listing.total_documents.toLocaleString()} here`
    : "";
  return (
    <section className="border-b lg:border-b-0 lg:border-r border-zinc-800 min-w-0">
      <PaneHeading title={title} detail={detail} />
      <div className="max-h-[420px] lg:max-h-[525px] overflow-y-auto divide-y divide-zinc-800/60">
        {documents.map((document) => {
          const tone = kindTone(document.kind);
          const selected = selectedId === document.source_id;
          return (
            <button
              key={document.source_id}
              onClick={() => onOpen(document.source_id)}
              className={`w-full border-l-2 px-3 py-2.5 text-left transition-colors ${selected ? tone.selected : "border-transparent hover:border-zinc-700 hover:bg-zinc-900/70"}`}
            >
              <div className="flex items-start gap-2">
                <span className="min-w-0 flex-1 truncate text-xs text-zinc-200" title={document.name}>
                  {document.name}
                </span>
                {document.doc_timestamp && (
                  <time className="shrink-0 rounded-full border border-sky-900/60 bg-sky-950/25 px-1.5 py-px text-[9px] font-mono text-sky-500">
                    {formatDate(document.doc_timestamp)}
                  </time>
                )}
              </div>
              <div className="mt-1.5 flex flex-wrap items-center gap-1.5 min-w-0 text-[9px] font-mono text-zinc-700">
                <ColorPill label={document.kind} tone={tone} dot={false} compact />
                <span className="tabular-nums">{document.chunk_count} chunks</span>
                {document.parent_path.length > 0 && (
                  <>
                    <span>·</span>
                    <span className="truncate" title={document.parent_path.join(" / ")}>
                      {document.parent_path.join(" / ")}
                    </span>
                  </>
                )}
              </div>
              {document.tags.length > 0 && (
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {document.tags.slice(0, 4).map((tag) => (
                    <ColorPill key={tag} label={tag} tone={tagTone(tag)} dot={false} compact />
                  ))}
                </div>
              )}
            </button>
          );
        })}
        {!busy && listing && !documents.length && (
          <div className="px-4 py-16 text-center text-xs text-zinc-600">
            <div className="mx-auto mb-2 h-2 w-2 rounded-full bg-zinc-700" />
            {query ? "No names, paths, or tags matched." : "Open a collection to see its documents."}
          </div>
        )}
        {busy && !listing && <LoadingRows count={7} />}
      </div>
      {!usingRecent && listing && listing.total_documents > listing.limit && (
        <div className="h-[45px] border-t border-zinc-800 flex items-center justify-between gap-2 px-3 text-[10px] font-mono text-zinc-600">
          <button
            disabled={listing.offset === 0}
            onClick={() => onPage(Math.max(0, listing.offset - listing.limit))}
            className="disabled:opacity-25 hover:text-zinc-200"
          >
            Previous
          </button>
          <span>
            {listing.offset + 1}-{Math.min(listing.offset + listing.documents.length, listing.total_documents)}
          </span>
          <button
            disabled={!listing.has_more}
            onClick={() => onPage(listing.offset + listing.limit)}
            className="disabled:opacity-25 hover:text-zinc-200"
          >
            Next
          </button>
        </div>
      )}
    </section>
  );
}

function PreviewPane({
  preview,
  listing,
  catalogBusy,
  busy,
  error,
}: {
  preview: ExplorerSource | null;
  listing: ExplorerListing | null;
  catalogBusy: boolean;
  busy: boolean;
  error: string | null;
}) {
  const topTags = useMemo(
    () => Object.entries(listing?.facets.tags || {}).sort((a, b) => b[1] - a[1]).slice(0, 8),
    [listing]
  );
  return (
    <section className="min-w-0">
      <PaneHeading title="Preview" detail={preview ? formatChars(preview.source_length) : ""} />
      {catalogBusy && (
        <div className="h-[590px] p-6 flex flex-col items-center justify-center text-center">
          <div className="relative mb-4 h-10 w-10 rounded-xl border border-sky-800/70 bg-sky-950/35 flex items-center justify-center">
            <span className="absolute inset-0 rounded-xl border border-sky-500/30 animate-ping" />
            <span className="h-2.5 w-2.5 rounded-sm bg-sky-400" />
          </div>
          <div className="text-sm text-sky-300">Cataloging sources</div>
          <div className="mt-1 max-w-xs text-[10px] font-mono text-zinc-600">
            Preparing folders and document metadata.
          </div>
        </div>
      )}
      {busy && <div className="p-4"><LoadingRows count={8} /></div>}
      {error && <div className="p-4 text-xs font-mono text-rose-400">{error}</div>}
      {!busy && preview && (
        <div className="flex flex-col h-[590px]">
          <div className={`relative overflow-hidden px-4 py-3 border-b ${kindTone(preview.kind).border} ${kindTone(preview.kind).bg}`}>
            <span className={`absolute inset-y-0 left-0 w-1 ${kindTone(preview.kind).dot}`} />
            <h3 className="text-sm text-zinc-100 break-words">{preview.name}</h3>
            <div className="mt-1 text-[9px] font-mono text-zinc-600 break-all">
              {preview.display_path.join(" / ")}
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[9px] font-mono text-zinc-600">
              <ColorPill label={preview.kind} tone={kindTone(preview.kind)} dot={false} compact />
              <ColorPill label={`${preview.chunk_count} chunks`} tone={neutralTone} dot={false} compact />
              {preview.doc_timestamp && (
                <ColorPill label={formatDate(preview.doc_timestamp)} tone={schemeTone("channels")} dot={false} compact />
              )}
              {preview.tags.map((tag) => (
                <ColorPill key={tag} label={tag} tone={tagTone(tag)} dot={false} compact />
              ))}
            </div>
          </div>
          <pre className="flex-1 overflow-auto whitespace-pre-wrap break-words px-4 py-3 text-[12px] leading-[1.25rem] font-mono text-zinc-400">
            {preview.text}
          </pre>
          {preview.truncated && (
            <div className="border-t border-zinc-800 px-4 py-2 text-[9px] font-mono text-amber-500/80">
              Preview capped at {formatChars(preview.text.length)} of {formatChars(preview.source_length)}.
            </div>
          )}
        </div>
      )}
      {!catalogBusy && !busy && !error && !preview && (
        <div className="p-5 text-xs text-zinc-600 space-y-5">
          <div className="flex items-start gap-3 rounded-lg border border-violet-900/50 bg-violet-950/20 p-3">
            <span className="mt-0.5 h-2 w-2 shrink-0 rounded-full bg-violet-400" />
            <p className="text-zinc-500">Select a document to read it here. Folder navigation stays metadata-only.</p>
          </div>
          {listing && (
            <>
              <div>
                <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-600 mb-2">Catalog mix</div>
                <div className="grid grid-cols-2 gap-2 font-mono text-[10px]">
                  {Object.entries(listing.facets.kinds).map(([kind, count]) => {
                    const tone = kindTone(kind);
                    return (
                      <div key={kind} className={`border rounded-lg px-2 py-2 flex items-center justify-between gap-2 ${tone.border} ${tone.bg}`}>
                        <span className={`truncate ${tone.text}`}>{kind}</span>
                        <span className={`tabular-nums ${tone.text}`}>{count.toLocaleString()}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
              {topTags.length > 0 && (
                <div>
                  <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-600 mb-2">Top tags</div>
                  <div className="flex flex-wrap gap-1.5">
                    {topTags.map(([tag, count]) => (
                      <ColorPill key={tag} label={tag} count={count} tone={tagTone(tag)} dot={false} compact />
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </section>
  );
}

function PaneHeading({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="h-[43px] px-3 border-b border-zinc-800 bg-zinc-950/35 flex items-center justify-between gap-2">
      <h2 className="flex items-center gap-2 text-[10px] font-mono uppercase tracking-wider text-zinc-400">
        <span className="h-1.5 w-1.5 rounded-full bg-violet-500" />
        {title}
      </h2>
      {detail && <span className="text-[9px] font-mono text-zinc-700 truncate">{detail}</span>}
    </div>
  );
}

function ColorPill({
  label,
  tone,
  count,
  dot = true,
  compact = false,
}: {
  label: string;
  tone: BrowseTone;
  count?: number;
  dot?: boolean;
  compact?: boolean;
}) {
  return (
    <span
      className={`inline-flex max-w-full items-center gap-1.5 rounded-full border font-mono ${tone.border} ${tone.bg} ${tone.text} ${compact ? "px-1.5 py-px text-[9px]" : "px-2 py-1 text-[10px]"}`}
      title={label}
    >
      {dot && <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${tone.dot}`} />}
      <span className="truncate">{label}</span>
      {count !== undefined && (
        <span className="border-l border-current/20 pl-1.5 tabular-nums opacity-70">{count.toLocaleString()}</span>
      )}
    </span>
  );
}

function LoadingRows({ count }: { count: number }) {
  return (
    <div className="animate-pulse divide-y divide-zinc-900">
      {Array.from({ length: count }, (_, index) => (
        <div key={index} className="h-11 px-2 py-3"><div className="h-2.5 bg-zinc-900 rounded w-3/4" /></div>
      ))}
    </div>
  );
}

function schemeLabel(scheme: string) {
  const labels: Record<string, string> = {
    channels: "channel / year / month",
    sessions: "client / year / month",
    kinds: "record type",
    conversations: "year / month",
    web: "domain / path",
    records: "record type",
    files: "folder / file",
  };
  return labels[scheme] || scheme;
}

function formatDate(timestamp: number) {
  return new Date(timestamp * 1000).toISOString().slice(0, 10);
}

function formatChars(value: number) {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}m chars`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k chars`;
  return `${value} chars`;
}
