import { useEffect, useMemo, useState } from "react";
import {
  api,
  ExplorerDocument,
  ExplorerListing,
  ExplorerSort,
  ExplorerSource,
} from "../api";

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
  }, [corpus, path, query, sort, offset]);

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

  return (
    <div className="space-y-3">
      <header className="border border-zinc-800 rounded-xl bg-zinc-950/30 px-3 py-2.5 flex flex-col lg:flex-row lg:items-center gap-3">
        <div className="min-w-0 lg:mr-auto">
          <div className="text-sm text-zinc-200 truncate">{corpus}</div>
          <div className="text-[10px] font-mono text-zinc-600">
            {listing ? schemeLabel(listing.scheme) : "Building document catalog"}
          </div>
        </div>
        <form onSubmit={submitFilter} className="flex min-w-0 flex-1 lg:max-w-xl gap-2">
          <input
            value={queryInput}
            onChange={(event) => setQueryInput(event.target.value)}
            placeholder="Filter names, paths, or tags"
            className="min-w-0 flex-1 bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-xs text-zinc-200 placeholder:text-zinc-700 focus:outline-none focus:border-zinc-600"
          />
          <button
            type="submit"
            className="border border-zinc-700 rounded-lg px-3 text-[10px] font-mono text-zinc-400 hover:text-zinc-100 hover:border-zinc-500"
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
          className="bg-zinc-900 border border-zinc-800 rounded-lg px-2 py-2 text-[10px] font-mono text-zinc-400 focus:outline-none"
        >
          <option value="name">Name</option>
          <option value="newest">Newest</option>
          <option value="oldest">Oldest</option>
        </select>
      </header>

      {error && (
        <div className="border border-red-900 bg-red-950/30 rounded-lg p-3 text-sm text-red-400 font-mono">
          {error}
        </div>
      )}

      <div className="grid lg:grid-cols-[minmax(190px,0.7fr)_minmax(260px,1fr)_minmax(340px,1.45fr)] border border-zinc-800 rounded-xl overflow-hidden bg-zinc-950/20 min-h-[640px]">
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
      <div className="px-2 py-2 border-b border-zinc-800/70 flex flex-wrap gap-1 text-[10px] font-mono">
        <button onClick={() => onOpen([])} className="text-zinc-500 hover:text-zinc-200">Root</button>
        {path.map((segment, index) => (
          <span key={`${segment}-${index}`} className="min-w-0 flex items-center gap-1">
            <span className="text-zinc-800">/</span>
            <button
              onClick={() => onOpen(path.slice(0, index + 1))}
              className="text-zinc-500 hover:text-zinc-200 truncate max-w-[150px]"
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
            className="w-full rounded-md px-2 py-2 text-left text-xs font-mono text-zinc-600 hover:bg-zinc-900 hover:text-zinc-300"
          >
            .. / Parent
          </button>
        )}
        {listing?.folders.map((folder) => (
          <button
            key={folder.path.join("\u0000")}
            onClick={() => onOpen(folder.path)}
            className="w-full grid grid-cols-[minmax(0,1fr)_auto] gap-2 rounded-md px-2 py-2 text-left hover:bg-zinc-900 group"
          >
            <span className="truncate text-xs text-zinc-400 group-hover:text-zinc-100" title={folder.name}>
              {folder.name}
            </span>
            <span className="text-[10px] font-mono tabular-nums text-zinc-700">
              {folder.document_count.toLocaleString()}
            </span>
          </button>
        ))}
        {!busy && listing && !listing.folders.length && (
          <div className="px-2 py-8 text-center text-xs text-zinc-700">No subfolders</div>
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
        {documents.map((document) => (
          <button
            key={document.source_id}
            onClick={() => onOpen(document.source_id)}
            className={`w-full px-3 py-2.5 text-left transition-colors ${
              selectedId === document.source_id
                ? "bg-zinc-800/80"
                : "hover:bg-zinc-900/70"
            }`}
          >
            <div className="flex items-start gap-2">
              <span className="min-w-0 flex-1 truncate text-xs text-zinc-300" title={document.name}>
                {document.name}
              </span>
              {document.doc_timestamp && (
                <time className="shrink-0 text-[9px] font-mono text-zinc-700">
                  {formatDate(document.doc_timestamp)}
                </time>
              )}
            </div>
            <div className="mt-1 flex items-center gap-1.5 min-w-0 text-[9px] font-mono text-zinc-700">
              <span>{document.chunk_count} chunks</span>
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
              <div className="mt-1 flex flex-wrap gap-1">
                {document.tags.slice(0, 4).map((tag) => (
                  <span key={tag} className="border border-zinc-800 rounded px-1 text-[9px] font-mono text-zinc-600">
                    {tag}
                  </span>
                ))}
              </div>
            )}
          </button>
        ))}
        {!busy && listing && !documents.length && (
          <div className="px-4 py-16 text-center text-xs text-zinc-600">
            {query ? "No document names, paths, or tags matched." : "This folder only contains subfolders."}
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
  busy,
  error,
}: {
  preview: ExplorerSource | null;
  listing: ExplorerListing | null;
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
      {busy && <div className="p-4"><LoadingRows count={8} /></div>}
      {error && <div className="p-4 text-xs font-mono text-red-400">{error}</div>}
      {!busy && preview && (
        <div className="flex flex-col h-[590px]">
          <div className="px-4 py-3 border-b border-zinc-800/70">
            <h3 className="text-sm text-zinc-200 break-words">{preview.name}</h3>
            <div className="mt-1 text-[9px] font-mono text-zinc-600 break-all">
              {preview.display_path.join(" / ")}
            </div>
            <div className="mt-2 flex flex-wrap gap-1.5 text-[9px] font-mono text-zinc-600">
              <span className="border border-zinc-800 rounded px-1.5 py-0.5">{preview.kind}</span>
              <span className="border border-zinc-800 rounded px-1.5 py-0.5">{preview.chunk_count} chunks</span>
              {preview.doc_timestamp && (
                <span className="border border-zinc-800 rounded px-1.5 py-0.5">
                  {formatDate(preview.doc_timestamp)}
                </span>
              )}
              {preview.tags.map((tag) => (
                <span key={tag} className="border border-zinc-800 rounded px-1.5 py-0.5">{tag}</span>
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
      {!busy && !error && !preview && (
        <div className="p-5 text-xs text-zinc-600 space-y-5">
          <p>Select a document to read it here. Folder navigation never loads chunk bodies.</p>
          {listing && (
            <>
              <div>
                <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-700 mb-2">Catalog</div>
                <div className="grid grid-cols-2 gap-2 font-mono text-[10px]">
                  {Object.entries(listing.facets.kinds).map(([kind, count]) => (
                    <div key={kind} className="border border-zinc-800 rounded-md px-2 py-1.5 flex justify-between gap-2">
                      <span className="truncate">{kind}</span><span>{count.toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </div>
              {topTags.length > 0 && (
                <div>
                  <div className="text-[9px] font-mono uppercase tracking-wider text-zinc-700 mb-2">Top tags</div>
                  <div className="flex flex-wrap gap-1.5">
                    {topTags.map(([tag, count]) => (
                      <span key={tag} className="border border-zinc-800 rounded px-1.5 py-0.5 text-[9px] font-mono">
                        {tag} · {count}
                      </span>
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
    <div className="h-[43px] px-3 border-b border-zinc-800 flex items-center justify-between gap-2">
      <h2 className="text-[10px] font-mono uppercase tracking-wider text-zinc-500">{title}</h2>
      {detail && <span className="text-[9px] font-mono text-zinc-700 truncate">{detail}</span>}
    </div>
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
