// Thin client for the FastAPI backend. Uses relative URLs so the same
// build works whether served by Vite (dev, proxied) or by FastAPI itself.

export type Corpus = {
  name: string;
  embed_backend: string;
  embed_model: string;
  dim: number;
  chunker: string;
  doc_count: number;
  chunk_count: number;
  sources: string[];
  created_at: number;
  updated_at: number;
};

export type SearchMode = "hybrid" | "vector" | "bm25";

// The score-display calibration the server used for the searched corpus's embed
// model. The tuning UI seeds its sliders from this so the client-side re-derived
// % matches the server (the old hardcoded default drifted on non-nomic corpora).
export type Calibration = {
  cosine_center: number;
  cosine_slope: number;
  bm25_top: number;
  bm25_floor: number;
};

export type SearchHit = {
  similarity_pct: number;
  chunk: string;
  context_before: string;
  context_after: string;
  source_id: string;
  corpus: string;
  metadata: Record<string, unknown>;
  chunk_id: string;
  matched_by: string[];
  doc_timestamp: number | null;
  line_start: number | null;
  line_end: number | null;
  anchor: string;
  relevance_label: string;
  // Raw retriever scores — always populated by the server. Used by the
  // tuning UI to re-derive display % without re-querying.
  explain?: {
    vector_cosine?: number;
    vector_rank?: number;
    bm25_score?: number;
    bm25_rank?: number;
    bm25_max?: number;
    rrf?: number;
    rerank_score?: number;
  };
};

export type SearchStub = {
  chunk_id: string;
  corpus: string;
  source_id: string;
  doc_timestamp: number | null;
  snippet: string;
  similarity_pct: number;
};

export type SearchResponse = {
  hits: SearchHit[];
  stubs: SearchStub[];
  calibration?: Calibration;
};

export type SearchOptions = {
  topK: number;
  mode: SearchMode;
  rerank: boolean;
  filters: string[];
};

function _authHeaders(): Record<string, string> {
  // If the operator has set VECGREP_API_TOKEN server-side, the UI needs to
  // send the same token. We read it from localStorage so the user can drop
  // it in via DevTools without us shipping a login screen.
  try {
    const tok = window.localStorage.getItem("vecgrep_token");
    return tok ? { Authorization: `Bearer ${tok}` } : {};
  } catch {
    return {};
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ..._authHeaders(),
      ...((init.headers as Record<string, string>) || {}),
    },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      if (j.detail) detail = j.detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  listCorpora: () => request<Corpus[]>("/api/corpora"),
  deleteCorpus: (name: string) =>
    request<{ deleted: string }>(`/api/corpora/${encodeURIComponent(name)}`, {
      method: "DELETE",
    }),
  index: (source: string, corpus: string, chunker = "sentence_window") =>
    request<{ docs: number; chunks: number }>("/api/index", {
      method: "POST",
      body: JSON.stringify({ source, corpus, chunker }),
    }),
  search: (query: string, corpus: string | null, options: SearchOptions) =>
    request<SearchResponse>("/api/search", {
      method: "POST",
      body: JSON.stringify({
        query,
        corpus,
        mode: options.mode,
        rerank: options.rerank,
        filters: options.filters,
        explain: true,
        budget: true,
        full_k: Math.min(8, options.topK),
        max_total: options.topK,
        token_ceiling: 8000,
      }),
    }),
  timeline: (
    query: string,
    corpus: string | null,
    maxGroups = 4,
    filters: string[] = []
  ) =>
    request<TimelineGroup[]>("/api/timeline", {
      method: "POST",
      body: JSON.stringify({ query, corpus, max_groups: maxGroups, filters }),
    }),
  related: (corpus: string, chunkId: string, topK = 8) =>
    request<{ hits: SearchHit[] }>(
      `/api/related/${encodeURIComponent(corpus)}/${encodeURIComponent(chunkId)}?top_k=${topK}`
    ),
  stats: (corpus: string) =>
    request<CorpusStats>(`/api/stats/${encodeURIComponent(corpus)}`),
  config: () => request<Record<string, unknown>>("/api/config"),
  getChunk: (corpus: string, chunkId: string, window: number | "full" = 2000) =>
    request<ChunkWindow>(
      `/api/chunk/${encodeURIComponent(corpus)}/${encodeURIComponent(chunkId)}?window=${window}`
    ),
  compare: (query: string, corpus: string, windows: CompareWindows, topK = 10) =>
    request<CompareResponse>("/api/compare", {
      method: "POST",
      body: JSON.stringify({ query, corpus, ...windows, top_k: topK }),
    }),
  browse: (options: BrowseRequest) =>
    request<BrowseGroup[]>("/api/browse", {
      method: "POST",
      body: JSON.stringify(options),
    }),
  explore: (corpus: string, options: ExplorerOptions = {}) => {
    const params = new URLSearchParams();
    for (const segment of options.path || []) params.append("path", segment);
    if (options.query) params.set("q", options.query);
    if (options.sort) params.set("sort", options.sort);
    params.set("offset", String(options.offset || 0));
    params.set("limit", String(options.limit || 50));
    return request<ExplorerListing>(
      `/api/explorer/${encodeURIComponent(corpus)}?${params.toString()}`
    );
  },
  explorerSource: (corpus: string, sourceId: string, maxChars = 100_000) => {
    const params = new URLSearchParams({
      source_id: sourceId,
      max_chars: String(maxChars),
    });
    return request<ExplorerSource>(
      `/api/explorer/${encodeURIComponent(corpus)}/source?${params.toString()}`
    );
  },
  incident: (
    query: string,
    corpus: string | null,
    mode: SearchMode = "hybrid",
    filters: string[] = []
  ) =>
    request<Incident | null>("/api/incident", {
      method: "POST",
      body: JSON.stringify({ query, corpus, mode, filters }),
    }),
};

export type TimelineEvent = {
  speaker: string;
  time: string;
  text: string;
};

export type TimelineGroup = {
  corpus: string;
  source_id: string;
  doc_timestamp: number | null;
  slice_start: number;
  slice_end: number;
  events: TimelineEvent[];
  // Set only when the source is not a transcript (no parseable events).
  slice_text: string;
};

export type BrowseGroup = {
  corpus: string;
  source_id: string;
  doc_timestamp: number | null;
  events: TimelineEvent[];
  slice_text: string;
};

export type BrowseRequest = {
  corpus: string;
  channel?: string;
  date?: string;
  source_path?: string;
  since?: string;
  until?: string;
  tail?: number;
};

export type ExplorerSort = "name" | "newest" | "oldest";

export type ExplorerOptions = {
  path?: string[];
  query?: string;
  sort?: ExplorerSort;
  offset?: number;
  limit?: number;
};

export type ExplorerFolder = {
  name: string;
  path: string[];
  document_count: number;
  latest_timestamp: number | null;
};

export type ExplorerDocument = {
  source_id: string;
  name: string;
  parent_path: string[];
  display_path: string[];
  kind: string;
  doc_timestamp: number | null;
  chunk_count: number;
  tags: string[];
};

export type ExplorerListing = {
  corpus: string;
  scheme: string;
  path: string[];
  folders: ExplorerFolder[];
  documents: ExplorerDocument[];
  recent_documents: ExplorerDocument[];
  total_documents: number;
  offset: number;
  limit: number;
  has_more: boolean;
  facets: {
    kinds: Record<string, number>;
    tags: Record<string, number>;
  };
};

export type ExplorerSource = ExplorerDocument & {
  corpus: string;
  metadata: Record<string, unknown>;
  text: string;
  source_length: number;
  truncated: boolean;
};

export type CompareWindows = {
  a_after?: string;
  a_before?: string;
  b_after?: string;
  b_before?: string;
};

export type CompareSide = { results: SearchHit[]; sources: string[] };

export type CompareResponse = {
  windows: {
    a: { after: string | null; before: string | null };
    b: { after: string | null; before: string | null };
  };
  a: CompareSide;
  b: CompareSide;
  only_in_a: string[];
  only_in_b: string[];
  in_both: string[];
};

export type Incident = {
  title: string;
  confidence: string;
  sources: string[];
  participants: string[];
  time_range: { start: string; end: string };
  primary_source: string;
  primary_timeline: TimelineEvent[];
  related: TimelineGroup[];
};

export type CorpusStats = {
  corpus: string;
  chunks: number;
  docs: number;
  date_span: { first: string | null; last: string | null };
  days_covered: number;
  gap_days: number;
  sources: Record<string, number>;
  sources_truncated: boolean;
};

export type ChunkWindow = {
  corpus: string;
  chunk_id: string;
  source_id: string;
  chunk_start: number;
  chunk_end: number;
  before: string;
  chunk: string;
  after: string;
  source_length: number;
  window: number;
};
