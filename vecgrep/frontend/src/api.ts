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
  search: (
    query: string,
    corpus: string | null,
    top_k: number,
    mode: SearchMode = "hybrid",
    rerank = false,
    filters: string[] = []
  ) =>
    request<{ hits: SearchHit[]; calibration?: Calibration }>("/api/search", {
      method: "POST",
      body: JSON.stringify({ query, corpus, top_k, mode, rerank, filters }),
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
