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

export type SearchHit = {
  similarity_pct: number;
  chunk: string;
  context_before: string;
  context_after: string;
  source_id: string;
  corpus: string;
  metadata: Record<string, unknown>;
  matched_by: string[];
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
    headers: {
      "Content-Type": "application/json",
      ..._authHeaders(),
      ...((init.headers as Record<string, string>) || {}),
    },
    ...init,
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
    rerank = false
  ) =>
    request<{ hits: SearchHit[] }>("/api/search", {
      method: "POST",
      body: JSON.stringify({ query, corpus, top_k, mode, rerank }),
    }),
  config: () => request<Record<string, unknown>>("/api/config"),
};
