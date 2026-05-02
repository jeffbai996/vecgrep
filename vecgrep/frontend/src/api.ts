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

export type SearchHit = {
  similarity_pct: number;
  chunk: string;
  context_before: string;
  context_after: string;
  source_id: string;
  corpus: string;
  metadata: Record<string, unknown>;
};

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
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
  search: (query: string, corpus: string | null, top_k: number) =>
    request<{ hits: SearchHit[] }>("/api/search", {
      method: "POST",
      body: JSON.stringify({ query, corpus, top_k }),
    }),
  config: () => request<Record<string, unknown>>("/api/config"),
};
