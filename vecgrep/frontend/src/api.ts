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

async function adminRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  let token = "";
  try {
    token = window.localStorage.getItem("vecgrep_admin_token") || "";
  } catch {
    /* loopback access does not need a token */
  }
  return request<T>(path, {
    ...init,
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...((init.headers as Record<string, string>) || {}),
    },
  });
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
    request<{ hits: SearchHit[]; calibration?: Calibration }>("/api/search", {
      method: "POST",
      body: JSON.stringify({ query, corpus, top_k, mode, rerank }),
    }),
  config: () => request<Record<string, unknown>>("/api/config"),
  adminConfig: () => adminRequest<AdminConfig>("/api/admin/config"),
  updateAdminConfig: (values: Record<string, unknown>, confirm_qdrant?: string) =>
    adminRequest<AdminConfigUpdate>("/api/admin/config", {
      method: "PATCH",
      body: JSON.stringify({ values, confirm_qdrant }),
    }),
  reloadAdminConfig: () =>
    adminRequest<AdminConfig & { reloaded: boolean }>("/api/admin/config/reload", {
      method: "POST",
    }),
  listBackups: () => adminRequest<Backup[]>("/api/admin/backups"),
  createBackup: () => adminRequest<Backup>("/api/admin/backups", { method: "POST" }),
  verifyBackup: (id: string) =>
    adminRequest<Backup>(`/api/admin/backups/${encodeURIComponent(id)}/verify`, { method: "POST" }),
  downloadBackup: async (id: string) => {
    let token = "";
    try { token = window.localStorage.getItem("vecgrep_admin_token") || ""; } catch { /* loopback */ }
    const response = await fetch(`/api/admin/backups/${encodeURIComponent(id)}/download`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    if (!response.ok) throw new Error(`Download failed: HTTP ${response.status}`);
    return response.blob();
  },
  restoreBackup: (id: string, confirm: string) =>
    adminRequest<{ restored: string; safety_backup: string }>(`/api/admin/backups/${encodeURIComponent(id)}/restore`, {
      method: "POST", body: JSON.stringify({ confirm }),
    }),
  backupSchedule: () => adminRequest<BackupSchedule>("/api/admin/backup-schedule"),
  updateBackupSchedule: (values: Partial<BackupSchedule>) =>
    adminRequest<BackupSchedule>("/api/admin/backup-schedule", {
      method: "PATCH", body: JSON.stringify({ values }),
    }),
  getChunk: (corpus: string, chunkId: string, window: number | "full" = 2000) =>
    request<ChunkWindow>(
      `/api/chunk/${encodeURIComponent(corpus)}/${encodeURIComponent(chunkId)}?window=${window}`
    ),
};

export type AdminConfig = {
  values: Record<string, string | number | boolean | null>;
  home: string;
  provenance: Record<string, "default" | "file" | "env">;
  read_only: string[];
  secrets: Record<string, boolean>;
};

export type AdminConfigUpdate = AdminConfig & {
  restart_required: boolean;
  warnings: string[];
};

export type Backup = {
  backup_id: string;
  created_at: string;
  origin: "manual" | "scheduled" | "pre-restore";
  storage_mode: "embedded" | "server";
  path: string;
  size: number;
  corpora: Array<{ name: string; chunk_count: number }>;
  invalid?: boolean;
};

export type BackupSchedule = {
  backup_enabled: boolean;
  backup_frequency: "daily" | "weekly";
  backup_time: string;
  backup_weekday: number;
  backup_destination: string | null;
  backup_retention: number;
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
