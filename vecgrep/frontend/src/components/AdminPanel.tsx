import { FormEvent, useEffect, useState } from "react";
import { AdminConfig, api } from "../api";
import BackupPanel from "./BackupPanel";

const labels: Record<string, string> = {
  ollama_url: "Ollama primary URL",
  ollama_fallback_url: "Ollama fallback URL",
  embed_model: "Ollama embedding model",
  openai_embed_model: "OpenAI embedding model",
  api_host: "API host",
  api_port: "API port",
  default_top_k: "Default result count",
  oauth_enabled: "OAuth enabled",
  oauth_issuer_url: "OAuth issuer URL",
  qdrant_url: "Qdrant URL",
};

export default function AdminPanel() {
  const [config, setConfig] = useState<AdminConfig | null>(null);
  const [values, setValues] = useState<Record<string, string | number | boolean | null>>({});
  const [qdrantConfirm, setQdrantConfirm] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);

  const load = async () => {
    setError("");
    try {
      const next = await api.adminConfig();
      setConfig(next);
      setValues(next.values);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    }
  };

  useEffect(() => { load(); }, []);

  const changed = config
    ? Object.fromEntries(Object.entries(values).filter(([key, value]) => value !== config.values[key]))
    : {};
  const qdrantChanged = Object.prototype.hasOwnProperty.call(changed, "qdrant_url");

  const save = async (event: FormEvent) => {
    event.preventDefault();
    setSaving(true);
    setError("");
    setMessage("");
    try {
      const result = await api.updateAdminConfig(
        changed,
        qdrantChanged ? qdrantConfirm : undefined
      );
      setConfig(result);
      setValues(result.values);
      setQdrantConfirm("");
      const parts = ["Configuration applied."];
      if (result.restart_required) parts.push("Service restart required.");
      if (result.warnings.length) parts.push(...result.warnings);
      setMessage(parts.join(" "));
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setSaving(false);
    }
  };

  if (!config && !error) return <div className="text-sm text-zinc-500 font-mono">Loading configuration...</div>;

  return (
    <div className="max-w-5xl space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Runtime configuration</h2>
        {config && <p className="text-xs text-zinc-500 font-mono mt-1">{config.home}/config.json</p>}
      </div>

      {error && <div className="text-sm text-red-400 border border-red-900 bg-red-950/30 rounded p-3">{error}</div>}
      {message && <div className="text-sm text-emerald-300 border border-emerald-900 bg-emerald-950/30 rounded p-3">{message}</div>}

      {config && (
        <form onSubmit={save} className="space-y-5">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {Object.keys(labels).map((key) => {
              const value = values[key];
              const readOnly = config.read_only.includes(key);
              const source = config.provenance[key];
              return (
                <label key={key} className="block text-sm">
                  <span className="flex items-center justify-between mb-1.5">
                    <span className="text-zinc-300">{labels[key]}</span>
                    <span className={`text-[10px] uppercase font-mono ${source === "env" ? "text-amber-400" : "text-zinc-600"}`}>{source}</span>
                  </span>
                  {typeof value === "boolean" ? (
                    <input
                      type="checkbox"
                      checked={value}
                      disabled={readOnly}
                      onChange={(e) => setValues({ ...values, [key]: e.target.checked })}
                      className="h-4 w-4 accent-emerald-500"
                    />
                  ) : (
                    <input
                      type={key === "api_port" || key === "default_top_k" ? "number" : "text"}
                      value={value ?? ""}
                      readOnly={readOnly}
                      min={key === "api_port" || key === "default_top_k" ? 1 : undefined}
                      max={key === "api_port" ? 65535 : undefined}
                      onChange={(e) => setValues({
                        ...values,
                        [key]: e.target.type === "number" ? Number(e.target.value) : e.target.value,
                      })}
                      className="w-full h-10 px-3 rounded border border-zinc-700 bg-zinc-950 text-zinc-200 font-mono text-sm read-only:text-zinc-500 read-only:bg-zinc-900 focus:outline-none focus:border-emerald-600"
                    />
                  )}
                </label>
              );
            })}
          </div>

          {qdrantChanged && (
            <label className="block text-sm border border-amber-800 bg-amber-950/20 rounded p-3">
              <span className="block text-amber-300 mb-2">Type the new Qdrant URL to confirm the storage switch</span>
              <input
                value={qdrantConfirm}
                onChange={(e) => setQdrantConfirm(e.target.value)}
                className="w-full h-10 px-3 rounded border border-amber-800 bg-zinc-950 font-mono text-sm focus:outline-none"
              />
            </label>
          )}

          <div className="border-t border-zinc-800 pt-4">
            <h3 className="text-sm font-medium mb-3">Secrets</h3>
            <div className="flex flex-wrap gap-x-6 gap-y-2 text-xs font-mono text-zinc-400">
              {Object.entries(config.secrets).map(([key, present]) => (
                <span key={key}>{key}: <span className={present ? "text-emerald-400" : "text-zinc-600"}>{present ? "configured" : "not configured"}</span></span>
              ))}
            </div>
          </div>

          <div className="flex gap-3">
            <button
              type="submit"
              disabled={saving || Object.keys(changed).length === 0 || (qdrantChanged && qdrantConfirm !== values.qdrant_url)}
              className="h-10 px-4 rounded bg-emerald-600 text-zinc-950 font-medium text-sm disabled:opacity-40"
            >
              {saving ? "Applying..." : "Apply"}
            </button>
            <button
              type="button"
              onClick={async () => { await api.reloadAdminConfig(); await load(); setMessage("Configuration reloaded."); }}
              className="h-10 px-4 rounded border border-zinc-700 text-sm hover:border-zinc-500"
            >
              Reload
            </button>
          </div>
        </form>
      )}
      <BackupPanel />
    </div>
  );
}
