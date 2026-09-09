import { useCallback, useEffect, useState } from "react";
import { api, InstanceHealth as Health } from "../api";

/** Whole-instance health, so "is vecgrep well" is a page rather than a question
 *  you have to ask someone.
 *
 *  Every row is a check the backend already made; this renders the verdict and
 *  the figure behind it. A tick with no number beside it is a dashboard that
 *  cannot be argued with, which is the wrong kind.
 */

function bytes(n: number | null | undefined): string {
  if (!n && n !== 0) return "—";
  const unit = ["B", "KB", "MB", "GB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < unit.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v >= 10 || i === 0 ? Math.round(v) : v.toFixed(1)} ${unit[i]}`;
}

const num = (n: number | null | undefined) =>
  n || n === 0 ? n.toLocaleString() : "—";

function Verdict({ ok }: { ok: boolean | null }) {
  const [tone, mark, label] =
    ok === true
      ? ["text-emerald-400", "●", "pass"]
      : ok === false
        ? ["text-red-400", "▲", "fail"]
        : ["text-zinc-600", "○", "not applicable"];
  return (
    <span className={`font-mono text-xs ${tone}`} title={label} aria-label={label}>
      {mark}
    </span>
  );
}

export default function InstanceHealth() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);

  const load = useCallback(() => {
    setLoading(true);
    api
      .health()
      .then((h) => {
        setHealth(h);
        setError("");
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(load, [load]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-[11px] font-mono uppercase tracking-wider text-zinc-300">
          Instance health
        </h2>
        <button
          type="button"
          onClick={load}
          disabled={loading}
          className="rounded-md border border-zinc-800 px-2.5 py-1 text-[10px] font-mono text-zinc-400 hover:bg-zinc-900/40 hover:text-zinc-200 disabled:opacity-50"
        >
          {loading ? "checking" : "re-check"}
        </button>
      </div>

      {error && (
        <p className="rounded-lg border border-red-900/50 bg-red-950/20 p-3 text-xs text-red-300">
          {error}
        </p>
      )}

      {health && (
        <>
          <ul className="divide-y divide-zinc-900 rounded-xl border border-zinc-800 bg-zinc-950/20">
            {health.checks.map((c) => (
              <li key={c.name} className="flex items-baseline gap-3 px-3 py-2.5">
                <Verdict ok={c.ok} />
                <span className="w-36 shrink-0 font-mono text-[11px] text-zinc-300">
                  {c.name}
                </span>
                <span className="min-w-0 flex-1 break-words text-[11px] text-zinc-500">
                  {c.detail}
                </span>
              </li>
            ))}
          </ul>

          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {[
              ["chunks", num(health.chunks)],
              ["embeddings cached", num(health.cache.rows)],
              ["cache on disk", bytes(health.cache.bytes)],
              ["resident memory", bytes(health.process.rss_bytes)],
            ].map(([label, value]) => (
              <div
                key={label}
                className="rounded-lg border border-zinc-800 bg-zinc-950/40 px-3 py-2"
              >
                <div className="font-mono text-sm tabular-nums text-zinc-200">{value}</div>
                <div className="text-[9px] font-mono uppercase tracking-wide text-zinc-600">
                  {label}
                </div>
              </div>
            ))}
          </div>

          <div className="overflow-x-auto rounded-xl border border-zinc-800">
            <table className="w-full text-[11px]">
              <thead>
                <tr className="text-left font-mono text-[9px] uppercase tracking-wide text-zinc-600">
                  <th className="px-3 py-2">corpus</th>
                  <th className="px-3 py-2 text-right">documents</th>
                  <th className="px-3 py-2 text-right">chunks</th>
                  <th className="px-3 py-2">model</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-900">
                {health.corpora.map((c) => (
                  <tr key={c.name} className="text-zinc-400">
                    <td className="px-3 py-2 font-mono text-zinc-300">{c.name}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{num(c.documents)}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{num(c.chunks)}</td>
                    <td className="px-3 py-2 font-mono text-zinc-600">
                      {c.embed_model ?? "—"}
                      {c.dim ? ` · ${c.dim}d` : ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
