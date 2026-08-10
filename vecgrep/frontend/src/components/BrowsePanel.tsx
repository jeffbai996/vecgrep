import { useState } from "react";
import { api, BrowseGroup } from "../api";

type Props = { corpus: string | null };

export default function BrowsePanel({ corpus }: Props) {
  const [channel, setChannel] = useState("");
  const [date, setDate] = useState("");
  const [since, setSince] = useState("7d");
  const [until, setUntil] = useState("");
  const [sourcePath, setSourcePath] = useState("");
  const [tail, setTail] = useState(100);
  const [groups, setGroups] = useState<BrowseGroup[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!corpus) return;
    setBusy(true);
    setError(null);
    try {
      setGroups(await api.browse({
        corpus,
        channel: channel || undefined,
        date: date || undefined,
        since: date ? undefined : since || undefined,
        until: date ? undefined : until || undefined,
        source_path: sourcePath || undefined,
        tail,
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setGroups([]);
    } finally {
      setBusy(false);
    }
  }

  if (!corpus) {
    return <div className="py-12 text-center text-sm text-zinc-500">Choose one corpus to browse its sources and chronology.</div>;
  }

  return (
    <div className="space-y-4">
      <form onSubmit={submit} className="border border-zinc-800 rounded-xl overflow-hidden bg-zinc-950/30">
        <div className="px-3 py-2.5 flex items-center justify-between border-b border-zinc-800">
          <div><div className="text-sm text-zinc-300">Browse {corpus}</div><div className="text-[10px] text-zinc-600">Location-first reading without a search query</div></div>
          <button disabled={busy || !(channel || date || since || until || sourcePath)} className="bg-zinc-100 text-zinc-950 px-4 py-2 rounded-lg font-mono text-xs font-semibold disabled:opacity-40">{busy ? "Loading" : "Browse"}</button>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3 p-3">
          <Field label="Channel" value={channel} setValue={setChannel} placeholder="exact channel name" />
          <Field label="Exact date" value={date} setValue={setDate} placeholder="YYYY-MM-DD" />
          <Field label="Source path" value={sourcePath} setValue={setSourcePath} placeholder="*journal* or path glob" />
          <Field label="Since" value={since} setValue={setSince} placeholder="7d or YYYY-MM-DD" disabled={Boolean(date)} />
          <Field label="Until" value={until} setValue={setUntil} placeholder="today or YYYY-MM-DD" disabled={Boolean(date)} />
          <label className="text-[10px] font-mono text-zinc-600">Newest events
            <select value={tail} onChange={(e) => setTail(Number(e.target.value))} className="mt-1 w-full bg-zinc-900 border border-zinc-800 rounded-md px-2 py-1.5 text-xs text-zinc-300 focus:outline-none">
              <option value={25}>25</option><option value={50}>50</option><option value={100}>100</option><option value={250}>250</option><option value={500}>500</option>
            </select>
          </label>
        </div>
      </form>
      {error && <div className="border border-red-900 bg-red-950/30 rounded-lg p-3 text-sm text-red-400 font-mono">{error}</div>}
      {groups === null && !error && <div className="py-12 text-center text-sm text-zinc-600">Pick where or when, then read the actual sequence.</div>}
      {groups && <BrowseResults groups={groups} />}
    </div>
  );
}

function Field({ label, value, setValue, placeholder, disabled = false }: { label: string; value: string; setValue: (value: string) => void; placeholder: string; disabled?: boolean }) {
  return <label className="text-[10px] font-mono text-zinc-600">{label}<input value={value} onChange={(e) => setValue(e.target.value)} placeholder={placeholder} disabled={disabled} className="mt-1 w-full bg-zinc-900 border border-zinc-800 rounded-md px-2 py-1.5 text-xs text-zinc-300 placeholder:text-zinc-700 focus:outline-none focus:border-zinc-600 disabled:opacity-35" /></label>;
}

function BrowseResults({ groups }: { groups: BrowseGroup[] }) {
  if (!groups.length) return <div className="py-12 text-center text-sm text-zinc-600">Nothing matched those selectors.</div>;
  const eventCount = groups.reduce((sum, group) => sum + (group.events.length || 1), 0);
  return (
    <section className="border border-zinc-800 rounded-xl overflow-hidden">
      <header className="px-3 py-2 border-b border-zinc-800 text-[11px] font-mono text-zinc-500">{eventCount} event{eventCount === 1 ? "" : "s"} across {groups.length} source{groups.length === 1 ? "" : "s"}</header>
      <div className="divide-y divide-zinc-800">
        {groups.map((group) => <BrowseSource key={group.source_id} group={group} />)}
      </div>
    </section>
  );
}

function BrowseSource({ group }: { group: BrowseGroup }) {
  const [open, setOpen] = useState(true);
  return (
    <div>
      <button type="button" onClick={() => setOpen((value) => !value)} className="w-full px-3 py-2 flex items-center gap-2 text-left text-[11px] font-mono hover:bg-zinc-900/30">
        <span className={`text-zinc-600 transition-transform ${open ? "rotate-90" : ""}`}>▸</span>
        {group.doc_timestamp && <span className="text-zinc-300">{new Date(group.doc_timestamp * 1000).toISOString().slice(0, 10)}</span>}
        <span className="text-zinc-500 truncate">{shortSource(group.source_id)}</span>
        <span className="ml-auto text-zinc-700">{group.events.length || 1}</span>
      </button>
      {open && (group.events.length ? (
        <div className="divide-y divide-zinc-800/60 border-t border-zinc-800/60">
          {group.events.map((event, index) => (
            <div key={index} className="grid grid-cols-[3rem_8rem_minmax(0,1fr)] gap-2 px-4 py-1.5 text-xs font-mono">
              <span className="text-zinc-700">{event.time}</span><span className="text-sky-400 truncate">{event.speaker}</span><span className="text-zinc-300 whitespace-pre-wrap break-words">{event.text}</span>
            </div>
          ))}
        </div>
      ) : <pre className="border-t border-zinc-800/60 p-3 text-xs font-mono text-zinc-300 whitespace-pre-wrap">{group.slice_text}</pre>)}
    </div>
  );
}

function shortSource(id: string) { const parts = id.split("/"); return id.startsWith("http") ? id : parts.slice(-2).join("/"); }
