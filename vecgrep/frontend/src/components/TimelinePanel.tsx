import { useState } from "react";
import { api, Incident, TimelineGroup } from "../api";

type Props = { corpus: string | null; corpusCount: number };
type Mode = "timeline" | "incident";

export default function TimelinePanel({ corpus, corpusCount }: Props) {
  const [mode, setMode] = useState<Mode>("timeline");
  const [query, setQuery] = useState("");
  const [filtersRaw, setFiltersRaw] = useState("");
  const [groups, setGroups] = useState<TimelineGroup[] | null>(null);
  const [incident, setIncident] = useState<Incident | null | undefined>(undefined);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim() || corpusCount === 0) return;
    setBusy(true);
    setError(null);
    const filters = filtersRaw.trim() ? filtersRaw.trim().split(/\s+/) : [];
    try {
      if (mode === "timeline") {
        setGroups(await api.timeline(query.trim(), corpus, 6, filters));
        setIncident(undefined);
      } else {
        setIncident(await api.incident(query.trim(), corpus, "hybrid", filters));
        setGroups(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      if (mode === "timeline") setGroups([]); else setIncident(null);
    } finally {
      setBusy(false);
    }
  }

  const placeholder = corpusCount === 0
    ? "Index a corpus first"
    : mode === "incident"
    ? "What happened, who was involved, and where is the evidence?"
    : "Reconstruct what happened around...";

  return (
    <div className="space-y-4">
      <form onSubmit={submit} className="border border-zinc-800 rounded-xl overflow-hidden bg-zinc-950/30">
        <div className="flex items-center gap-2 p-2">
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder={placeholder} disabled={busy || corpusCount === 0} className="min-w-0 flex-1 bg-transparent px-2 py-2.5 focus:outline-none placeholder:text-zinc-600 disabled:opacity-40" />
          <button disabled={busy || corpusCount === 0 || !query.trim()} className="bg-zinc-100 text-zinc-950 px-4 py-2 rounded-lg font-mono text-xs font-semibold disabled:opacity-40">{busy ? "Working" : mode === "timeline" ? "Reconstruct" : "Build brief"}</button>
        </div>
        <div className="border-t border-zinc-800 p-2 flex items-center gap-2 flex-wrap">
          <div className="flex bg-zinc-900 rounded-lg p-0.5 text-[11px] font-mono">
            <button type="button" onClick={() => setMode("timeline")} className={`px-3 py-1 rounded-md ${mode === "timeline" ? "bg-zinc-700 text-zinc-100" : "text-zinc-500"}`}>Timeline</button>
            <button type="button" onClick={() => setMode("incident")} className={`px-3 py-1 rounded-md ${mode === "incident" ? "bg-zinc-700 text-zinc-100" : "text-zinc-500"}`}>Incident brief</button>
          </div>
          <input value={filtersRaw} onChange={(e) => setFiltersRaw(e.target.value)} placeholder="Optional filters: after:7d speaker:alice source:*journal*" className="min-w-[240px] flex-1 bg-transparent px-2 py-1 text-[11px] font-mono text-zinc-400 placeholder:text-zinc-700 focus:outline-none" />
        </div>
      </form>

      {error && <div className="border border-red-900 bg-red-950/30 rounded-lg p-3 text-sm text-red-400 font-mono">{error}</div>}
      {mode === "timeline" ? (
        groups === null ? <Empty body="Chronological slices appear here, grouped by source instead of chopped into relevance-ranked chunks." /> : <TimelineGroups groups={groups} />
      ) : (
        incident === undefined ? <Empty body="Incident mode separates the strongest primary timeline from related evidence and names participants, sources, and time bounds." /> : incident ? <IncidentBrief incident={incident} /> : <Empty body="No incident evidence matched that query and filter set." />
      )}
    </div>
  );
}

function IncidentBrief({ incident }: { incident: Incident }) {
  return (
    <div className="space-y-4">
      <section className="border border-zinc-800 rounded-xl overflow-hidden">
        <header className="p-3 border-b border-zinc-800">
          <div className="flex items-center justify-between gap-3"><h2 className="text-sm text-zinc-200">{incident.title}</h2><span className="text-[10px] font-mono uppercase tracking-wide text-emerald-400">{incident.confidence}</span></div>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[10px] font-mono text-zinc-600">
            <span>{incident.sources.length} sources</span><span>{incident.participants.length} participants</span>{incident.time_range.start && <span>{incident.time_range.start} → {incident.time_range.end}</span>}
          </div>
          {incident.participants.length > 0 && <div className="mt-2 text-[10px] font-mono text-zinc-500">{incident.participants.join(" · ")}</div>}
        </header>
        <div className="p-3">
          <div className="text-[10px] font-mono uppercase tracking-wider text-zinc-600 mb-2">Primary · {shortSource(incident.primary_source)}</div>
          <EventList events={incident.primary_timeline} />
        </div>
      </section>
      {incident.related.length > 0 && <div><div className="text-[10px] font-mono uppercase tracking-wider text-zinc-600 mb-2">Related context kept separate</div><TimelineGroups groups={incident.related} /></div>}
    </div>
  );
}

function TimelineGroups({ groups }: { groups: TimelineGroup[] }) {
  if (!groups.length) return <Empty body="Nothing matched that query and window." />;
  return (
    <ol className="border border-zinc-800 rounded-xl overflow-hidden divide-y divide-zinc-800">
      {groups.map((group, index) => (
        <li key={`${group.source_id}-${index}`}>
          <header className="px-3 py-2 flex items-center gap-2 text-[10px] font-mono bg-zinc-950/30">
            {group.doc_timestamp && <time className="text-zinc-300">{new Date(group.doc_timestamp * 1000).toISOString().slice(0, 10)}</time>}
            <span className="text-zinc-500 truncate">{shortSource(group.source_id)}</span><span className="text-zinc-700 ml-auto">{group.corpus}</span>
          </header>
          <div className="border-t border-zinc-800/60">
            {group.events.length ? <EventList events={group.events} /> : <pre className="p-3 text-xs font-mono text-zinc-300 whitespace-pre-wrap">{group.slice_text}</pre>}
          </div>
        </li>
      ))}
    </ol>
  );
}

function EventList({ events }: { events: { time: string; speaker: string; text: string }[] }) {
  return <div className="divide-y divide-zinc-800/60">{events.map((event, index) => <div key={index} className="grid grid-cols-[3rem_8rem_minmax(0,1fr)] gap-2 px-2 py-1.5 text-xs font-mono"><span className="text-zinc-700">{event.time}</span><span className="text-sky-400 truncate">{event.speaker}</span><span className="text-zinc-300 whitespace-pre-wrap break-words">{event.text}</span></div>)}</div>;
}

function Empty({ body }: { body: string }) { return <div className="py-12 text-center text-sm text-zinc-600 max-w-xl mx-auto">{body}</div>; }
function shortSource(id: string) { const parts = id.split("/"); return id.startsWith("http") ? id : parts.slice(-2).join("/"); }
