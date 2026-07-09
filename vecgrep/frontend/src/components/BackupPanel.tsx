import { useEffect, useState } from "react";
import { api, Backup, BackupSchedule } from "../api";

const weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

export default function BackupPanel() {
  const [backups, setBackups] = useState<Backup[]>([]);
  const [schedule, setSchedule] = useState<BackupSchedule | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [confirmation, setConfirmation] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = async () => {
    const [items, timing] = await Promise.all([api.listBackups(), api.backupSchedule()]);
    setBackups(items); setSchedule(timing);
  };
  useEffect(() => { refresh().catch(e => setMessage(String(e.message || e))); }, []);

  const act = async (fn: () => Promise<unknown>, done: string) => {
    setBusy(true); setMessage("");
    try { await fn(); await refresh(); setMessage(done); }
    catch (e) { setMessage(String((e as Error).message ?? e)); }
    finally { setBusy(false); }
  };
  const download = async (item: Backup) => {
    const blob = await api.downloadBackup(item.backup_id);
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = item.path.split("/").pop() || `${item.backup_id}.vgbak`;
    anchor.click(); URL.revokeObjectURL(url);
  };

  return <section className="border-t border-zinc-800 pt-6 space-y-5">
    <div className="flex items-center justify-between gap-4">
      <div><h2 className="text-lg font-semibold">Backups</h2><p className="text-xs text-zinc-500 mt-1">Verified Qdrant snapshots and authoritative metadata</p></div>
      <button disabled={busy} onClick={() => act(api.createBackup, "Backup created and verified.")} className="h-9 px-3 rounded bg-emerald-600 text-zinc-950 text-sm font-medium disabled:opacity-40">Create backup</button>
    </div>

    <div className="overflow-x-auto border border-zinc-800 rounded">
      <table className="w-full text-sm">
        <thead className="text-left text-xs text-zinc-500 bg-zinc-900"><tr><th className="p-3">Created</th><th className="p-3">Origin</th><th className="p-3">Corpora</th><th className="p-3">Size</th><th className="p-3"></th></tr></thead>
        <tbody>{backups.map(item => <tr key={item.path} className="border-t border-zinc-800">
          <td className="p-3 font-mono text-xs">{item.created_at}</td><td className="p-3">{item.origin}</td>
          <td className="p-3">{item.corpora?.length ?? 0}</td><td className="p-3">{Math.ceil(item.size / 1024 / 1024)} MB</td>
          <td className="p-3 whitespace-nowrap text-right space-x-3">
            <button onClick={() => act(() => api.verifyBackup(item.backup_id), "Backup verified.")} className="text-emerald-400 hover:text-emerald-300">Verify</button>
            <button onClick={() => act(() => download(item), "Backup downloaded.")} className="text-zinc-400 hover:text-zinc-200">Download</button>
            <button onClick={() => { setConfirming(item.backup_id); setConfirmation(""); }} className="text-amber-400 hover:text-amber-300">Restore</button>
          </td>
        </tr>)}</tbody>
      </table>
      {!backups.length && <div className="p-5 text-sm text-zinc-500">No backups yet.</div>}
    </div>

    {confirming && <div className="border border-amber-800 bg-amber-950/20 rounded p-4 space-y-3">
      <label className="block text-sm text-amber-300">Type backup ID <span className="font-mono">{confirming}</span> to restore</label>
      <input value={confirmation} onChange={e => setConfirmation(e.target.value)} className="w-full h-10 px-3 rounded border border-amber-800 bg-zinc-950 font-mono text-sm" />
      <div className="flex gap-2"><button disabled={confirmation !== confirming || busy} onClick={() => act(() => api.restoreBackup(confirming, confirmation), "Restore completed; a safety backup was retained.").then(() => setConfirming(null))} className="h-9 px-3 rounded bg-amber-500 text-zinc-950 text-sm disabled:opacity-40">Restore</button><button onClick={() => setConfirming(null)} className="h-9 px-3 rounded border border-zinc-700 text-sm">Cancel</button></div>
    </div>}

    {schedule && <div className="space-y-3">
      <h3 className="text-sm font-medium">Schedule</h3>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <label className="text-sm flex items-center gap-2"><input type="checkbox" checked={schedule.backup_enabled} onChange={e => setSchedule({...schedule, backup_enabled:e.target.checked})} className="accent-emerald-500" />Enabled</label>
        <select value={schedule.backup_frequency} onChange={e => setSchedule({...schedule, backup_frequency:e.target.value as "daily"|"weekly"})} className="h-9 rounded border border-zinc-700 bg-zinc-950 px-2 text-sm"><option value="daily">Daily</option><option value="weekly">Weekly</option></select>
        <input type="time" value={schedule.backup_time} onChange={e => setSchedule({...schedule, backup_time:e.target.value})} className="h-9 rounded border border-zinc-700 bg-zinc-950 px-2 text-sm" />
        {schedule.backup_frequency === "weekly" && <select value={schedule.backup_weekday} onChange={e => setSchedule({...schedule, backup_weekday:Number(e.target.value)})} className="h-9 rounded border border-zinc-700 bg-zinc-950 px-2 text-sm">{weekdays.map((day,index)=><option key={day} value={index}>{day}</option>)}</select>}
        <input type="number" min="1" value={schedule.backup_retention} onChange={e => setSchedule({...schedule, backup_retention:Number(e.target.value)})} aria-label="Scheduled backups to retain" className="h-9 rounded border border-zinc-700 bg-zinc-950 px-2 text-sm" />
        <input value={schedule.backup_destination || ""} onChange={e => setSchedule({...schedule, backup_destination:e.target.value || null})} placeholder="Default backup directory" className="h-9 rounded border border-zinc-700 bg-zinc-950 px-2 text-sm" />
      </div>
      <button disabled={busy} onClick={() => act(() => api.updateBackupSchedule(schedule), "Backup schedule saved.")} className="h-9 px-3 rounded border border-zinc-700 text-sm">Save schedule</button>
    </div>}
    {message && <p className="text-sm text-zinc-300">{message}</p>}
  </section>;
}
