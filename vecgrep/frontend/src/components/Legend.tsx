/**
 * Sidebar legend — explains the V/K/VK badges and confidence-tier colors
 * at a glance. Kept compact: one-line per row, monospace, same color
 * tokens used in ResultList so what you see in the legend is what you
 * get on hits.
 */

const ROW = "flex items-center gap-2";

export default function Legend() {
  return (
    <div className="border border-zinc-800 rounded p-3 space-y-3 text-xs font-mono">
      <div className="text-zinc-500 uppercase tracking-wider">legend</div>

      <div className="space-y-1.5">
        <div className="text-[10px] uppercase tracking-wider text-zinc-600">
          match method
        </div>
        <div className={ROW}>
          <Badge tone="emerald">K</Badge>
          <span className="text-zinc-400">keyword (BM25)</span>
        </div>
        <div className={ROW}>
          <Badge tone="sky">V</Badge>
          <span className="text-zinc-400">semantic (vector)</span>
        </div>
        <div className={ROW}>
          <Badge tone="violet">VK</Badge>
          <span className="text-zinc-400">both retrievers</span>
        </div>
      </div>

      <div className="space-y-1.5">
        <div className="text-[10px] uppercase tracking-wider text-zinc-600">
          confidence
        </div>
        <div className={ROW}>
          <span className="text-emerald-400 font-semibold w-12">90.0%</span>
          <span className="text-emerald-500/80 uppercase tracking-wider text-[10px]">
            high
          </span>
        </div>
        <div className={ROW}>
          <span className="text-amber-400 font-semibold w-12">80.0%</span>
          <span className="text-amber-500/80 uppercase tracking-wider text-[10px]">
            soft
          </span>
        </div>
        <div className={ROW}>
          <span className="text-zinc-500 font-semibold w-12">72.0%</span>
          <span className="text-zinc-600 uppercase tracking-wider text-[10px]">
            weak
          </span>
        </div>
      </div>

      <p className="text-[10px] text-zinc-500 leading-relaxed">
        confidence considers <span className="text-zinc-400">both</span> the
        score <span className="text-zinc-400">and</span> the match method.
        only <span className="text-violet-300">VK</span> (both retrievers
        agreed) or a high score auto-reads as high &mdash; K alone goes
        soft so common keywords don&apos;t carpet the list green.
      </p>
    </div>
  );
}

function Badge({
  tone,
  children,
}: {
  tone: "emerald" | "sky" | "violet";
  children: React.ReactNode;
}) {
  const tones = {
    emerald: "bg-emerald-900/40 border-emerald-700/60 text-emerald-300",
    sky: "bg-sky-900/40 border-sky-700/60 text-sky-300",
    violet: "bg-violet-900/40 border-violet-700/60 text-violet-300",
  } as const;
  return (
    <span
      className={`text-[10px] font-bold border rounded px-1.5 py-px ${tones[tone]}`}
    >
      {children}
    </span>
  );
}
