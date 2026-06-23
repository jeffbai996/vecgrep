import { useState } from "react";
import { DEFAULT_TUNING, Tuning } from "../tuning";

type Props = {
  tuning: Tuning;
  onChange: (t: Tuning) => void;
};

type SliderRow = {
  key: keyof Tuning;
  label: string;
  min: number;
  max: number;
  step: number;
  help: string;
  format?: (v: number) => string;
};

const SLIDERS: SliderRow[] = [
  {
    key: "cosineCenter",
    label: "vector center",
    min: 0.4,
    max: 0.9,
    step: 0.01,
    help: "cosine value that maps to 50%. lower = scores read higher overall.",
    format: (v) => v.toFixed(2),
  },
  {
    key: "cosineSlope",
    label: "vector slope",
    min: 4,
    max: 24,
    step: 0.5,
    help: "sigmoid steepness. higher = sharper noise vs signal boundary.",
    format: (v) => v.toFixed(1),
  },
  {
    key: "bm25Top",
    label: "bm25 top",
    min: 60,
    max: 100,
    step: 1,
    help: "display % ceiling for the strongest bm25 hit in a query.",
    format: (v) => `${Math.round(v)}%`,
  },
  {
    key: "bm25Floor",
    label: "bm25 floor",
    min: 0,
    max: 50,
    step: 1,
    help: "display % floor for weakest bm25 hit. lower = wider spread.",
    format: (v) => `${Math.round(v)}%`,
  },
  {
    key: "bm25Bias",
    label: "bm25 bias",
    min: -20,
    max: 20,
    step: 1,
    help: "in hybrid, nudge scores toward bm25 (positive) or vector (negative).",
    format: (v) => (v > 0 ? `+${v}` : `${v}`),
  },
];

export default function TuningPanel({ tuning, onChange }: Props) {
  const [open, setOpen] = useState(false);

  const update = (key: keyof Tuning, value: number) => {
    onChange({ ...tuning, [key]: value });
  };

  const isDefault =
    tuning.cosineCenter === DEFAULT_TUNING.cosineCenter &&
    tuning.cosineSlope === DEFAULT_TUNING.cosineSlope &&
    tuning.bm25Top === DEFAULT_TUNING.bm25Top &&
    tuning.bm25Floor === DEFAULT_TUNING.bm25Floor &&
    tuning.bm25Bias === DEFAULT_TUNING.bm25Bias;

  return (
    <div className="border border-zinc-800 rounded">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full px-3 py-2 flex items-center justify-between text-xs font-mono text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/40 rounded transition-colors"
        title="adjust how raw retriever scores map to display percentages"
      >
        <span className="flex items-center gap-2">
          <span>{open ? "▾" : "▸"}</span>
          <span className="uppercase tracking-wider">score tuning</span>
          {!isDefault && (
            <span className="text-amber-500/70 text-[10px]">· modified</span>
          )}
        </span>
        {open && !isDefault && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onChange({ ...DEFAULT_TUNING });
            }}
            className="text-[10px] text-zinc-500 hover:text-zinc-200 underline-offset-2 hover:underline"
          >
            reset
          </button>
        )}
      </button>
      {open && (
        <div className="border-t border-zinc-800 p-3 space-y-3">
          {SLIDERS.map((s) => {
            const value = tuning[s.key] as number;
            const fmt = s.format ?? ((v: number) => v.toString());
            return (
              <div key={s.key}>
                <div className="flex items-baseline justify-between mb-0.5">
                  <label
                    className="text-[10px] font-mono uppercase tracking-wider text-zinc-500"
                    title={s.help}
                  >
                    {s.label}
                  </label>
                  <span className="text-[10px] font-mono text-zinc-300 tabular-nums">
                    {fmt(value)}
                  </span>
                </div>
                <input
                  type="range"
                  min={s.min}
                  max={s.max}
                  step={s.step}
                  value={value}
                  onChange={(e) => update(s.key, parseFloat(e.target.value))}
                  className="w-full accent-violet-400"
                />
              </div>
            );
          })}
          <div className="text-[10px] font-mono text-zinc-600 leading-relaxed pt-1">
            tuning re-derives display % from raw scores on the client — no
            re-query. saved to localStorage; reset to revert.
          </div>
        </div>
      )}
    </div>
  );
}
