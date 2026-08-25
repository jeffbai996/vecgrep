import { useState } from "react";
import { Tuning } from "../tuning";

type Props = {
  tuning: Tuning;
  customized: boolean;
  onChange: (t: Tuning) => void;
  onReset: () => void;
};

type SliderRow = {
  key: keyof Tuning;
  label: string;
  min: number;
  max: number;
  step: number;
  help: string;
  lowLabel: string;
  highLabel: string;
  format: (value: number) => string;
};

const SEMANTIC_SLIDERS: SliderRow[] = [
  {
    key: "cosineCenter",
    label: "Match threshold",
    min: 0.4,
    max: 0.9,
    step: 0.01,
    help: "The raw semantic match that reads as 50%. Raise it to score borderline matches more strictly.",
    lowLabel: "permissive",
    highLabel: "strict",
    format: (value) => value.toFixed(2),
  },
  {
    key: "cosineSlope",
    label: "Score contrast",
    min: 4,
    max: 30,
    step: 0.5,
    help: "How quickly semantic scores separate around the threshold. Higher values create clearer winners and losers.",
    lowLabel: "gentle",
    highLabel: "decisive",
    format: (value) => value.toFixed(1),
  },
];

const KEYWORD_SLIDERS: SliderRow[] = [
  {
    key: "bm25Top",
    label: "Best keyword hit",
    min: 60,
    max: 100,
    step: 1,
    help: "The displayed percentage assigned to the strongest exact-term hit in each query.",
    lowLabel: "reserved",
    highLabel: "confident",
    format: (value) => `${Math.round(value)}%`,
  },
  {
    key: "bm25Floor",
    label: "Weakest keyword hit",
    min: 0,
    max: 50,
    step: 1,
    help: "The displayed floor for exact-term results. Lower values spread weak and strong keyword hits farther apart.",
    lowLabel: "wide spread",
    highLabel: "compressed",
    format: (value) => `${Math.round(value)}%`,
  },
];

function vectorPct(raw: number, tuning: Tuning) {
  const x = tuning.cosineSlope * (raw - tuning.cosineCenter);
  return Math.round(100 / (1 + Math.exp(-x)));
}

function SliderControl({
  slider,
  value,
  onChange,
}: {
  slider: SliderRow;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block">
      <span className="flex items-baseline justify-between gap-3">
        <span className="text-[11px] font-mono text-zinc-300">{slider.label}</span>
        <span className="text-[11px] font-mono text-zinc-100 tabular-nums">
          {slider.format(value)}
        </span>
      </span>
      <span className="mt-1 block text-[10px] leading-relaxed text-zinc-600">
        {slider.help}
      </span>
      <input
        type="range"
        min={slider.min}
        max={slider.max}
        step={slider.step}
        value={value}
        onChange={(event) => onChange(parseFloat(event.target.value))}
        className="mt-2 block w-full accent-violet-400"
      />
      <span className="mt-0.5 flex justify-between text-[9px] font-mono text-zinc-700">
        <span>{slider.lowLabel}</span>
        <span>{slider.highLabel}</span>
      </span>
    </label>
  );
}

export default function TuningPanel({
  tuning,
  customized,
  onChange,
  onReset,
}: Props) {
  const [open, setOpen] = useState(false);

  const update = (key: keyof Tuning, value: number) => {
    onChange({ ...tuning, [key]: value });
  };

  const preview = [
    { label: "Below", raw: tuning.cosineCenter - 0.08 },
    { label: "Boundary", raw: tuning.cosineCenter },
    { label: "Above", raw: tuning.cosineCenter + 0.08 },
  ];
  const biasLabel =
    tuning.bm25Bias < -2
      ? "Semantic favored"
      : tuning.bm25Bias > 2
      ? "Keyword favored"
      : "Even weighting";

  return (
    <div className="overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950/20">
      <div className="flex items-stretch">
        <button
          type="button"
          onClick={() => setOpen(!open)}
          className="min-w-0 flex-1 px-3 py-2.5 flex items-center justify-between gap-3 text-left hover:bg-zinc-900/40 transition-colors"
          aria-expanded={open}
        >
          <span className="flex min-w-0 items-center gap-2">
            <span className="font-mono text-xs text-zinc-600">{open ? "▾" : "▸"}</span>
            <span className="min-w-0">
              <span className="block text-[11px] font-mono uppercase tracking-wider text-zinc-300">
                Score interpretation
              </span>
              <span className="mt-0.5 block truncate text-[10px] text-zinc-600">
                How raw matches become percentages
              </span>
            </span>
          </span>
          <span
            className={`shrink-0 rounded-full border px-2 py-0.5 text-[9px] font-mono ${
              customized
                ? "border-amber-900/70 bg-amber-950/25 text-amber-400"
                : "border-sky-900/70 bg-sky-950/25 text-sky-400"
            }`}
          >
            {customized ? "Custom" : "Automatic"}
          </span>
        </button>
        {customized && (
          <button
            type="button"
            onClick={onReset}
            className="border-l border-zinc-800 px-3 text-[10px] font-mono text-zinc-600 hover:bg-zinc-900/40 hover:text-zinc-200"
          >
            Use automatic
          </button>
        )}
      </div>

      {open && (
        <div className="border-t border-zinc-800 p-3 sm:p-4 space-y-3">
          <div className="rounded-lg border border-violet-900/50 bg-violet-950/15 p-3">
            <p className="text-xs leading-relaxed text-zinc-400">
              Search retrieval stays unchanged. This remaps raw signals into the
              displayed percentages and result order, so the numbers match how
              selective you want the result list to feel.
            </p>
            <p className="mt-1 text-[10px] font-mono text-zinc-600">
              Deep rerank uses its own score while enabled.
            </p>
          </div>

          <section className="rounded-lg border border-sky-900/45 bg-sky-950/10 p-3">
            <div className="flex items-center justify-between gap-3">
              <h3 className="text-[10px] font-mono uppercase tracking-wider text-sky-400">
                Semantic scoring
              </h3>
              <span className="text-[9px] font-mono text-zinc-700">live preview</span>
            </div>
            <div className="mt-2 grid grid-cols-3 gap-1.5">
              {preview.map((item) => (
                <div key={item.label} className="rounded-md border border-zinc-800 bg-zinc-950/50 px-2 py-2 text-center">
                  <div className="text-base font-mono tabular-nums text-zinc-200">
                    {vectorPct(item.raw, tuning)}%
                  </div>
                  <div className="text-[9px] font-mono uppercase tracking-wide text-zinc-700">
                    {item.label}
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-4 space-y-4">
              {SEMANTIC_SLIDERS.map((slider) => (
                <SliderControl
                  key={slider.key}
                  slider={slider}
                  value={tuning[slider.key]}
                  onChange={(value) => update(slider.key, value)}
                />
              ))}
            </div>
          </section>

          <section className="rounded-lg border border-emerald-900/45 bg-emerald-950/10 p-3">
            <h3 className="text-[10px] font-mono uppercase tracking-wider text-emerald-400">
              Keyword scoring
            </h3>
            <div className="mt-3 space-y-4">
              {KEYWORD_SLIDERS.map((slider) => (
                <SliderControl
                  key={slider.key}
                  slider={slider}
                  value={tuning[slider.key]}
                  onChange={(value) => update(slider.key, value)}
                />
              ))}
            </div>
          </section>

          <section className="rounded-lg border border-violet-900/45 bg-violet-950/10 p-3">
            <div className="flex items-baseline justify-between gap-3">
              <h3 className="text-[10px] font-mono uppercase tracking-wider text-violet-400">
                Hybrid balance
              </h3>
              <span className="text-[10px] font-mono text-zinc-400">{biasLabel}</span>
            </div>
            <p className="mt-1 text-[10px] leading-relaxed text-zinc-600">
              Break close calls toward meaning or exact terms when both retrievers find a result.
            </p>
            <input
              type="range"
              min={-20}
              max={20}
              step={1}
              value={tuning.bm25Bias}
              onChange={(event) => update("bm25Bias", parseFloat(event.target.value))}
              className="mt-3 block w-full accent-violet-400"
              aria-label="Hybrid balance"
            />
            <div className="mt-0.5 flex justify-between text-[9px] font-mono text-zinc-700">
              <span>semantic</span>
              <span>even</span>
              <span>keyword</span>
            </div>
          </section>

          <div className="flex flex-wrap items-center justify-between gap-2 px-1 text-[9px] font-mono text-zinc-700">
            <span>
              {customized
                ? "Custom calibration is saved in this browser."
                : "Automatic calibration follows the searched corpus model."}
            </span>
            <span>Changes apply instantly; no re-query.</span>
          </div>
        </div>
      )}
    </div>
  );
}
