export type BrowseTone = {
  border: string;
  bg: string;
  text: string;
  dot: string;
  selected: string;
};

const tones: BrowseTone[] = [
  {
    border: "border-violet-800/70",
    bg: "bg-violet-950/35",
    text: "text-violet-300",
    dot: "bg-violet-400",
    selected: "border-violet-400 bg-violet-950/35",
  },
  {
    border: "border-sky-800/70",
    bg: "bg-sky-950/35",
    text: "text-sky-300",
    dot: "bg-sky-400",
    selected: "border-sky-400 bg-sky-950/35",
  },
  {
    border: "border-emerald-800/70",
    bg: "bg-emerald-950/35",
    text: "text-emerald-300",
    dot: "bg-emerald-400",
    selected: "border-emerald-400 bg-emerald-950/35",
  },
  {
    border: "border-amber-800/70",
    bg: "bg-amber-950/35",
    text: "text-amber-300",
    dot: "bg-amber-400",
    selected: "border-amber-400 bg-amber-950/35",
  },
  {
    border: "border-rose-800/70",
    bg: "bg-rose-950/35",
    text: "text-rose-300",
    dot: "bg-rose-400",
    selected: "border-rose-400 bg-rose-950/35",
  },
  {
    border: "border-cyan-800/70",
    bg: "bg-cyan-950/35",
    text: "text-cyan-300",
    dot: "bg-cyan-400",
    selected: "border-cyan-400 bg-cyan-950/35",
  },
  {
    border: "border-fuchsia-800/70",
    bg: "bg-fuchsia-950/35",
    text: "text-fuchsia-300",
    dot: "bg-fuchsia-400",
    selected: "border-fuchsia-400 bg-fuchsia-950/35",
  },
  {
    border: "border-lime-800/70",
    bg: "bg-lime-950/30",
    text: "text-lime-300",
    dot: "bg-lime-400",
    selected: "border-lime-400 bg-lime-950/30",
  },
];

export const neutralTone: BrowseTone = {
  border: "border-zinc-700",
  bg: "bg-zinc-900/70",
  text: "text-zinc-300",
  dot: "bg-zinc-500",
  selected: "border-zinc-400 bg-zinc-800/70",
};

const kindTones: Record<string, number> = {
  memory: 0,
  insight: 0,
  conversation: 1,
  session: 1,
  code: 2,
  file: 2,
  todo: 3,
  task: 3,
  correction: 4,
  decision: 4,
  reference: 5,
  analysis: 5,
  journal: 6,
  web: 7,
};

const schemeTones: Record<string, number> = {
  channels: 1,
  sessions: 0,
  kinds: 4,
  conversations: 6,
  web: 5,
  records: 3,
  files: 2,
};

export function corpusTone(corpus: string): BrowseTone {
  return tones[stableIndex(`corpus:${corpus}`)];
}

export function folderTone(path: string[]): BrowseTone {
  return tones[stableIndex(`folder:${path.join("/")}`)];
}

export function tagTone(tag: string): BrowseTone {
  return tones[stableIndex(`tag:${tag}`)];
}

export function kindTone(kind: string): BrowseTone {
  const key = kind.trim().toLowerCase();
  return tones[kindTones[key] ?? stableIndex(`kind:${key}`)];
}

export function schemeTone(scheme: string): BrowseTone {
  return tones[schemeTones[scheme] ?? stableIndex(`scheme:${scheme}`)];
}

function stableIndex(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) % tones.length;
}
