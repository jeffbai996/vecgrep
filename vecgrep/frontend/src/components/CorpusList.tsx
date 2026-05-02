import { Corpus, api } from "../api";

type Props = {
  corpora: Corpus[];
  selected: string | null;
  onSelect: (name: string | null) => void;
  onDeleted: () => void;
};

export default function CorpusList({ corpora, selected, onSelect, onDeleted }: Props) {
  const onDelete = async (name: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm(`delete corpus '${name}'? this is irreversible.`)) return;
    try {
      await api.deleteCorpus(name);
      if (selected === name) onSelect(null);
      onDeleted();
    } catch (e) {
      alert(`error: ${(e as Error).message}`);
    }
  };

  return (
    <div className="border border-zinc-800 rounded p-3 space-y-2">
      <div className="text-xs text-zinc-500 font-mono uppercase tracking-wider">
        corpora
      </div>
      <div className="space-y-1">
        <button
          onClick={() => onSelect(null)}
          className={`w-full text-left text-xs font-mono px-2 py-1 rounded ${
            selected === null
              ? "bg-zinc-800 text-zinc-100"
              : "text-zinc-400 hover:bg-zinc-900"
          }`}
        >
          all
        </button>
        {corpora.map((c) => (
          <div
            key={c.name}
            onClick={() => onSelect(c.name)}
            className={`group cursor-pointer px-2 py-1 rounded flex items-center justify-between ${
              selected === c.name
                ? "bg-zinc-800 text-zinc-100"
                : "text-zinc-400 hover:bg-zinc-900"
            }`}
          >
            <div className="min-w-0">
              <div className="text-xs font-mono truncate">{c.name}</div>
              <div className="text-[10px] text-zinc-600 font-mono">
                {c.doc_count} docs · {c.chunk_count} chunks · {c.embed_model}
              </div>
            </div>
            <button
              onClick={(e) => onDelete(c.name, e)}
              title="delete"
              className="opacity-0 group-hover:opacity-100 text-zinc-500 hover:text-red-400 px-1 text-xs"
            >
              x
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
