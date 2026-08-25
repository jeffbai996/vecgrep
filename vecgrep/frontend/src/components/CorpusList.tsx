import { Corpus, api } from "../api";
import { corpusTone } from "../browseTones";

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
        {corpora.map((c) => {
          const tone = corpusTone(c.name);
          return (
            <div
              key={c.name}
              onClick={() => onSelect(c.name)}
              className={`group cursor-pointer border-l-2 px-2 py-1.5 rounded-r flex items-center justify-between transition-colors ${
                selected === c.name
                  ? tone.selected
                  : "border-transparent text-zinc-400 hover:border-zinc-700 hover:bg-zinc-900"
              }`}
            >
              <div className="min-w-0">
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${tone.dot}`} />
                  <div className={`text-xs font-mono truncate ${selected === c.name ? tone.text : ""}`}>
                    {c.name}
                  </div>
                </div>
                <div className="pl-3 text-[10px] text-zinc-600 font-mono">
                  {c.doc_count} docs · {c.chunk_count} chunks · {c.embed_model}
                </div>
              </div>
              <button
                onClick={(e) => onDelete(c.name, e)}
                title="delete"
                className="opacity-0 group-hover:opacity-100 text-zinc-500 hover:text-rose-400 px-1 text-xs"
              >
                x
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
