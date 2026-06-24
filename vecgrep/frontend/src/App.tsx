import { useEffect, useState } from "react";
import { api, Corpus, SearchHit, SearchMode } from "./api";
import SearchBar from "./components/SearchBar";
import IndexPanel from "./components/IndexPanel";
import CorpusList from "./components/CorpusList";
import ResultList from "./components/ResultList";
import Legend from "./components/Legend";
import HowSearchWorks from "./components/HowSearchWorks";
import AboutFooter from "./components/AboutFooter";
import TuningPanel from "./components/TuningPanel";
import {
  loadTuning,
  saveTuning,
  hasSavedTuning,
  tuningFromCalibration,
  Tuning,
} from "./tuning";

export default function App() {
  const [corpora, setCorpora] = useState<Corpus[]>([]);
  const [selectedCorpus, setSelectedCorpus] = useState<string | null>(null);
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tuning, setTuning] = useState<Tuning>(() => loadTuning());
  // Until the user customizes the sliders, seed them from each search's server
  // calibration so the displayed % matches the server for whatever model the
  // corpus uses. Once they drag a slider, their values win.
  const [tuningCustom, setTuningCustom] = useState<boolean>(() => hasSavedTuning());

  const updateTuning = (t: Tuning) => {
    setTuning(t);
    saveTuning(t);
    setTuningCustom(true);
  };

  const refresh = async () => {
    try {
      setCorpora(await api.listCorpora());
    } catch (e) {
      setError(String((e as Error).message ?? e));
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const onSearch = async (
    query: string,
    topK: number,
    mode: SearchMode,
    rerank: boolean
  ) => {
    setError(null);
    setSearching(true);
    try {
      const r = await api.search(query, selectedCorpus, topK, mode, rerank);
      setHits(r.hits);
      // Seed the tuning sliders from the server's actual calibration for this
      // corpus's model — but only while the user hasn't customized them.
      if (!tuningCustom && r.calibration) {
        setTuning((prev) => tuningFromCalibration(r.calibration!, prev));
      }
    } catch (e) {
      setError(String((e as Error).message ?? e));
      setHits([]);
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-zinc-800 px-6 py-4 flex items-center justify-between">
        <div className="flex items-baseline gap-3">
          <h1 className="text-xl font-mono font-semibold tracking-tight">vecgrep</h1>
          <span className="text-zinc-500 text-sm">grep for meaning, not keywords.</span>
        </div>
        <div className="flex items-center gap-4">
          {/* Optional companion-app link, only renders if VITE_COMPANION_URL */}
          {/* is set at build time. Empty by default to keep OSS builds clean   */}
          {/* (no internal hostnames in source per CLAUDE.md "OSS — no PII").   */}
          {import.meta.env.VITE_COMPANION_URL && (
            <a
              href={import.meta.env.VITE_COMPANION_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-mono text-zinc-500 hover:text-zinc-200 transition-colors"
              title={import.meta.env.VITE_COMPANION_LABEL || "companion app"}
            >
              {import.meta.env.VITE_COMPANION_LABEL || "companion"} ↗
            </a>
          )}
          <span className="text-xs text-zinc-500 font-mono">
            {corpora.length} corp{corpora.length === 1 ? "us" : "ora"}
          </span>
        </div>
      </header>

      <main className="flex-1 grid grid-cols-12 gap-6 px-6 py-6 max-w-7xl mx-auto w-full">
        <aside className="col-span-12 md:col-span-3 space-y-6">
          <IndexPanel onIndexed={refresh} />
          <CorpusList
            corpora={corpora}
            selected={selectedCorpus}
            onSelect={setSelectedCorpus}
            onDeleted={refresh}
          />
          <Legend />
          <HowSearchWorks />
        </aside>

        <section className="col-span-12 md:col-span-9 space-y-4">
          <SearchBar
            onSearch={onSearch}
            disabled={searching}
            corpus={selectedCorpus}
            corpusCount={corpora.length}
          />
          <TuningPanel tuning={tuning} onChange={updateTuning} />
          {error && (
            <div className="text-sm text-red-400 font-mono whitespace-pre-wrap border border-red-800 bg-red-950/40 rounded p-3">
              {error}
            </div>
          )}
          <ResultList hits={hits} searching={searching} tuning={tuning} />
          <AboutFooter />
        </section>
      </main>
    </div>
  );
}
