import { useEffect, useState } from "react";
import { api, Corpus, SearchOptions, SearchResponse } from "./api";
import SearchBar from "./components/SearchBar";
import IndexPanel from "./components/IndexPanel";
import CorpusList from "./components/CorpusList";
import CorpusHealth from "./components/CorpusHealth";
import ResultList from "./components/ResultList";
import Legend from "./components/Legend";
import HowSearchWorks from "./components/HowSearchWorks";
import AboutFooter from "./components/AboutFooter";
import TuningPanel from "./components/TuningPanel";
import TimelinePanel from "./components/TimelinePanel";
import ComparePanel from "./components/ComparePanel";
import BrowsePanel from "./components/BrowsePanel";
import {
  loadTuning,
  saveTuning,
  clearTuning,
  hasSavedTuning,
  tuningFromCalibration,
  DEFAULT_TUNING,
  Tuning,
} from "./tuning";

export default function App() {
  const [view, setView] = useState<"search" | "timeline" | "compare" | "browse">("search");
  const [corpora, setCorpora] = useState<Corpus[]>([]);
  const [selectedCorpus, setSelectedCorpus] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [revealSource, setRevealSource] = useState<{
    corpus: string;
    sourceId: string;
    nonce: number;
  } | null>(null);
  const [tuning, setTuning] = useState<Tuning>(() => loadTuning());
  const [automaticTuning, setAutomaticTuning] = useState<Tuning>(() => ({
    ...DEFAULT_TUNING,
  }));
  // Until the user customizes the sliders, seed them from each search's server
  // calibration so the displayed % matches the server for whatever model the
  // corpus uses. Once they drag a slider, their values win.
  const [tuningCustom, setTuningCustom] = useState<boolean>(() => hasSavedTuning());

  const updateTuning = (t: Tuning) => {
    setTuning(t);
    saveTuning(t);
    setTuningCustom(true);
  };

  const resetTuning = () => {
    clearTuning();
    setTuningCustom(false);
    setTuning({ ...automaticTuning });
  };

  const primeSearch = (nextQuery: string) => {
    setQuery(nextQuery);
    requestAnimationFrame(() => {
      const input = document.getElementById("global-search-input") as HTMLInputElement | null;
      input?.focus();
      input?.setSelectionRange(nextQuery.length, nextQuery.length);
    });
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

  const onSearch = async (query: string, options: SearchOptions) => {
    setError(null);
    setSearching(true);
    try {
      const r = await api.search(query, selectedCorpus, options);
      setResponse(r);
      // Seed the tuning sliders from the server's actual calibration for this
      // corpus's model — but only while the user hasn't customized them.
      if (!tuningCustom && r.calibration) {
        const calibrated = tuningFromCalibration(r.calibration, DEFAULT_TUNING);
        setAutomaticTuning(calibrated);
        setTuning(calibrated);
      } else if (r.calibration) {
        setAutomaticTuning(tuningFromCalibration(r.calibration, DEFAULT_TUNING));
      }
    } catch (e) {
      setError(String((e as Error).message ?? e));
      setResponse({ hits: [], stubs: [] });
    } finally {
      setSearching(false);
    }
  };

  const revealInExplorer = (corpus: string, sourceId: string) => {
    setSelectedCorpus(corpus);
    setRevealSource({ corpus, sourceId, nonce: Date.now() });
    setView("browse");
  };

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-zinc-800 px-4 sm:px-6 py-3 flex items-center justify-between gap-3">
        <div className="flex items-baseline gap-3 min-w-0">
          <h1 className="text-xl font-mono font-semibold tracking-tight">vecgrep</h1>
          <span className="text-zinc-500 text-sm hidden sm:inline">grep for meaning, not keywords.</span>
        </div>
        <div className="flex items-center gap-3 min-w-0">
          <nav className="flex border border-zinc-800 rounded-lg p-0.5 overflow-x-auto" aria-label="Primary">
            {(["search", "timeline", "compare", "browse"] as const).map((item) => (
              <button
                key={item}
                onClick={() => setView(item)}
                className={`px-2.5 sm:px-3 h-7 text-[11px] font-mono rounded-md whitespace-nowrap ${view === item ? "bg-zinc-800 text-zinc-100" : "text-zinc-500 hover:text-zinc-300"}`}
              >
                {item[0].toUpperCase() + item.slice(1)}
              </button>
            ))}
          </nav>
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
          <span className="text-xs text-zinc-500 font-mono hidden lg:inline">
            {corpora.length} corp{corpora.length === 1 ? "us" : "ora"}
          </span>
        </div>
      </header>

      <main className="flex-1 grid grid-cols-12 gap-4 lg:gap-6 px-4 sm:px-6 py-5 max-w-[1600px] mx-auto w-full">
        <div className="col-span-12 md:hidden">
          <label className="flex items-center gap-2 border border-zinc-800 rounded-lg px-3 py-2 text-[11px] font-mono text-zinc-500">
            <span>Corpus</span>
            <select
              value={selectedCorpus || ""}
              onChange={(event) => setSelectedCorpus(event.target.value || null)}
              className="min-w-0 flex-1 bg-transparent text-zinc-200 focus:outline-none"
            >
              <option value="">All corpora</option>
              {corpora.map((corpus) => (
                <option key={corpus.name} value={corpus.name}>
                  {corpus.name} · {corpus.chunk_count.toLocaleString()} chunks
                </option>
              ))}
            </select>
          </label>
        </div>
        <aside className="hidden md:block md:col-span-3 xl:col-span-2 space-y-4">
          {view === "search" && <IndexPanel onIndexed={refresh} />}
          <CorpusList
            corpora={corpora}
            selected={selectedCorpus}
            onSelect={setSelectedCorpus}
            onDeleted={refresh}
          />
          {selectedCorpus && <CorpusHealth corpus={selectedCorpus} />}
          {view === "search" && (
            <>
              <Legend />
              <HowSearchWorks />
            </>
          )}
        </aside>

        {view === "search" ? (
          <section className="col-span-12 md:col-span-9 xl:col-span-10 space-y-3">
            <SearchBar
              query={query}
              onQueryChange={setQuery}
              onSearch={onSearch}
              disabled={searching}
              corpus={selectedCorpus}
              corpusCount={corpora.length}
            />
            <TuningPanel
              tuning={tuning}
              customized={tuningCustom}
              onChange={updateTuning}
              onReset={resetTuning}
            />
            {error && (
              <div className="text-sm text-red-400 font-mono whitespace-pre-wrap border border-red-800 bg-red-950/40 rounded p-3">
                {error}
              </div>
            )}
            <ResultList
              response={response}
              searching={searching}
              tuning={tuning}
              corpus={selectedCorpus}
              corpusCount={corpora.length}
              onPrimeQuery={primeSearch}
              onRevealSource={revealInExplorer}
            />
            <AboutFooter />
          </section>
        ) : (
          <section className="col-span-12 md:col-span-9 xl:col-span-10 space-y-4">
            {view === "timeline" && <TimelinePanel corpus={selectedCorpus} corpusCount={corpora.length} />}
            {view === "compare" && <ComparePanel corpus={selectedCorpus} />}
            {view === "browse" && (
              <BrowsePanel
                corpus={selectedCorpus}
                corpusNames={corpora.map((item) => item.name)}
                revealSource={revealSource}
                onRevealHandled={() => setRevealSource(null)}
              />
            )}
          </section>
        )}
      </main>
    </div>
  );
}
