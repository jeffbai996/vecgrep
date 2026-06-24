/**
 * Collapsible "how search works" explainer for the sidebar. Covers the hybrid
 * retrieval pipeline, what reranking does and its tradeoff, and how to read the
 * score. Kept terse and in the same monospace/zinc style as Legend.
 */
import { useState } from "react";

export default function HowSearchWorks() {
  const [open, setOpen] = useState(false);

  return (
    <div className="border border-zinc-800 rounded p-3 text-xs font-mono">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between text-zinc-500 uppercase tracking-wider hover:text-zinc-300 transition-colors"
        aria-expanded={open}
      >
        <span>how search works</span>
        <span className="text-zinc-600">{open ? "−" : "+"}</span>
      </button>

      {open && (
        <div className="mt-3 space-y-3 text-zinc-400 leading-relaxed">
          <Section title="hybrid retrieval (default)">
            Two retrievers run in parallel: a <Hl>vector</Hl> search (embeddings —
            finds <em>meaning</em>) and <Hl>BM25</Hl> (keywords — finds exact
            words). Each returns its top 50; their rankings are fused with{" "}
            <Hl>Reciprocal Rank Fusion</Hl> (BM25 weighted 1.5&times; so genuine
            keyword hits float over vector noise), so a hit that both agree on (
            <span className="text-violet-300">VK</span>) rises to the top. Sub-noise
            vector hits are dropped before fusion, and overlapping near-duplicate
            chunks from one source are collapsed. The right default for almost
            everything.
          </Section>

          <Section title="rerank (opt-in)">
            With rerank on, the fused candidates are re-scored by a{" "}
            <Hl>cross-encoder</Hl> (BAAI/bge-reranker-base) — a slower model that
            reads the query and each chunk <em>together</em> instead of comparing
            pre-computed vectors. It&apos;s more accurate on hard, meaning-heavy
            queries where plain vector search whiffs.
            <br />
            <span className="text-amber-400/90">Tradeoff:</span> it adds roughly{" "}
            <Hl>~127ms</Hl> per search and doesn&apos;t help — can even hurt — on
            easy literal queries. So it&apos;s <Hl>off by default</Hl>; flip it on
            when a hybrid search returns near-misses for something you know is in
            there. Reranked hits also gain a <Hl>rerank</Hl> tag.
          </Section>

          <Section title="reading the score">
            The <Hl>%</Hl> is a calibrated relevance estimate, not a raw cosine.
            It&apos;s tuned per embedding model so the number means the same thing
            across corpora. Roughly: <span className="text-emerald-400">90%+</span>{" "}
            strong, <span className="text-amber-400">~50%</span> uncertain,{" "}
            <span className="text-zinc-500">&lt;30%</span> likely noise. BM25-only
            hits (no cosine) are rescaled per query into a{" "}
            <Hl>~25–90%</Hl> band so a real keyword match doesn&apos;t read as raw
            noise. When rerank is on the % comes straight from the cross-encoder.
            Hover any score to see the raw cosine / BM25 / RRF numbers.
          </Section>

          <Section title="recency">
            Some corpora apply <Hl>time decay</Hl> — older chunks are gently
            down-ranked so a stale match can&apos;t outrank a fresh one on wording
            alone. Tuned per corpus (fast for chat logs, off for durable
            reference).
          </Section>
        </div>
      )}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <div className="text-[10px] uppercase tracking-wider text-zinc-600">
        {title}
      </div>
      <p className="text-[11px]">{children}</p>
    </div>
  );
}

function Hl({ children }: { children: React.ReactNode }) {
  return <span className="text-zinc-200">{children}</span>;
}
