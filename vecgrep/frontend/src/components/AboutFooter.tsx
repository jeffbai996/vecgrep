/**
 * Bottom-of-page primer on how vecgrep finds things. Plain English,
 * no jargon without a definition. Hidden by default behind a <details>
 * so it doesn't crowd the search UI for repeat users.
 */
export default function AboutFooter() {
  return (
    <details className="border-t border-zinc-800 mt-8 pt-4 text-sm font-mono text-zinc-400 group">
      <summary className="cursor-pointer select-none hover:text-zinc-200 list-none flex items-center gap-2 text-zinc-500">
        <span className="inline-block transition-transform group-open:rotate-90">▸</span>
        <span className="uppercase tracking-wider text-xs">how does this work?</span>
      </summary>

      <div className="mt-4 space-y-4 leading-relaxed max-w-3xl">
        <section>
          <h3 className="text-zinc-200 font-semibold mb-1">two ways to find a result</h3>
          <p>
            vecgrep runs <span className="text-zinc-200">two retrievers</span> in
            parallel and fuses their rankings. each is good at different things:
          </p>
        </section>

        <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="border border-sky-900/40 rounded p-3 bg-sky-950/20">
            <div className="text-sky-300 text-xs uppercase tracking-wider mb-1">
              vector (semantic)
            </div>
            <p className="text-zinc-300">
              embeds your query and every chunk into a dense vector, then ranks
              by cosine similarity. each corpus pins the backend and model it
              was built with, so queries always use compatible vectors. the
              search dashboard&apos;s <span className="text-zinc-100">active model inventory</span>{" "}
              shows what this server is actually using instead of advertising
              historical alternatives.
            </p>
            <p className="text-zinc-400 text-xs mt-2">
              good at: paraphrase, concept-match, &ldquo;the idea is similar even
              though the words are different.&rdquo;
            </p>
            <p className="text-zinc-500 text-xs mt-1">
              weakness: has a noise floor for any English query &mdash; it always
              finds <em>something</em> close to your sentence. vecgrep drops
              vector hits whose cosine sits below the model&apos;s calibrated
              floor <em>before</em> fusion, so sub-noise matches can&apos;t flood
              the ranking; calibration then spreads the real range out.
            </p>
          </div>

          <div className="border border-emerald-900/40 rounded p-3 bg-emerald-950/20">
            <div className="text-emerald-300 text-xs uppercase tracking-wider mb-1">
              BM25 (keyword)
            </div>
            <p className="text-zinc-300">
              classical inverted-index search. ranks by{" "}
              <span className="text-zinc-100">term frequency &times; inverse document frequency</span>
              {" "}&mdash; rare words found in your query weight more, common
              ones (&ldquo;the&rdquo;, &ldquo;and&rdquo;) weigh less.
            </p>
            <p className="text-zinc-400 text-xs mt-2">
              good at: literal token match, names, technical terms, codenames.
              if the exact word is in the doc, BM25 finds it.
            </p>
            <p className="text-zinc-500 text-xs mt-1">
              weakness: blind to synonyms. &ldquo;car&rdquo; doesn&apos;t match a doc
              that only says &ldquo;automobile.&rdquo;
            </p>
          </div>
        </section>

        <section>
          <h3 className="text-zinc-200 font-semibold mb-1">hybrid (default)</h3>
          <p>
            both retrievers run, and results are fused with{" "}
            <span className="text-zinc-200">Reciprocal Rank Fusion</span>{" "}
            (RRF): each chunk&apos;s score is{" "}
            <code className="text-zinc-300">w / (k + rank)</code> summed across
            retrievers. vecgrep weights BM25 1.5&times; vector by default, so a
            literal-keyword hit floats above vector noise on short queries
            without crowding out semantic hits on long ones.
          </p>
          <p className="text-zinc-500 text-xs mt-2">
            override with <code className="text-zinc-400">VECGREP_BM25_WEIGHT</code>{" "}
            or pick a single retriever in the search bar (mode: vector / bm25).
          </p>
        </section>

        <section>
          <h3 className="text-zinc-200 font-semibold mb-1">what the badges mean</h3>
          <p>
            every result shows which retriever(s) placed it:{" "}
            <span className="text-emerald-300 border border-emerald-700/60 rounded px-1 mx-0.5 text-xs">K</span>
            keyword only,{" "}
            <span className="text-sky-300 border border-sky-700/60 rounded px-1 mx-0.5 text-xs">V</span>
            vector only,{" "}
            <span className="text-violet-300 border border-violet-700/60 rounded px-1 mx-0.5 text-xs">VK</span>
            both. <span className="text-zinc-100">VK and K are the strongest signals</span>
            {" "}&mdash; either both methods agreed, or your query was a literal
            match in the source.
          </p>
        </section>

        <section>
          <h3 className="text-zinc-200 font-semibold mb-1">why the % can lie</h3>
          <p>
            vector cosine and BM25 raw scores aren&apos;t comparable. a BM25-only
            hit&apos;s raw fused RRF score is around{" "}
            <span className="text-zinc-300">1.6%</span>, but if the keyword is
            genuinely in the source it&apos;s a <em>real</em> match. so vecgrep
            rescales the displayed % per query: the strongest BM25 hit reads at{" "}
            <span className="text-zinc-300">~90%</span> and weaker ones taper
            down toward a <span className="text-zinc-300">~25%</span> floor. the
            cosine % is a sigmoid <span className="text-zinc-100">calibrated
            per embedding model</span> (each model parks its noise floor and
            signal range in a different place), so the same % means the same
            thing across corpora. when both retrievers fire, the higher of the
            two signals wins &mdash; agreement should raise confidence, not
            average it down. the underlying ranking always uses the raw fused
            score; trust the badges and order over the absolute number.
          </p>
        </section>

        <section>
          <h3 className="text-zinc-200 font-semibold mb-1">the pipeline, in order</h3>
          <p>
            <code className="text-zinc-300">retrieve (vector + BM25) → RRF fuse → recency decay → dedup → optional rerank → top-k</code>
          </p>
          <ul className="text-xs text-zinc-500 mt-2 space-y-1 list-disc list-inside">
            <li>
              <span className="text-zinc-400">recency decay</span> (optional,
              per corpus): a hit&apos;s fused score is multiplied by{" "}
              <code className="text-zinc-400">0.5 ** (age_days / half_life)</code>,
              applied <em>before</em> the top-k cut so a fresh chunk can outrank
              a stale one and wording alone can&apos;t float old content up. off
              by default; undated chunks are never penalized. tuned fast for
              chat/journal corpora, off for durable reference.
            </li>
            <li>
              <span className="text-zinc-400">dedup</span>: the
              sentence-window chunker emits overlapping windows, so one passage
              can surface as several near-identical hits. vecgrep collapses
              same-source hits whose character ranges overlap, keeping the
              strongest, before truncating to top-k.
            </li>
            <li>
              by default only <span className="text-zinc-400">active</span>{" "}
              chunks are returned &mdash; if the write-tool has marked a chunk
              superseded by a newer version, the stale one is hidden unless you
              explicitly ask for it.
            </li>
          </ul>
        </section>

        <section className="pt-2 border-t border-zinc-800/50">
          <p className="text-xs text-zinc-500">
            optional: <span className="text-zinc-300">rerank</span> in the
            search bar runs the configured cross-encoder over the fused
            candidate pool. it reads query and chunk <em>together</em> instead
            of comparing pre-computed vectors &mdash; more precise on hard,
            paraphrase-heavy queries, but latency varies materially with the
            model and candidate pool and it can be a wash on easy literal
            queries, so it&apos;s off by default. when it&apos;s on, every hit also picks up a{" "}
            <span className="text-zinc-400">rerank</span> tag and the displayed
            % comes straight from the cross-encoder&apos;s own score (the
            cleanest P(relevant) proxy), not the cosine/BM25 mix.
          </p>
        </section>
      </div>
    </details>
  );
}
