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
              embeds your query and every chunk into a 768-dim vector with
              <span className="text-zinc-100"> nomic-embed-text</span> (or
              OpenAI&apos;s <span className="text-zinc-100">text-embedding-3-small</span>),
              then ranks by cosine similarity.
            </p>
            <p className="text-zinc-400 text-xs mt-2">
              good at: paraphrase, concept-match, &ldquo;the idea is similar even
              though the words are different.&rdquo;
            </p>
            <p className="text-zinc-500 text-xs mt-1">
              weakness: floors at ~70&ndash;75% similarity for any English query
              (it always finds <em>something</em> close to your sentence).
              short rare-word queries can drown in noise.
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
            hit&apos;s raw fused score is &lt; 2%, but if the keyword is genuinely
            in the source it&apos;s a <em>real</em> match. vecgrep rescales the
            displayed % per query so BM25-only hits land in a readable band
            (60&ndash;90%); the underlying ranking is unchanged. trust the
            badges and ranking order, not the absolute number.
          </p>
        </section>

        <section className="pt-2 border-t border-zinc-800/50">
          <p className="text-xs text-zinc-500">
            optional: <span className="text-zinc-300">rerank</span> in the
            search bar runs a cross-encoder (BAAI/bge-reranker-base) over the
            top results. expensive but more precise &mdash; useful when the
            top-N look noisy and you want the model to look at query and
            chunk together.
          </p>
        </section>
      </div>
    </details>
  );
}
