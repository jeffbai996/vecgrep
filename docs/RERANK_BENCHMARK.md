# Offline reranker batching comparison

`python scripts/rerank_benchmark.py` compares two **clean committed trees** on
identical frozen query/candidate text. It defaults to `NOT_RUN`: planning does
not import Torch, load weights, contact a service, or read a corpus.

Use the parent of the length-batching merge as the baseline and the candidate
commit as the second tree. For PR #45 these are `2e57b8e` and `736c707`.
Create detached worktrees; keep inputs, reports and local model files outside
this public repository. Existing permitted evaluation gold should be preserved,
not regenerated from current search results. Frozen positive pools must contain
an expected source. Include uniform and mixed lengths, long outliers, CJK and
negative queries. Use a bounded, representative subset rather than a new corpus.

Input shape (one or more cases):

```json
{"cases":[{"id":"example","corpus":"notes","query":"queue admission timeout",
 "want":["admission.md"],"forbid":[],"negative":false,
 "candidates":[{"source_id":"admission.md","text":"Queue admission has a bounded timeout."}]}]}
```

Prepare a plan:

```sh
python scripts/rerank_benchmark.py \
  --baseline-root /tmp/vecgrep-baseline --candidate-root /tmp/vecgrep-candidate \
  --cases /tmp/frozen-candidates.json --output /tmp/rerank-report.json
```

After confirming a safe CPU work budget, repeat with `--run --model
/path/to/local/model-directory`. The model must already exist locally. Record
its origin/revision in the private evidence alongside the report (the report also
fingerprints local model-file bytes); never copy
weights or private gold into Git. The worker uses offline Hugging Face mode,
an isolated `VECGREP_HOME`, a temporary model-cache directory, CPU only and one
Torch thread by default. It never calls the HTTP service or clears a shared
cache. Maximums are 50 cases, 50 candidates each, 20 repetitions, 32 pairs per
batch, four CPU threads and a 30-minute subprocess timeout; defaults are five
repetitions, batch size eight and a ten-minute timeout.

The same loaded model and tokenizer score both variants; model input limits,
texts and batch size are held constant. AB/BA order alternates by repetition.
Warm-up is separate. Every **warm uncached** trial clears only that module's
isolated score cache and must call prediction. Every **score-cache hit** trial
primes that cache outside timing and must make zero prediction calls. Neither
measurement includes retrieval or HTTP. Initialization time is reported
separately; filesystem cache is uncontrolled, so it is not a cold-model latency
comparison.

Reports contain commit/tree/blob identities, fixture fingerprint, dependency
versions, CPU/platform, character/token distributions, tokenizer ceiling,
predict-call composition, per-trial scores/order and median/p95/max latency.
Recording overhead is included equally in both paths. Predict calls describe
inputs handed to the library; the library may sort/subbatch internally. Peak
RSS covers the whole worker with the shared model and cannot establish which
variant used less memory. Inspect sample count before interpreting tails.

Source-level hit@k, MRR, precision and negative-query metrics reuse
`vecgrep.eval.metrics`. These are **fixed-pool rerank diagnostics**, using sigmoid
scores and an explicit 60% negative floor. They do not establish full retrieval
quality or the application's final display-confidence behavior. End-to-end
retrieval comparisons still belong in the existing `vecgrep.eval` harness.

The output is invalidated before work starts; model failure, timeout or
interruption cannot retain a previous green result. A `NOT_RUN` plan or mocked
harness test is never speedup evidence. Publish a speedup claim only after
representative repeated uncached measurements, score/order checks and the
existing retrieval evaluation support it.
