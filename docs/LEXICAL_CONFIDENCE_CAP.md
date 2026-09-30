# Lexical-only display confidence

The per-query BM25 display ceiling is 39%, below the semantic related and
strong bands. The lexical-only label remains explicit. Raw BM25 and fused
RRF scores, corpus rank weights, and ordering are unchanged.

Synthetic service evaluation:

| Case | Previous display | New display | Retrieval |
| --- | --- | --- | --- |
| Highest lexical hit at raw score 0.001, 1, or 100 | 90% | 39% | Same raw score |
| Exact unique term in a two-document corpus | Up to 90% | Below 40% | Matching document retained |
| Absent term in that corpus | No result | No result | Unchanged |

Focused relevance, calibration, decay, and lexical tests: 32 passed.
Full branch: 1,141 test assertions passed. Unchanged baseline: 1,139 passed.
Both full runs aborted during native teardown after pytest's summary (exit
134), so these are assertion results rather than clean full-process exits.
No new failing test names were introduced. The focused run exited normally.
