# Response budget allocation comparison

Three fixed candidate sets (two repository queries and one transcript query)
were captured once with generous output limits, then replayed through the old
allocator at `4260886` and the passage-first allocator. All three searches
reranked 50 candidates. No storage, recency, retrieval, or ranking settings were
changed. Raw source content stays outside the repository.

The 8k/12k rows request six full passages and use the explore preview allowance;
16k/24k request ten and use the deep allowance. This compares within each row;
it is not a pure budget-only comparison between explore and deep. Each emitted
response was independently counted with cl100k_base and was within its cap.

| Sample | Budget | Before full / previews | After full / previews |
| --- | ---: | ---: | ---: |
| transcript-1 | 8,000 | 0 / 40 | 6 / 21 |
| transcript-1 | 12,000 | 0 / 61 | 6 / 41 |
| transcript-1 | 16,000 | 0 / 84 | 10 / 51 |
| transcript-1 | 24,000 | 10 / 75 | 10 / 75 |
| repository-1 | 8,000 | 0 / 40 | 6 / 12 |
| repository-1 | 12,000 | 0 / 62 | 6 / 24 |
| repository-1 | 16,000 | 3 / 69 | 10 / 27 |
| repository-1 | 24,000 | 10 / 62 | 10 / 62 |
| repository-2 | 8,000 | 0 / 41 | 6 / 11 |
| repository-2 | 12,000 | 0 / 62 | 6 / 30 |
| repository-2 | 16,000 | 0 / 86 | 10 / 31 |
| repository-2 | 24,000 | 10 / 76 | 10 / 75 |

## Decision and limits

Keep existing defaults: lookup 4k, explore 8k, legacy breadth 12k, deep 16k.
The allocation fix restored the requested full passages at 8k and 16k on these
samples. Raising to 24k mostly bought extra previews and surrounding context,
so keep that as an explicit override rather than a universal increase.

This is a small allocation regression experiment, not a relevance or answer
quality benchmark. Fewer returned sources may miss useful secondary evidence;
expand previews and reformulate queries when necessary. The 70% head share is
an engineering starting point, not an optimized parameter. No change to recency
weighting is justified by this experiment.

The focused tests in `tests/test_search_policy.py` reproduce preview crowding,
strongest-passage borrowing, oversized leading hits, multilingual token caps,
and preservation of fitting responses with generic fixtures. Numeric results
and per-call serialization timing are in [the JSON companion](response-budget-results.json).
Timings cover local allocation only, not retrieval or model response latency.
