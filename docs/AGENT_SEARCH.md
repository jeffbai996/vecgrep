# Agent search

MCP search supports optional `profile: lookup|explore|deep`. Explicit parameters
win over preset defaults. Corpus selection and hard filters are never broadened.
These are initial policies, not benchmarked quality optima; ranking is unchanged.

| Profile | Initial evidence | Whole response cap |
| --- | --- | --- |
| lookup | 5 hits | 4,000 tokens |
| explore | 6 full hits plus previews | 8,000 tokens |
| deep | 10 full hits plus previews | 16,000 tokens |

All presets request reranking. `deep` increases returned evidence, not the
reranker candidate cap or search exhaustiveness. Search may still fall back;
inspect `matched_by`, warnings, and `search_info.reranked_results` (the number
reranked before response trimming). A request for reranking is not proof it ran.

Without a profile, existing automatic budget/rerank selection is preserved.
`top_k` applies outside budget mode. Budget mode retrieves up to 100 results and
reports an explicitly supplied `top_k` in `search_info.ignored_parameters`.
`full_k` is an upper bound on full results; trimming can demote hits to stubs.
`token_ceiling` retains its legacy meaning: approximate preview-tail allowance.

`response_token_ceiling` caps the complete serialized JSON text, including
metadata, using the existing cl100k_base tokenizer. It defaults to 12,000 for
budget mode; non-budget legacy calls retain their existing output contract unless
this parameter or a profile is provided. Minimum 512. This is not a universal
model token count and excludes MCP transport wrapping. Surrounding context is
trimmed first, then full chunks become expandable stubs, then lowest-ranked
stubs are removed. Check `search_info.truncated`; an empty truncated result is
not evidence that nothing matched. Diagnostics that cannot fit produce an error.

## Search playbook

- Choose lookup for a narrow fact, explore for discovery, deep for a larger
  evidence review. Use only parameters advertised by the connected tool.
- Preserve requested channel, source, speaker, and time scope. For broader
  questions choose relevant corpora explicitly; saved facts and original
  conversations serve different purposes.
- Read the preview tail and expand promising IDs with `get_chunk`. Use timeline
  or source context for event order, speaker attribution, and surrounding claims.
- Reformulate a weak query around a different discriminating term or subquestion;
  don't repeatedly raise counts on the same unproductive query.
- Check warnings, truncation, and actual reranking. Search failures, omitted
  evidence, and unreranked scores must not become confident negative findings.
- Compare independent supporting and conflicting evidence before answering.

Tune future defaults using the existing evaluation harness and permitted gold
sets: source-level recall, negative queries, response tokens, latency, and
follow-up expansion cost. Do not claim that larger outputs improve answers
without measuring it.
