"""MCP search intent defaults and tokenizer-specific response accounting.

Presets are starting policies, not claims of measured retrieval optimality.
Explicit caller scope and overrides always survive resolution.
"""
from __future__ import annotations

from copy import deepcopy
import json

import tiktoken

PRESETS = {
    'lookup': dict(budget=False, top_k=5, rerank=True, response_token_ceiling=4000),
    'explore': dict(budget=True, full_k=6, token_ceiling=4000,
                    rerank=True, response_token_ceiling=8000),
    'deep': dict(budget=True, full_k=10, token_ceiling=6000,
                 rerank=True, response_token_ceiling=16000),
}
DEFAULT_RESPONSE_TOKEN_CEILING = 12000
FULL_BUDGET_FRACTION = 0.7


def resolve_search(args: dict) -> dict:
    """Fill only omitted knobs; never infer or broaden scope."""
    profile = args.get('profile')
    if profile is not None and profile not in PRESETS:
        raise ValueError('profile must be lookup, explore, or deep')
    resolved = {**PRESETS.get(profile, {}),
                **{k: v for k, v in args.items() if v is not None}}
    for key in ('top_k', 'full_k', 'token_ceiling', 'response_token_ceiling'):
        value = resolved.get(key)
        minimum = 512 if key == 'response_token_ceiling' else 1
        if value is not None and (type(value) is not int or value < minimum):
            raise ValueError(f'{key} must be an integer >= {minimum}')
    if resolved.get('full_k', 1) > 100:
        raise ValueError('full_k cannot exceed the 100-result breadth cap')
    return resolved


def bounded_payload(payload: dict, ceiling: int, info: dict) -> str:
    """Bound the complete JSON text in cl100k_base tokens, not transport wrappers.

    When trimming is necessary, protect a full-passage head before allocating
    preview space. The strongest passage may borrow the preview share if it
    fits the total. Warnings and expansion pointers survive normal trimming.
    """
    data = deepcopy(payload)
    full_key = 'full' if 'full' in data else 'hits'
    full = data.setdefault(full_key, [])
    stubs = data.setdefault('stubs', [])
    count = len(full) + len(stubs)
    metadata = data['search_info'] = {
        **info, 'tokenizer': 'cl100k_base', 'response_token_ceiling': ceiling,
        'available_results': count, 'returned_results': count, 'truncated': False,
    }
    encoding = tiktoken.get_encoding('cl100k_base')

    def render(include_stubs: bool = True) -> tuple[str, int]:
        metadata['returned_results'] = len(full) + len(stubs)
        view = data if include_stubs else {**data, 'stubs': []}
        text = json.dumps(view, ensure_ascii=False, separators=(',', ':'))
        return text, len(encoding.encode(text, disallowed_special=()))

    text, tokens = render()
    if tokens <= ceiling:
        return text
    metadata['truncated'] = True
    original_tail = list(stubs)
    rank = {id(result): i for i, result in enumerate(full)}
    demoted: dict[int, dict] = {}

    def demote(result: dict) -> None:
        stub = {k: result[k] for k in (
            'chunk_id', 'corpus', 'source_id', 'doc_timestamp', 'matched_by',
            'relevance_pct', 'relevance_label') if k in result}
        stub['snippet'] = ' '.join(result.get('chunk', '').split())[:160]
        demoted[rank[id(result)]] = stub
        stubs[:] = [demoted[i] for i in sorted(demoted)] + original_tail

    # An unfit first hit must not cause every smaller later passage to be
    # demoted as well. Keep its pointer, and consider the next fitting hit.
    for result in list(full):
        bare = {**result, 'context_before': '', 'context_after': ''}
        one = {**data, full_key: [bare], 'stubs': []}
        text = json.dumps(one, ensure_ascii=False, separators=(',', ':'))
        if len(encoding.encode(text, disallowed_special=())) > ceiling:
            full.remove(result)
            demote(result)

    # A preview-heavy response must not consume the passage allowance. When
    # there was no preview tail, use the whole allowance for the full head.
    head_ceiling = int(ceiling * FULL_BUDGET_FRACTION) if stubs else ceiling
    while full and render(include_stubs=False)[1] > head_ceiling:
        contextual = [r for r in full if r.get('context_before') or r.get('context_after')]
        if contextual:
            result = contextual[-1]
            result['context_before'] = result['context_after'] = ''
            continue
        if len(full) == 1 and render(include_stubs=False)[1] <= ceiling:
            break
        demote(full.pop())
    while True:
        text, tokens = render()
        if tokens <= ceiling:
            return text
        if stubs:
            stubs.pop()
        else:
            raise ValueError('response_token_ceiling cannot fit search diagnostics; increase it')
