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

    Context is removed before chunks; oversized chunks become expandable stubs.
    Lowest-ranked stubs are the final sacrifice. Warnings are never hidden.
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
    while True:
        metadata['returned_results'] = len(full) + len(stubs)
        text = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        if len(encoding.encode(text, disallowed_special=())) <= ceiling:
            return text
        metadata['truncated'] = True
        contextual = [r for r in full if r.get('context_before') or r.get('context_after')]
        if contextual:
            result = contextual[-1]
            result['context_before'] = result['context_after'] = ''
        elif full:
            result = full.pop()
            stub = {k: result[k] for k in (
                'chunk_id', 'corpus', 'source_id', 'doc_timestamp', 'matched_by',
                'relevance_pct', 'relevance_label') if k in result}
            stub['snippet'] = ' '.join(result.get('chunk', '').split())[:160]
            stubs.insert(0, stub)
        elif stubs:
            stubs.pop()
        else:
            # An unusually large diagnostic must fail visibly, not masquerade
            # as a successful search with its failure warnings stripped.
            raise ValueError('response_token_ceiling cannot fit search diagnostics; increase it')
