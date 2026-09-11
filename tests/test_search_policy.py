"""Agent search contracts: explicit scope, bounded evidence, honest metadata."""
import json

import pytest
import tiktoken

from vecgrep.mcp.search_policy import resolve_search, bounded_payload


def test_presets_and_explicit_overrides_preserve_scope():
    args = resolve_search({'profile': 'explore', 'corpus': 'notes',
                           'filters': ['channel:example'], 'full_k': 3,
                           'rerank': False})
    assert args['full_k'] == 3
    assert args['budget'] is True
    assert args['rerank'] is False
    assert args['corpus'] == 'notes'
    assert args['filters'] == ['channel:example']
    assert resolve_search({'profile': 'lookup'})['budget'] is False
    assert resolve_search({'profile': 'deep'})['response_token_ceiling'] > args['response_token_ceiling']


def test_legacy_parameters_unchanged():
    assert resolve_search({'query': 'hello', 'top_k': 10}) == {'query': 'hello', 'top_k': 10}


@pytest.mark.parametrize('args', [{'profile': 'typo'}, {'full_k': 0},
                                {'response_token_ceiling': 10}])
def test_invalid_policy_fails_before_search(args):
    with pytest.raises(ValueError):
        resolve_search(args)


def test_whole_response_including_metadata_fits_and_retains_expansion_pointer():
    result = {'chunk_id': 'example-1', 'source_id': 'example.txt', 'corpus': 'notes',
              'chunk': '示例内容🙂' * 5000, 'context_before': 'before ' * 5000,
              'context_after': 'after ' * 5000, 'matched_by': ['rerank']}
    payload = {'full': [result], 'stubs': [], 'warnings': []}
    text = bounded_payload(payload, 1000, {'ignored_parameters': ['top_k']})
    assert len(tiktoken.get_encoding('cl100k_base').encode(text)) <= 1000
    data = json.loads(text)
    assert data['search_info']['truncated'] is True
    assert data['stubs'][0]['chunk_id'] == 'example-1'
    assert data['search_info']['tokenizer'] == 'cl100k_base'
    assert data['search_info']['ignored_parameters'] == ['top_k']
    assert payload['full'][0]['context_before']  # no caller mutation


def test_small_response_preserves_evidence_and_failure_warning():
    payload = {'full': [], 'stubs': [{'chunk_id': 'x', 'snippet': 'small'}],
               'warnings': [{'message': 'one corpus failed'}]}
    data = json.loads(bounded_payload(payload, 1000, {}))
    assert data['warnings'] == payload['warnings']
    assert data['search_info']['truncated'] is False
    assert data['search_info']['returned_results'] == 1


def test_mcp_budget_reports_ignored_knob_and_scope(monkeypatch):
    from vecgrep.mcp import server
    calls = []
    class Service:
        def search_budgeted_with_diagnostics(self, query, **kwargs):
            calls.append((query, kwargs))
            return [], [], []
    monkeypatch.setattr(server, '_svc', lambda: Service())
    data = json.loads(server._run_search({'query': 'example', 'profile': 'explore',
                                         'corpus': 'notes', 'top_k': 10}))
    assert calls[0][1]['corpus_name'] == 'notes'
    assert calls[0][1]['full_k'] == 6
    assert 'top_k' not in calls[0][1]
    assert data['search_info']['ignored_parameters'] == ['top_k']
    assert data['search_info']['reranked_results'] == 0


def test_stdio_schema_exposes_policy():
    import asyncio
    from mcp.types import ListToolsRequest
    from vecgrep.mcp.server import build_mcp_server
    server = build_mcp_server()
    result = asyncio.run(server.request_handlers[ListToolsRequest](ListToolsRequest(method='tools/list')))
    search = next(t for t in result.root.tools if t.name == 'search')
    assert search.inputSchema['properties']['profile']['enum'] == ['lookup', 'explore', 'deep']
    assert 'ignored' in search.inputSchema['properties']['top_k']['description']


def test_special_token_text_is_evidence_not_a_tokenizer_error():
    data = json.loads(bounded_payload({'hits': [{'chunk': '<|endoftext|>',
                                                'chunk_id': 'literal'}],
                                       'warnings': []}, 512, {}))
    assert data['hits'][0]['chunk'] == '<|endoftext|>'


def test_diagnostics_are_not_silently_dropped_to_fit():
    with pytest.raises(ValueError, match='diagnostics'):
        bounded_payload({'hits': [], 'warnings': ['failure ' * 2000]}, 512, {})


def _evidence(full_count=6, stub_count=90, words=250):
    return {
        'full': [{'chunk_id': f'full-{i}', 'corpus': 'notes',
                  'source_id': f'example-{i}.txt',
                  'chunk': ('important evidence ' * words),
                  'context_before': '', 'context_after': ''}
                 for i in range(full_count)],
        'stubs': [{'chunk_id': f'stub-{i}', 'snippet': 'preview ' * 40}
                  for i in range(stub_count)],
        'warnings': [],
    }


def test_preview_tail_cannot_crowd_out_top_passages():
    original = _evidence()
    text = bounded_payload(original, 4000, {})
    data = json.loads(text)
    assert [r['chunk_id'] for r in data['full']][:3] == ['full-0', 'full-1', 'full-2']
    assert data['stubs']
    assert data['full'][0]['chunk'] == original['full'][0]['chunk']
    assert len(tiktoken.get_encoding('cl100k_base').encode(text)) <= 4000


def test_top_passage_can_borrow_preview_share_when_it_fits_total():
    original = _evidence(full_count=1, words=1250)
    data = json.loads(bounded_payload(original, 3000, {}))
    assert data['full'][0]['chunk_id'] == 'full-0'
    assert data['full'][0]['chunk'] == original['full'][0]['chunk']


def test_roomy_response_is_not_trimmed_to_enforce_artificial_split():
    original = _evidence(full_count=1, stub_count=0, words=1250)
    data = json.loads(bounded_payload(original, 3000, {}))
    assert data['full'] == original['full']
    assert not data['search_info']['truncated']


def test_larger_budget_preserves_at_least_as_many_full_passages():
    original = _evidence()
    counts = [len(json.loads(bounded_payload(original, size, {}))['full'])
              for size in (2000, 4000, 8000)]
    assert counts == sorted(counts)
    assert counts[0] > 0


def test_oversized_top_hit_does_not_displace_next_fitting_passage():
    original = _evidence(full_count=2, words=200)
    original['full'][0]['chunk'] = 'oversized ' * 5000
    data = json.loads(bounded_payload(original, 2000, {}))
    assert [r['chunk_id'] for r in data['full']] == ['full-1']
    assert data['stubs'][0]['chunk_id'] == 'full-0'
