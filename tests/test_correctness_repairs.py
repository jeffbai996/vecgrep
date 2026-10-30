"""Deterministic retrieval and incremental-index correctness contracts."""
import pytest

from vecgrep.backend.ingestion.chunkers.turn_window import TurnWindowChunker
from vecgrep.backend.store.bm25_sqlite import BM25SqliteStore
from vecgrep.backend.store.bm25_store import BM25Store


@pytest.mark.parametrize('body', ['No', 'Alice', '是', '👍', 'x' * 10, 'x' * 11, 'x' * 12])
def test_short_turn_is_preserved_with_its_speaker(body):
    text = '**user** · 10:00\nA long question providing the conversation context.\n\n**user** · 10:01\n' + body
    chunks = TurnWindowChunker(turns=1, stride=1).chunk(text)
    assert any(body in c.text and '10:01' in c.text for c in chunks)
    assert all(text[c.start:c.end] == c.text for c in chunks)

@pytest.mark.parametrize('text', ['', '   ', '**user** · 10:00\n> \n\n**user** · 10:01\n> ...'])
def test_empty_or_punctuation_only_turns_do_not_create_chunks(text):
    assert TurnWindowChunker().chunk(text) == []

@pytest.mark.parametrize('store_type', [BM25Store, BM25SqliteStore])
def test_upsert_replaces_identity_and_matches_clean_rebuild(tmp_path, store_type):
    store = store_type(tmp_path / 'incremental')
    fresh = store_type(tmp_path / 'fresh')
    def payload(cid, text):
        return {'source_id': cid + '.md', 'text': text, 'metadata': {'status': 'active'}}
    store.upsert('c', ['a', 'b', 'c'], ['obsoleteword', 'lasting anchor', 'removeonly'],
                 [payload('a', 'obsoleteword'), payload('b', 'lasting anchor'), payload('c', 'removeonly')])
    for _ in range(3):
        store.upsert('c', ['a'], ['replacementword'], [payload('a', 'replacementword')])
    store.delete_ids('c', ['c'])
    fresh.upsert('c', ['a', 'b'], ['replacementword', 'lasting anchor'],
                 [payload('a', 'replacementword'), payload('b', 'lasting anchor')])
    assert store.count('c') == 2
    for query in ['replacementword', 'lasting', 'obsoleteword', 'removeonly', 'unrelated']:
        assert store.search('c', query, 10) == fresh.search('c', query, 10)
    store.delete_by_source('c', 'a.md')
    assert store.search('c', 'replacementword', 10) == []

@pytest.mark.parametrize('backend', ['pickle', 'sqlite'])
@pytest.mark.parametrize('mode', ['vector', 'bm25', 'hybrid'])
def test_superseded_candidates_do_not_consume_result_budget(tmp_path, backend, mode):
    from tests.test_area3_retrieval import _DeterministicEmbedder
    from vecgrep.backend.config import Settings
    from vecgrep.backend.service import VecgrepService
    svc = VecgrepService(Settings(home=tmp_path/'state', bm25_backend=backend, qdrant_url=None))
    svc._backend_cache['auto'] = _DeterministicEmbedder()
    docs = tmp_path / 'docs'; docs.mkdir()
    for i in range(65):
        status = 'superseded' if i < 60 else 'active'
        (docs/f'{i:03}.md').write_text(f'---\nstatus: {status}\n---\nUnique {i} target_keyword evidence.\n')
    svc.index(str(docs), 'c')
    hits = svc.search('target_keyword', 'c', top_k=3, mode=mode)
    assert len(hits) == 3
    assert all(h.metadata['status'] == 'active' for h in hits)
    assert len(svc.search('target_keyword', 'c', top_k=10, mode=mode)) == 5
    full, stubs, warnings = svc.search_budgeted_with_diagnostics(
        'target_keyword', 'c', full_k=2, max_total=5, mode=mode)
    assert len(full) == 2 and len(stubs) == 3 and not warnings
    assert len({h.chunk_id for h in [*full, *stubs]}) == 5
    for stub in stubs:
        assert svc.get_chunk_window('c', stub.chunk_id, window=400) is not None
    assert len(svc.search('target_keyword', 'c', mode=mode, filters=['source:*064.md'])) == 1
    assert len(svc.search('target_keyword', 'c', mode=mode, filters=['meta.status=superseded'])) == 5
    svc.store.client.close()

@pytest.mark.parametrize('failure', [TimeoutError('private endpoint token'), RuntimeError('private endpoint token')])
def test_storage_failure_is_typed_and_recovery_returns_results(svc, make_doc, monkeypatch, failure):
    from vecgrep.backend.store.qdrant_store import SearchBackendError
    doc = make_doc('a.md', 'Distinct orchid evidence with enough words to index.')
    svc.index(str(doc), 'c')
    original = svc.store.client.query_points
    def broken(**kwargs): raise failure
    monkeypatch.setattr(svc.store.client, 'query_points', broken)
    with pytest.raises(SearchBackendError, match='unavailable') as exc:
        svc.search('orchid', 'c', mode='vector')
    assert 'private' not in str(exc.value)
    degraded = svc.search_with_diagnostics('orchid', 'c', mode='hybrid')
    assert degraded.results and degraded.warnings[0].code == 'vector_unavailable'
    monkeypatch.setattr(svc.store.client, 'query_points', original)
    recovered = svc.search_with_diagnostics('orchid', 'c', mode='hybrid')
    assert recovered.results and not recovered.warnings
    empty = svc.search_with_diagnostics('nonexistentword', 'c', mode='bm25')
    assert empty.results == [] and empty.warnings == []


def test_embedder_initialization_failure_reports_keyword_fallback(svc, make_doc, monkeypatch):
    from vecgrep.backend.embed.base import EmbedBackendError
    svc.index(str(make_doc('a.md', 'Distinct orchid evidence in this document.')), 'c')
    def unavailable(*args, **kwargs): raise EmbedBackendError('dependency unavailable')
    monkeypatch.setattr(svc, '_backend_for', unavailable)
    with pytest.raises(EmbedBackendError): svc.search('orchid', 'c', mode='vector')
    outcome = svc.search_with_diagnostics('orchid', 'c', mode='hybrid')
    assert outcome.results and outcome.warnings[0].code == 'vector_unavailable'


def test_cached_zero_ingestion_vector_is_not_a_query(svc, monkeypatch):
    from vecgrep.backend.embed.base import EmbedBackendError
    inner = svc._backend_cache['auto']._inner
    monkeypatch.setattr(inner, 'embed_one', lambda text: [0.0] * inner.dim)
    with pytest.raises(EmbedBackendError):
        svc._backend_cache['auto'].embed_one('query')


def test_http_dependency_error_does_not_expose_exception_detail(svc, make_doc, monkeypatch):
    from fastapi import HTTPException

    from vecgrep.backend.api import routes
    from vecgrep.backend.api.schemas import SearchRequest
    from vecgrep.backend.embed.base import EmbedBackendError
    svc.index(str(make_doc('a.md', 'Orchid evidence from synthetic source.')), 'c')
    monkeypatch.setattr(routes, '_service', lambda: svc)
    def unavailable(*args, **kwargs):
        raise EmbedBackendError('https://private.invalid/?token=sensitive')
    monkeypatch.setattr(svc, '_backend_for', unavailable)
    with pytest.raises(HTTPException) as caught:
        routes.search(
            SearchRequest(
                query="orchid",
                corpus="c",
                mode="vector",
                top_k=None,
                full_k=8,
                max_total=100,
                token_ceiling=4000,
            )
        )
    assert caught.value.status_code == 503
    assert 'sensitive' not in caught.value.detail


@pytest.mark.parametrize('backend', ['pickle', 'sqlite'])
def test_incremental_transcript_keeps_short_corrections_without_duplicates(tmp_path, backend):
    from tests.test_retrieval_fixture import FixtureEmbedder
    from vecgrep.backend.config import Settings
    from vecgrep.backend.service import VecgrepService
    svc = VecgrepService(Settings(home=tmp_path/'state', qdrant_url=None, bm25_backend=backend))
    svc._backend_cache['auto'] = FixtureEmbedder()
    doc = tmp_path/'chat.md'
    text = '**Alice** · 10:00\nThe orchid name needs confirmation before filing this note.\n\n**Bob** · 10:01\nNo.\n'
    doc.write_text(text)
    svc.index(str(doc), 'c', chunker_name='turn_window')
    doc.write_text(text + '\n**Alice** · 10:02\n雪山\n')
    svc.index(str(doc), 'c', chunker_name='turn_window')
    first = list(svc.bm25.iter_payloads('c'))
    for _ in range(2): svc.index(str(doc), 'c', chunker_name='turn_window')
    assert list(svc.bm25.iter_payloads('c')) == first
    assert any('No.' in p['text'] and '雪山' in p['text'] for p in first)
    svc.store.client.close()
