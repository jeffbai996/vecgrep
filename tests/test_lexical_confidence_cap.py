from vecgrep.backend.service import _bm25_to_result


def test_best_lexical_hit_cannot_claim_strong_semantic_confidence():
    for score in (0.001, 1.0, 100.0):
        hit = _bm25_to_result('notes', 'chunk', score, {'corpus':'notes','source_id':'example.md','text':'example'}, ['bm25'], max_score=score)
        assert hit.similarity_pct < 40
        assert hit.relevance_label == 'lexical-only'


def test_service_lexical_hits_are_bounded_and_negative_query_stays_empty(svc, make_doc):
    svc.index(str(make_doc('one.md', 'Cobalt river crossing. Unique orange bridge.')), 'notes')
    svc.index(str(make_doc('two.md', 'Violet mountain hiking.')), 'notes')
    hits = svc.search('cobalt', 'notes', mode='bm25', rerank=False)
    assert hits and all(hit.similarity_pct < 40 for hit in hits)
    assert not svc.search('zzzzmissingterm', 'notes', mode='bm25', rerank=False)
