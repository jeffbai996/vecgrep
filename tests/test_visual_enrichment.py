from typing import ClassVar

import pytest

from vecgrep.backend.ingestion.visual_queue import VisualQueue


def test_jobs_are_deduplicated_and_results_survive_restart(tmp_path):
    source = tmp_path / 'chart.png'
    source.write_bytes(b'first')
    q = VisualQueue(tmp_path / 'queue.db')
    key = q.enqueue(source)
    assert q.enqueue(source) == key
    job = q.claim(now=100)
    assert job is not None
    q.complete(job, 'Revenue rises', now=101)
    assert VisualQueue(tmp_path / 'queue.db').result(source) == 'Revenue rises'
    assert q.claim(now=102) is None


def test_changed_source_never_receives_old_enrichment(tmp_path):
    source = tmp_path / 'chart.png'
    source.write_bytes(b'first')
    q = VisualQueue(tmp_path / 'queue.db')
    q.enqueue(source)
    job = q.claim(now=100)
    assert job is not None
    source.write_bytes(b'second')
    assert not q.complete(job, 'Old chart', now=101)
    assert q.result(source) is None
    assert q.enqueue(source) != job['key']


def test_claim_is_exclusive_and_expired_lease_is_recoverable(tmp_path):
    source = tmp_path / 'chart.png'
    source.write_bytes(b'first')
    q = VisualQueue(tmp_path / 'queue.db')
    q.enqueue(source)
    first = q.claim(now=100, lease_s=10)
    assert first is not None
    assert q.claim(now=105) is None
    second = q.claim(now=111, lease_s=10)
    assert second is not None
    assert second['key'] == first['key']
    assert not q.complete(first, 'Late answer', now=112)
    assert q.complete(second, 'Current answer', now=112)


def test_retry_keeps_ocr_text_available_and_records_error(tmp_path):
    source = tmp_path / 'scan.png'
    source.write_bytes(b'scan')
    q = VisualQueue(tmp_path / 'queue.db')
    q.enqueue(source)
    job = q.claim(now=100)
    assert job is not None
    q.retry(job, 'worker unavailable', text='Invoice 123', now=101)
    assert q.result(source) == 'Invoice 123'
    assert q.claim(now=102) is None
    assert q.status()[0]['state'] == 'pending'
    assert q.status()[0]['error'] == 'worker unavailable'


def test_deleted_source_is_cancelled_without_reindexing(tmp_path):
    source = tmp_path / 'scan.png'
    source.write_bytes(b'scan')
    q = VisualQueue(tmp_path / 'queue.db')
    q.enqueue(source)
    job = q.claim(now=100)
    assert job is not None
    source.unlink()
    assert not q.complete(job, 'Invoice', now=101)
    assert q.status()[0]['state'] == 'cancelled'


def test_delivery_targets_are_durable_and_cancelled_on_source_removal(tmp_path):
    source = tmp_path / 'scan.png'
    source.write_bytes(b'scan')
    q = VisualQueue(tmp_path / 'queue.db')
    q.enqueue(source)
    q.target(source, 'documents', 'sentence_window')
    job = q.claim(now=100)
    assert job is not None
    q.complete(job, 'Invoice', now=101)
    deliveries = q.deliveries()
    assert len(deliveries) == 1
    q.delivered(deliveries[0])
    assert not q.deliveries()
    q.cancel_target(str(source.resolve()), 'documents')
    assert not q.deliveries()


def test_image_adapter_is_disabled_by_default(tmp_path):
    from vecgrep.backend.ingestion.adapters.image import ImageAdapter
    source = tmp_path / 'scan.png'
    source.write_bytes(b'scan')
    assert not ImageAdapter.matches(str(source))


def test_image_adapter_returns_pending_then_cached_text(tmp_path, monkeypatch):
    from vecgrep.backend.ingestion.adapters.image import ImageAdapter
    monkeypatch.setenv('VECGREP_VISUAL_ENABLED', '1')
    source = tmp_path / 'scan.png'
    source.write_bytes(b'scan')
    doc = next(ImageAdapter().load(str(source)))
    assert doc.text == ''
    assert doc.metadata['visual_state'] == 'pending'
    q = VisualQueue.default()
    job = q.claim()
    assert job is not None
    q.complete(job, '[page 1]\nInvoice 123')
    doc = next(ImageAdapter().load(str(source)))
    assert 'Invoice 123' in doc.text
    assert doc.source_id == str(source.resolve())


def test_pdf_keeps_native_text_when_vision_is_pending(tmp_path, monkeypatch):
    from vecgrep.backend.ingestion.adapters import pdf
    monkeypatch.setenv('VECGREP_VISUAL_ENABLED', '1')
    class Page:
        def extract_text(self): return 'Native report text'
    class Reader:
        pages: ClassVar[list[Page]] = [Page()]
    monkeypatch.setattr(pdf, 'PdfReader', lambda _: Reader())
    source = tmp_path / 'report.pdf'
    source.write_bytes(b'pdf')
    doc = next(pdf.PDFAdapter().load(str(source)))
    assert 'Native report text' in doc.text
    assert doc.metadata['visual_state'] == 'pending'


def test_model_output_must_be_complete_and_nonempty():
    from vecgrep.backend.ingestion.visual_processor import parse_vision
    assert parse_vision({'done': True, 'done_reason': 'stop', 'message': {'content': 'Chart of revenue'}}) == 'Chart of revenue'
    for response in ({'done': False}, {'done': True, 'done_reason': 'length', 'message': {'content': 'partial'}}, {'done': True, 'message': {'content': ''}}):
        with pytest.raises(ValueError): parse_vision(response)


def test_visual_worker_keeps_ocr_on_vision_failure(tmp_path):
    from vecgrep.backend.ingestion.visual_worker import process_one
    source = tmp_path / 'scan.png'
    source.write_bytes(b'scan')
    q = VisualQueue(tmp_path / 'queue.db')
    q.enqueue(source)
    def processor(path):
        return 'Invoice 123', False, 'vision unavailable'
    assert process_one(q, processor=processor)
    assert q.result(source) == 'Invoice 123'
    assert q.status()[0]['state'] == 'pending'


def test_empty_pending_image_does_not_embed_and_completion_uses_existing_backend(svc, tmp_path, monkeypatch):
    monkeypatch.setenv('VECGREP_VISUAL_ENABLED', '1')
    source = tmp_path / 'scan.png'
    source.write_bytes(b'scan')
    assert svc.index(str(source), 'documents')[1] == 0
    q = VisualQueue.default()
    job = q.claim()
    assert job is not None
    q.complete(job, 'Invoice 123')
    assert svc.index(str(source), 'documents')[1] > 0
    assert svc.index(str(source), 'documents')[2] == 1
    assert svc.registry.get('documents').embed_model == 'stub-1'


def test_delayed_delivery_cannot_resurrect_deleted_source(svc, tmp_path, monkeypatch):
    monkeypatch.setenv('VECGREP_VISUAL_ENABLED', '1')
    source = tmp_path / 'scan.png'
    source.write_bytes(b'scan')
    seed = tmp_path / 'seed.txt'
    seed.write_text('Existing searchable document')
    svc.index(str(seed), 'documents')
    svc.index(str(source), 'documents')
    q = VisualQueue.default()
    job = q.claim()
    assert job is not None
    q.complete(job, 'Invoice 123')
    item = q.deliveries()[0]
    svc.delete_source('documents', str(source.resolve()))
    from vecgrep.backend.store.corpora import CorpusError
    with pytest.raises(CorpusError):
        svc.index(str(source), 'documents', visual_key=item['key'])
    assert not q.deliveries()


def test_worker_delivery_retries_index_failures(tmp_path):
    import httpx

    from vecgrep.backend.ingestion.visual_worker import deliver
    source = tmp_path / 'scan.png'
    source.write_bytes(b'scan')
    q = VisualQueue(tmp_path / 'queue.db')
    q.enqueue(source)
    q.target(source, 'documents', 'sentence_window')
    claimed = q.claim()
    assert claimed is not None
    q.complete(claimed, 'Invoice 123')
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503))) as client:
        assert deliver(q, url='http://example.test', client=client) == 0
    assert len(q.deliveries()) == 1
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={'failed': []}))) as client:
        assert deliver(q, url='http://example.test', client=client) == 1
    assert not q.deliveries()


def test_processor_renders_image_and_uses_ocr_when_laptop_is_offline(tmp_path, monkeypatch):
    import httpx
    from PIL import Image

    from vecgrep.backend.ingestion import visual_processor as vp
    source = tmp_path / 'scan.png'
    Image.new('RGB', (200, 100), 'white').save(source)
    monkeypatch.setattr(vp, 'ocr', lambda *args, **kwargs: 'Invoice 123')
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503))) as client:
        processor = vp.VisualProcessor(vision_url='http://example.test', model='test', scratch=tmp_path, client=client)
        text, complete, error = processor(source)
    assert 'Invoice 123' in text and '[page 1]' in text
    assert not complete and 'Vision page 1' in error


def test_pdf_render_has_page_anchors_and_rejects_excess_pages(tmp_path):
    from pypdf import PdfWriter

    from vecgrep.backend.ingestion.visual_processor import page_images
    source = tmp_path / 'report.pdf'
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=100)
    writer.add_blank_page(width=200, height=100)
    with source.open('wb') as handle: writer.write(handle)
    assert [number for number, _ in page_images(source)] == [1, 2]
    with pytest.raises(ValueError, match='page enrichment limit'):
        list(page_images(source, max_pages=1))


def test_pdf_ocr_preserves_each_page_when_vision_is_offline(tmp_path, monkeypatch):
    import httpx
    from pypdf import PdfWriter

    from vecgrep.backend.ingestion import visual_processor as vp

    source = tmp_path / 'report.pdf'
    writer = PdfWriter()
    for _ in range(2):
        writer.add_blank_page(width=200, height=100)
    with source.open('wb') as handle:
        writer.write(handle)

    pages = []
    requests = []

    def ocr(image, **kwargs):
        assert image.startswith(b'\x89PNG\r\n\x1a\n')
        pages.append(image)
        return f'Invoice page {len(pages)}'

    def offline(request):
        requests.append(request)
        raise httpx.ConnectError('worker offline', request=request)

    monkeypatch.setattr(vp, 'ocr', ocr)
    with httpx.Client(transport=httpx.MockTransport(offline)) as client:
        processor = vp.VisualProcessor(
            vision_url='http://example.test', model='test', scratch=tmp_path, client=client,
        )
        text, complete, error = processor(source)

    assert text == (
        '[page 1]\n[OCR text]\nInvoice page 1\n\n'
        '[page 2]\n[OCR text]\nInvoice page 2'
    )
    assert not complete
    assert error == 'Vision page 1: ConnectError'
    assert len(pages) == 2
    assert len(requests) == 1


def test_reverting_file_content_requeues_cancelled_work(tmp_path):
    source = tmp_path / 'chart.png'
    source.write_bytes(b'first')
    q = VisualQueue(tmp_path / 'queue.db')
    original = q.enqueue(source)
    source.write_bytes(b'second')
    q.enqueue(source)
    source.write_bytes(b'first')
    assert q.enqueue(source) == original
    claimed = q.claim()
    assert claimed is not None
    assert claimed['key'] == original


def test_visual_size_limit_does_not_block_native_pdf_text(tmp_path, monkeypatch):
    from vecgrep.backend.ingestion import visual_queue
    from vecgrep.backend.ingestion.adapters import pdf
    monkeypatch.setenv('VECGREP_VISUAL_ENABLED', '1')
    class Page:
        def extract_text(self): return 'Native report text'
    class Reader:
        pages: ClassVar[list[Page]] = [Page()]
    monkeypatch.setattr(pdf, 'PdfReader', lambda _: Reader())
    monkeypatch.setattr(visual_queue, 'MAX_SOURCE_BYTES', 2)
    source = tmp_path / 'report.pdf'
    source.write_bytes(b'larger than the limit')
    doc = next(pdf.PDFAdapter().load(str(source)))
    assert 'Native report text' in doc.text
    assert doc.metadata['visual_state'] == 'unavailable'
    assert 'limit' in doc.metadata['visual_error']
