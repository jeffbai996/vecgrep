import test from 'node:test';
import assert from 'node:assert/strict';
import { documentLink } from './documentLink.ts';

test('a document link preserves its corpus and exact encoded source', () => {
  const source = '/data/notes/a report + 中文.md';
  const params = new URLSearchParams({ corpus: 'notes', source });
  assert.deepEqual(documentLink('?' + params), { corpus: 'notes', sourceId: source, nonce: 1 });
});

test('an incomplete link does not select a document', () => {
  for (const search of ['', '?source=notes-1', '?corpus=notes', '?corpus=&source=notes-1']) {
    assert.equal(documentLink(search), null);
  }
});
