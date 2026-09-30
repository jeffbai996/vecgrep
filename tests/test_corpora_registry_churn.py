"""corpora.json must not be rewritten when nothing changed.

The registry is committed once per indexed source so each journal record has a
complete recovery boundary. That is correct, but _save() wrote the WHOLE file
(2.2 MB on a real install) plus two fsyncs every single time -- including when
the payload was byte-identical, which is what a no-op re-ingest produces. The
5-minute transcript ingests made that a steady drip onto the SSD for no
information gained.
"""
from __future__ import annotations

from pathlib import Path

from vecgrep.backend.store.corpora import Corpus, CorpusRegistry


def _corpus(name: str = "c", **kw) -> Corpus:
    return Corpus(name=name, embed_backend="ollama", embed_model="m", dim=4, **kw)


def test_identical_upsert_does_not_rewrite_the_file(tmp_path):
    path = tmp_path / "corpora.json"
    reg = CorpusRegistry(path, in_memory=False)
    reg.upsert(_corpus())

    # _save() writes a temp file and os.replace()s it in, so the inode -- not
    # the mtime, which the replacement carries over -- is what proves whether
    # 2.2 MB and two fsyncs actually hit the disk.
    before_ino = path.stat().st_ino
    before_bytes = path.read_bytes()

    reg.upsert(_corpus())

    assert path.stat().st_ino == before_ino
    assert path.read_bytes() == before_bytes


def test_a_real_change_still_writes(tmp_path):
    path = tmp_path / "corpora.json"
    reg = CorpusRegistry(path, in_memory=False)
    reg.upsert(_corpus())
    before = path.read_bytes()

    reg.upsert(_corpus(doc_count=7))

    assert path.read_bytes() != before
    assert reg.get("c").doc_count == 7


def test_delete_still_writes(tmp_path):
    path = tmp_path / "corpora.json"
    reg = CorpusRegistry(path, in_memory=False)
    reg.upsert(_corpus())
    reg.upsert(_corpus("d"))
    before = path.read_bytes()

    reg.delete("d")

    assert path.read_bytes() != before
    assert not reg.has("d")


def test_a_new_corpus_is_written_even_on_a_fresh_registry(tmp_path):
    path = tmp_path / "corpora.json"
    reg = CorpusRegistry(path, in_memory=False)
    reg.upsert(_corpus())
    assert path.exists()
    assert CorpusRegistry(path, in_memory=False).has("c")
