from __future__ import annotations

import hashlib
import multiprocessing
import threading
import time
from pathlib import Path

import pytest

from vecgrep.backend.mutation import CorpusLocks, LockTimeout, LockUpgradeError


def _col(name: str) -> str:
    from vecgrep.backend.service import _collection_for

    return _collection_for(name)


# ── test_mutation_locks ──

def _hold_writer(root: str, corpus: str, ready, release) -> None:
    locks = CorpusLocks(Path(root))
    with locks.write(corpus):
        ready.set()
        release.wait(10)


def _time_writer(root: str, corpus: str, ready, waiting, output) -> None:
    locks = CorpusLocks(Path(root))
    ready.wait(10)
    waiting.set()
    started = time.monotonic()
    with locks.write(corpus):
        output.put(time.monotonic() - started)


def test_same_corpus_writers_serialize_across_processes(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    ready, waiting, release, output = ctx.Event(), ctx.Event(), ctx.Event(), ctx.Queue()
    first = ctx.Process(target=_hold_writer, args=(str(tmp_path), "notes", ready, release))
    second = ctx.Process(target=_time_writer, args=(str(tmp_path), "notes", ready, waiting, output))
    first.start()
    second.start()
    assert ready.wait(10)
    assert waiting.wait(10)
    time.sleep(0.2)
    assert output.empty()
    release.set()
    assert output.get(timeout=5) >= 0.15
    first.join(5)
    second.join(5)
    assert first.exitcode == second.exitcode == 0


def test_different_corpora_do_not_block_each_other(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    ready, waiting, release, output = ctx.Event(), ctx.Event(), ctx.Event(), ctx.Queue()
    first = ctx.Process(target=_hold_writer, args=(str(tmp_path), "alpha", ready, release))
    second = ctx.Process(target=_time_writer, args=(str(tmp_path), "beta", ready, waiting, output))
    first.start()
    second.start()
    assert output.get(timeout=10) < 0.15
    release.set()
    first.join(5)
    second.join(5)
    assert first.exitcode == second.exitcode == 0


def test_nested_writer_may_read_but_reader_cannot_upgrade(tmp_path):
    locks = CorpusLocks(tmp_path)
    with locks.write("notes"), locks.read("notes"):
        pass
    with locks.read("notes"), pytest.raises(LockUpgradeError), locks.write("notes"):
        pass


def test_reentrant_read_does_not_block_behind_a_queued_writer(tmp_path):
    """A thread already holding the read lock must be able to re-enter it.

    The queued writer it would otherwise wait for is itself waiting for this
    thread's reader count to reach zero, so waiting here never ends -- and the
    corpus stays unsearchable for every other caller until the process dies.
    """
    locks = CorpusLocks(tmp_path)
    holding = threading.Event()
    writer_queued = threading.Event()
    reentered = threading.Event()
    released = threading.Event()

    def reader():
        with locks.read("notes"):
            holding.set()
            writer_queued.wait(2)
            time.sleep(0.2)
            with locks.read("notes"):
                reentered.set()
        released.set()

    def writer():
        holding.wait(2)
        writer_queued.set()
        with locks.write("notes"):
            pass

    r = threading.Thread(target=reader, daemon=True)
    w = threading.Thread(target=writer, daemon=True)
    r.start()
    w.start()
    assert reentered.wait(3), "reentrant read blocked behind a queued writer"
    assert released.wait(3)
    w.join(3)
    assert not w.is_alive()


def test_read_admission_gives_up_while_a_writer_holds_the_lock(tmp_path):
    locks = CorpusLocks(tmp_path)
    holding, release = threading.Event(), threading.Event()

    def writer():
        with locks.write("notes"):
            holding.set()
            release.wait(5)

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    assert holding.wait(2)
    started = time.monotonic()
    with pytest.raises(LockTimeout), locks.read("notes", timeout=0.3):
        pass
    assert time.monotonic() - started < 2
    release.set()
    t.join(5)


def test_write_admission_gives_up_while_a_reader_holds_the_lock(tmp_path):
    locks = CorpusLocks(tmp_path)
    holding, release = threading.Event(), threading.Event()

    def reader():
        with locks.read("notes"):
            holding.set()
            release.wait(5)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    assert holding.wait(2)
    with pytest.raises(LockTimeout), locks.write("notes", timeout=0.3):
        pass
    release.set()
    t.join(5)


def test_a_timed_out_reader_leaves_the_lock_usable(tmp_path):
    """A give-up must not leave half-held state behind."""
    locks = CorpusLocks(tmp_path)
    holding, release = threading.Event(), threading.Event()

    def writer():
        with locks.write("notes"):
            holding.set()
            release.wait(5)

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    assert holding.wait(2)
    with pytest.raises(LockTimeout), locks.read("notes", timeout=0.3):
        pass
    release.set()
    t.join(5)
    started = time.monotonic()
    with locks.read("notes", timeout=5):
        pass
    assert time.monotonic() - started < 2
    with locks.write("notes", timeout=5):
        pass


def test_cross_process_read_admission_is_bounded(tmp_path):
    """The file lock has to honour the deadline too, not just the thread lock."""
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    holder = ctx.Process(
        target=_hold_writer, args=(str(tmp_path), "notes", ready, release)
    )
    holder.start()
    assert ready.wait(5)
    started = time.monotonic()
    with pytest.raises(LockTimeout), CorpusLocks(tmp_path).read("notes", timeout=0.3):
        pass
    assert time.monotonic() - started < 2
    release.set()
    holder.join(5)
    assert holder.exitcode == 0


def test_no_timeout_still_waits_for_the_holder(tmp_path):
    """The default stays blocking: a caller that passes no timeout waits."""
    locks = CorpusLocks(tmp_path)
    holding, release = threading.Event(), threading.Event()
    acquired = threading.Event()

    def writer():
        with locks.write("notes"):
            holding.set()
            release.wait(5)

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    assert holding.wait(2)

    def reader():
        with locks.read("notes"):
            acquired.set()

    r = threading.Thread(target=reader, daemon=True)
    r.start()
    assert not acquired.wait(0.4)
    release.set()
    assert acquired.wait(5)
    t.join(5)
    r.join(5)

# ── test_mutation_recovery ──

def test_search_waits_for_same_corpus_commit(svc, make_doc, monkeypatch):
    source = make_doc("doc.md", "old marker sentence")
    svc.index(str(source), "notes")
    source.write_text("new marker sentence")

    entered = threading.Event()
    release = threading.Event()
    original = svc.store.upsert

    def paused(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(svc.store, "upsert", paused)
    writer = threading.Thread(
        target=lambda: svc.index(str(source), "notes", force=True), daemon=True
    )
    writer.start()
    assert entered.wait(5)

    finished = threading.Event()
    results = []

    def searcher():
        results.extend(svc.search("new marker", "notes"))
        finished.set()

    reader = threading.Thread(target=searcher, daemon=True)
    reader.start()
    time.sleep(0.15)
    assert not finished.is_set(), "search crossed a half-committed mutation"
    release.set()
    writer.join(5)
    reader.join(5)
    assert finished.is_set()
    assert results


def test_recovery_rolls_forward_after_qdrant_commit(svc, make_doc, monkeypatch):
    source = make_doc("doc.md", "old marker sentence")
    svc.index(str(source), "notes")
    source.write_text("new marker sentence")

    original = svc.bm25.upsert

    def fail_after_qdrant(*args, **kwargs):
        raise RuntimeError("fault after qdrant")

    monkeypatch.setattr(svc.bm25, "upsert", fail_after_qdrant)
    with pytest.raises(RuntimeError, match="fault after qdrant"):
        svc.index(str(source), "notes", force=True)
    record = svc.mutations.read("notes")
    assert record and record["phase"] == "qdrant_done"

    monkeypatch.setattr(svc.bm25, "upsert", original)
    assert svc.recover_pending_mutations() == ["notes"]
    assert svc.mutations.read("notes") is None
    assert svc.search("new marker", "notes")
    corpus = svc.registry.get("notes")
    assert corpus.source_hashes[str(source.resolve())] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert corpus.chunk_count == svc.store.count(_col("notes"))


def test_recovery_rolls_back_partial_qdrant_batch(svc, make_doc, monkeypatch):
    old_text = "Old one. Old two. Old three."
    source = make_doc("doc.md", old_text)
    svc.index(str(source), "notes")
    source.write_text("New one. New two. New three. New four.")

    original = svc.store.upsert

    def partial(collection, ids, vectors, payloads):
        original(collection, ids[:1], vectors[:1], payloads[:1])
        raise RuntimeError("fault in qdrant batch")

    monkeypatch.setattr(svc.store, "upsert", partial)
    with pytest.raises(RuntimeError, match="fault in qdrant batch"):
        svc.index(str(source), "notes", force=True)
    record = svc.mutations.read("notes")
    assert record and record["phase"] == "prepared"

    monkeypatch.setattr(svc.store, "upsert", original)
    assert svc.recover_pending_mutations() == ["notes"]
    payloads = list(svc.store.iter_payloads(_col("notes")))
    assert payloads
    # Only the carrier chunk holds the document, so assert on the carriers
    # rather than on every payload -- and check no chunk kept the new text.
    assert {p[1]["source_text"] for p in payloads if p[1].get("source_text")} == {
        old_text
    }
    assert all("New " not in (p[1].get("text") or "") for p in payloads)
    assert svc.search("Old two", "notes")


def test_recovery_completes_interrupted_delete(svc, make_doc, monkeypatch):
    source = make_doc("doc.md", "delete me marker")
    svc.index(str(source), "notes")

    original = svc.bm25.delete_by_source

    def fail_after_qdrant(*args, **kwargs):
        raise RuntimeError("fault during delete")

    monkeypatch.setattr(svc.bm25, "delete_by_source", fail_after_qdrant)
    with pytest.raises(RuntimeError, match="fault during delete"):
        svc.delete_source("notes", str(source.resolve()))
    monkeypatch.setattr(svc.bm25, "delete_by_source", original)

    assert svc.recover_pending_mutations() == ["notes"]
    assert svc.store.count(_col("notes")) == 0
    assert svc.registry.get("notes").chunk_count == 0
    assert not svc.bm25.search("notes", "delete", 10)


def test_journal_records_carry_metadata_not_the_whole_source_map(svc, make_doc, monkeypatch):
    """Recovery rebuilds sources/hashes from Qdrant, so the journal must not
    carry them: on a 5,791-source corpus that was a 4.4 MB fsync'd write, four
    times per source, 76% of an index run's bytes (2026-09-13)."""
    import json as _json
    for i in range(3):
        svc.index(str(make_doc(f"d{i}.md", f"marker {i}")), "notes")
    source = make_doc("d0.md", "changed marker")
    monkeypatch.setattr(svc.bm25, "upsert", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fault")))
    with pytest.raises(RuntimeError):
        svc.index(str(source), "notes", force=True)
    record = svc.mutations.read("notes")
    assert record and record["operation"] == "index_source"
    for key in ("corpus_before", "corpus_target"):
        assert record[key]["sources"] == [] and record[key]["source_hashes"] == {}
        assert record[key]["dim"] and record[key]["embed_model"]
    assert len(_json.dumps(record)) < 4000
    monkeypatch.undo()
    assert svc.recover_pending_mutations() == ["notes"]
    corpus = svc.registry.get("notes")
    assert len(corpus.sources) == 3 and len(corpus.source_hashes) == 3
    assert svc.search("changed marker", "notes")


def test_registry_is_saved_once_per_index_run_not_once_per_source(svc, make_doc, monkeypatch):
    """corpora.json was rewritten (2.4 MB, two fsyncs) after every source."""
    folder = make_doc("a.md", "alpha marker").parent
    for i in range(4):
        make_doc(f"b{i}.md", f"beta marker {i}")
    saves = {"n": 0}
    real_save = svc.registry._save

    def counted():
        saves["n"] += 1
        real_save()

    monkeypatch.setattr(svc.registry, "_save", counted)
    docs, _unused_chunks, skipped = svc.index(str(folder), "notes")
    assert docs >= 5
    assert saves["n"] == 1, f"registry saved {saves['n']}x for a {docs}-source run"
    corpus = svc.registry.get("notes")
    assert len(corpus.sources) == docs and len(corpus.source_hashes) == docs
    # a second run finds everything unchanged and saves at most once more
    # (updated_at moves); it must not go back to once per source
    _, _, skipped = svc.index(str(folder), "notes")
    assert skipped == docs and saves["n"] <= 2

# ── test_search_lock_timeout ──

@pytest.fixture
def two_corpora(svc, make_doc):
    svc.index(str(make_doc("a.md", "alpha bravo charlie delta echo")), "corpus-a")
    svc.index(str(make_doc("b.md", "alpha bravo foxtrot golf hotel")), "corpus-b")
    return svc


@pytest.fixture
def blocked_corpus_b(two_corpora):
    """Hold corpus-b's write lock for the duration of the test."""
    svc = two_corpora
    holding, release = threading.Event(), threading.Event()

    def writer():
        with svc.locks.write("corpus-b"):
            holding.set()
            release.wait(30)

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    assert holding.wait(5)
    try:
        yield svc
    finally:
        release.set()
        t.join(5)


def test_unscoped_search_answers_from_the_corpora_it_can_reach(
    blocked_corpus_b, monkeypatch
):
    svc = blocked_corpus_b
    monkeypatch.setattr(svc.settings, "search_lock_timeout_s", 0.3)

    started = time.monotonic()
    outcome = svc.search_with_diagnostics("alpha bravo", top_k=5)
    elapsed = time.monotonic() - started

    assert elapsed < 10, "unscoped search waited on the blocked corpus"
    assert outcome.results, "the reachable corpus returned nothing"
    assert {hit.corpus for hit in outcome.results} == {"corpus-a"}
    assert [w.corpus for w in outcome.warnings] == ["corpus-b"]


def test_named_corpus_search_reports_instead_of_hanging(
    blocked_corpus_b, monkeypatch
):
    svc = blocked_corpus_b
    monkeypatch.setattr(svc.settings, "search_lock_timeout_s", 0.3)

    started = time.monotonic()
    with pytest.raises(LockTimeout):
        svc.search("alpha bravo", corpus_name="corpus-b", top_k=5)
    assert time.monotonic() - started < 10


def test_an_unblocked_corpus_is_untouched_by_the_timeout(two_corpora, monkeypatch):
    svc = two_corpora
    monkeypatch.setattr(svc.settings, "search_lock_timeout_s", 0.3)
    outcome = svc.search_with_diagnostics("alpha bravo", top_k=5)
    assert outcome.warnings == []
    assert {hit.corpus for hit in outcome.results} == {"corpus-a", "corpus-b"}
