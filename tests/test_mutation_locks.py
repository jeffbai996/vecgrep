from __future__ import annotations

import multiprocessing
import threading
import time
from pathlib import Path

import pytest

from vecgrep.backend.mutation import CorpusLocks, LockTimeout, LockUpgradeError


def _hold_writer(root: str, corpus: str, ready, release) -> None:
    locks = CorpusLocks(Path(root))
    with locks.write(corpus):
        ready.set()
        release.wait(5)


def _time_writer(root: str, corpus: str, ready, output) -> None:
    locks = CorpusLocks(Path(root))
    ready.wait(5)
    started = time.monotonic()
    with locks.write(corpus):
        output.put(time.monotonic() - started)


def test_same_corpus_writers_serialize_across_processes(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    ready, release, output = ctx.Event(), ctx.Event(), ctx.Queue()
    first = ctx.Process(target=_hold_writer, args=(str(tmp_path), "notes", ready, release))
    second = ctx.Process(target=_time_writer, args=(str(tmp_path), "notes", ready, output))
    first.start()
    second.start()
    assert ready.wait(5)
    time.sleep(0.2)
    assert output.empty()
    release.set()
    assert output.get(timeout=5) >= 0.15
    first.join(5)
    second.join(5)
    assert first.exitcode == second.exitcode == 0


def test_different_corpora_do_not_block_each_other(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    ready, release, output = ctx.Event(), ctx.Event(), ctx.Queue()
    first = ctx.Process(target=_hold_writer, args=(str(tmp_path), "alpha", ready, release))
    second = ctx.Process(target=_time_writer, args=(str(tmp_path), "beta", ready, output))
    first.start()
    second.start()
    assert output.get(timeout=5) < 0.15
    release.set()
    first.join(5)
    second.join(5)
    assert first.exitcode == second.exitcode == 0


def test_nested_writer_may_read_but_reader_cannot_upgrade(tmp_path):
    locks = CorpusLocks(tmp_path)
    with locks.write("notes"):
        with locks.read("notes"):
            pass
    with locks.read("notes"):
        with pytest.raises(LockUpgradeError):
            with locks.write("notes"):
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
    with pytest.raises(LockTimeout):
        with locks.read("notes", timeout=0.3):
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
    with pytest.raises(LockTimeout):
        with locks.write("notes", timeout=0.3):
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
    with pytest.raises(LockTimeout):
        with locks.read("notes", timeout=0.3):
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
    with pytest.raises(LockTimeout):
        with CorpusLocks(tmp_path).read("notes", timeout=0.3):
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
