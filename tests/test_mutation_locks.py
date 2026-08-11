from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

import pytest

from vecgrep.backend.mutation import CorpusLocks, LockUpgradeError


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
