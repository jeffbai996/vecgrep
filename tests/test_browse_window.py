"""browse since/until (inclusive day range) + tail (newest-N events)."""
from __future__ import annotations

import pytest


def test_browse_since_until_range(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    # channel-a has 2026-01-08 and 2026-01-15; a range covering only the 15th.
    groups = svc.browse("evalchat", channel="channel-a",
                        since="2026-01-10", until="2026-01-15")
    assert {g["source_id"].split("/")[-1] for g in groups} == {"2026-01-15.md"}
    # until is inclusive of its whole day.
    groups = svc.browse("evalchat", channel="channel-a",
                        since="2026-01-08", until="2026-01-08")
    assert {g["source_id"].split("/")[-1] for g in groups} == {"2026-01-08.md"}


def test_browse_since_alone_is_a_valid_selector(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    groups = svc.browse("evalchat", since="2026-01-09")
    assert groups, "range-only browse must work without channel/date"
    for g in groups:
        assert g["source_id"].endswith("2026-01-15.md")


def test_browse_tail_keeps_newest_events(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    full = svc.browse("evalchat", channel="channel-a")
    all_events = [e for g in full for e in g["events"]]
    assert len(all_events) > 5
    tailed = svc.browse("evalchat", channel="channel-a", tail=5)
    got = [e for g in tailed for e in g["events"]]
    assert len(got) == 5
    assert got == all_events[-5:], "tail must be the newest events, in order"


def test_browse_tail_larger_than_corpus_returns_everything(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    full = svc.browse("evalchat", channel="channel-a")
    tailed = svc.browse("evalchat", channel="channel-a", tail=10_000)
    assert [e for g in tailed for e in g["events"]] == \
        [e for g in full for e in g["events"]]


def test_browse_bad_range_dates_raise(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    with pytest.raises(ValueError):
        svc.browse("evalchat", since="not-a-date")
    with pytest.raises(ValueError):
        svc.browse("evalchat", until="13/01/2026")
