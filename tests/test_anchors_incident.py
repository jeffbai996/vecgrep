"""Phase 8 (stretch) — precise source anchors + incident object.

8a — anchors: every result carries line_start/line_end (1-based, inclusive)
computed from the source text, plus the stable chunk_id it already had, so
an assistant can cite and re-open the exact region ("file#L12-L24").

8b — incident: a single structured answer for "what happened?" assembled
from the existing primitives (search → timeline): title, sources,
participants, time range, primary timeline vs related context, confidence.
No new retrieval — pure assembly.
"""
from __future__ import annotations


# ── 8a: line anchors ─────────────────────────────────────────────────────────

def test_results_carry_line_ranges(svc, make_doc) -> None:
    p = make_doc(
        "lines.md",
        "alpha first line here\nbeta second line here\ngamma third line here\n"
        "delta fourth line here\nepsilon fifth line here\n",
    )
    svc.index(str(p), "c")
    hits = svc.search("gamma third line", "c", top_k=3)
    assert hits
    h = hits[0]
    assert h.line_start is not None and h.line_end is not None
    assert 1 <= h.line_start <= h.line_end
    # The chunk containing 'gamma' must include line 3 in its range.
    assert h.line_start <= 3 <= h.line_end


def test_anchor_string_format(svc, make_doc) -> None:
    p = make_doc("anchor.md", "one\ntwo\nthree\nfour anchor target line\nfive\n")
    svc.index(str(p), "c")
    h = svc.search("four anchor target", "c", top_k=1)[0]
    a = h.anchor
    assert a.startswith(h.source_id)
    assert f"#L{h.line_start}" in a
    assert a.endswith(f"-L{h.line_end}") or a.endswith(f"#L{h.line_start}")


def test_eval_fixture_lines_are_plausible(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    hits = svc.search("ERR-0451", "evalchat", top_k=5)
    for h in hits:
        assert h.line_start is not None and h.line_start >= 1
        assert h.line_end >= h.line_start


# ── 8b: incident object ──────────────────────────────────────────────────────

def test_incident_assembles_structured_answer(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    inc = svc.incident("relay-service ERR-0451 outage", "evalchat")
    assert inc is not None
    assert inc["title"] == "relay-service ERR-0451 outage"
    assert inc["sources"], "must name its evidence files"
    assert "alice" in inc["participants"]
    assert inc["confidence"] in ("exact", "strong", "related", "weak")
    # Primary timeline is a chronological event list.
    events = inc["primary_timeline"]
    assert len(events) >= 5
    times = [e["time"] for e in events]
    assert times == sorted(times)


def test_incident_separates_primary_from_related(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    inc = svc.incident("relay-service ERR-0451 outage", "evalchat")
    primary_files = {inc["primary_source"]}
    related_files = {g["source_id"] for g in inc["related"]}
    assert primary_files.isdisjoint(related_files), (
        "the primary file must not repeat in related context"
    )


def test_incident_time_range_spans_events(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    inc = svc.incident("relay-service ERR-0451 outage", "evalchat")
    tr = inc["time_range"]
    assert tr["start"] and tr["end"] and tr["start"] <= tr["end"]


def test_incident_respects_filters(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    inc = svc.incident(
        "relay-service ERR-0451 outage", "evalchat",
        filters=["date:2026-01-15"],
    )
    all_sources = [inc["primary_source"], *(g["source_id"] for g in inc["related"])]
    assert all("2026-01-08" not in s for s in all_sources)


def test_incident_none_when_nothing_found(svc) -> None:
    from tests import eval_harness as eh

    eh.build_eval_corpus(svc)
    assert svc.incident("qqqzzznope", "evalchat", mode="bm25") is None
