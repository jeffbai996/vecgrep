"""The weekly-eval timer's alert rule.

The script itself lives outside the repo (it names the real host's paths and
the real alerts channel); what belongs under test is the decision it makes —
when to wake a human. That decision is a pure function of two summary dicts,
so it is tested directly and the subprocess/Discord halves are left alone.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path.home() / "scripts" / "vecgrep-weekly-eval.py"
pytestmark = pytest.mark.skipif(
    not SCRIPT.exists(), reason="host-side weekly eval script not installed")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("vecgrep_weekly_eval", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)      # type: ignore[union-attr]
    return m


def test_no_alert_when_nothing_moved(mod):
    prev = {"hit@3": 79.6, "neg_fp_rate": 3.8}
    cur = {"hit@3": 80.1, "neg_fp_rate": 5.0}
    assert mod.significant_deltas(prev, cur) == {}


def test_alert_when_hit_at_3_drops_more_than_three_points(mod):
    prev = {"hit@3": 82.8, "neg_fp_rate": 3.8}
    cur = {"hit@3": 75.0, "neg_fp_rate": 3.8}
    d = mod.significant_deltas(prev, cur)
    assert set(d) == {"hit@3"}
    assert d["hit@3"][2] == pytest.approx(-7.8)


def test_alert_when_hit_at_3_improves_past_the_threshold(mod):
    # A big improvement is also news: it usually means something changed that
    # nobody wrote down.
    d = mod.significant_deltas({"hit@3": 70.0}, {"hit@3": 82.0})
    assert d["hit@3"][2] == pytest.approx(12.0)


def test_alert_when_negative_false_positives_climb(mod):
    prev = {"hit@3": 80.0, "neg_fp_rate": 3.8}
    cur = {"hit@3": 80.0, "neg_fp_rate": 30.8}
    d = mod.significant_deltas(prev, cur)
    assert set(d) == {"neg_fp_rate"}


def test_threshold_is_exclusive_so_exactly_three_points_stays_quiet(mod):
    assert mod.significant_deltas({"hit@3": 80.0}, {"hit@3": 83.0}) == {}
    assert mod.significant_deltas({"hit@3": 80.0}, {"hit@3": 83.1}) != {}


def test_first_run_with_no_previous_report_never_alerts(mod):
    assert mod.significant_deltas(None, {"hit@3": 12.0, "neg_fp_rate": 99.0}) == {}


def test_missing_metric_is_not_treated_as_a_move(mod):
    assert mod.significant_deltas({"hit@3": 80.0}, {"mrr": 0.7}) == {}
    assert mod.significant_deltas({"hit@3": None}, {"hit@3": 10.0}) == {}


def test_tracked_summary_picks_the_named_config_row(mod, tmp_path: Path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps([
        {"config": {"name": "other"}, "summary": {"hit@3": 1.0}},
        {"config": {"name": mod.TRACKED}, "summary": {"hit@3": 2.0}},
    ]), encoding="utf-8")
    assert mod.tracked_summary(p) == {"hit@3": 2.0}


def test_tracked_summary_survives_a_corrupt_report(mod, tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert mod.tracked_summary(p) is None
    assert mod.tracked_summary(tmp_path / "missing.json") is None


def test_card_text_is_one_line_and_names_the_move(mod):
    text = mod.card_text("2026-09-13", {"hit@3": (82.8, 74.0, -8.8)},
                         {"hit@1": 55.9, "mrr": 0.702, "lat_p50_ms": 2425.0})
    assert "\n" not in text
    assert "hit@3 82.8 -> 74.0 (-8.8)" in text
    assert "2026-09-13" in text
