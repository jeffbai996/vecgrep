"""Temporal-decay tests (item 1b).

The correctness goal: a stale chunk must not outrank a fresh one on lexical
closeness alone when the corpus has a recency half-life set. Decay is applied
to the fused score BEFORE top_k truncation, and undated chunks are never
penalized (factor 1.0).
"""
from __future__ import annotations

import time

from vecgrep.backend.service import _recency_factor


def test_recency_factor_math() -> None:
    now = 1_000_000.0
    day = 86400.0
    # No half-life -> no decay.
    assert _recency_factor(now - 100 * day, None, now) == 1.0
    # No timestamp -> no decay.
    assert _recency_factor(None, 30.0, now) == 1.0
    # Exactly one half-life old -> 0.5.
    assert abs(_recency_factor(now - 30 * day, 30.0, now) - 0.5) < 1e-9
    # Two half-lives -> 0.25.
    assert abs(_recency_factor(now - 60 * day, 30.0, now) - 0.25) < 1e-9
    # Fresh (now) -> ~1.0.
    assert abs(_recency_factor(now, 30.0, now) - 1.0) < 1e-9
    # Future-dated (clock skew) clamps to 1.0, never boosts.
    assert _recency_factor(now + 100 * day, 30.0, now) == 1.0


def _saved(days_ago: float, body: str) -> str:
    ts = time.time() - days_ago * 86400.0
    from datetime import datetime, timezone

    iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    return f"Saved: {iso}\n\n{body}"


def test_decay_off_by_default_preserves_order(svc, make_doc):
    """With no half-life, a stale doc with strong lexical match still ranks."""
    fresh = make_doc("fresh.md", _saved(1, "alpha beta gamma delta."))
    stale = make_doc("stale.md", _saved(400, "alpha beta gamma epsilon."))
    svc.index(str(fresh), "c")
    svc.index(str(stale), "c")

    hits = svc.search("alpha beta gamma", "c", top_k=5)
    ids = {h.source_id for h in hits}
    assert str(fresh.resolve()) in ids
    assert str(stale.resolve()) in ids  # stale not dropped when decay off


def test_decay_demotes_stale_below_fresh(svc, make_doc):
    """With a short half-life, a 400-day-old chunk ranks below a 1-day-old one
    even when both match the query equally."""
    fresh = make_doc("fresh.md", _saved(1, "quantum widget assembly protocol."))
    stale = make_doc("stale.md", _saved(400, "quantum widget assembly protocol."))
    svc.index(str(fresh), "c")
    svc.index(str(stale), "c")

    svc.set_decay("c", 30.0)  # 30-day half-life
    hits = svc.search("quantum widget assembly", "c", top_k=5)

    pos = {h.source_id: i for i, h in enumerate(hits)}
    assert pos[str(fresh.resolve())] < pos[str(stale.resolve())], (
        "fresh chunk should rank ahead of the 400-day-old one under decay"
    )


def test_decay_explain_records_factor(svc, make_doc):
    p = make_doc("d.md", _saved(30, "recency factor under inspection here."))
    svc.index(str(p), "c")
    svc.set_decay("c", 30.0)
    hits = svc.search("recency factor", "c", top_k=3, explain=True)
    assert hits
    h = hits[0]
    assert "decay" in h.explain and "rrf" in h.explain
    # ~30 days at 30-day half-life -> ~0.5.
    assert 0.4 < h.explain["decay"] < 0.6


def test_set_decay_round_trips_and_clears(svc, make_doc):
    p = make_doc("d.md", "content")
    svc.index(str(p), "c")
    svc.set_decay("c", 14.0)
    assert svc.list_corpora()[0].decay_half_life_days == 14.0
    svc.set_decay("c", None)
    assert svc.list_corpora()[0].decay_half_life_days is None


def test_set_decay_rejects_nonpositive(svc, make_doc):
    p = make_doc("d.md", "content")
    svc.index(str(p), "c")
    import pytest
    from vecgrep.backend.store import CorpusError

    with pytest.raises(CorpusError):
        svc.set_decay("c", 0)
    with pytest.raises(CorpusError):
        svc.set_decay("c", -5)
