"""Per-corpus BM25 fusion weight.

The 2026-08 eval showed the right BM25 weight is a property of the CORPUS,
not the install: a notes corpus (short semantic memories) did best at 0.5
while a transcript corpus (names, dates, exact tokens, CJK phrases) needed
1.5. One global VECGREP_BM25_WEIGHT cannot serve both, so a corpus may pin
its own; unset falls through to the global.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from vecgrep.backend import service as sm
from vecgrep.backend.store.corpora import CorpusError


def _seed(tmp_path: Path) -> Path:
    root = tmp_path / "src"
    root.mkdir()
    (root / "a.md").write_text("alpha beta gamma delta.\n" * 3, encoding="utf-8")
    (root / "b.md").write_text("epsilon zeta eta theta.\n" * 3, encoding="utf-8")
    return root


def test_set_and_reset_bm25_weight(svc, tmp_path: Path) -> None:
    svc.index(str(_seed(tmp_path)), "docs")
    assert svc.registry.get("docs").bm25_weight is None
    c = svc.set_bm25_weight("docs", 0.5)
    assert c.bm25_weight == 0.5 and svc.registry.get("docs").bm25_weight == 0.5
    c = svc.set_bm25_weight("docs", None)
    assert c.bm25_weight is None
    with pytest.raises(CorpusError):
        svc.set_bm25_weight("docs", -1.0)


def test_corpus_weight_overrides_global_in_fusion(svc, tmp_path: Path, monkeypatch) -> None:
    svc.index(str(_seed(tmp_path)), "docs")
    seen = []
    real = sm._bm25_fusion_weight

    def spy(corpus):
        w = real(corpus)
        seen.append(w)
        return w

    monkeypatch.setattr(sm, "_bm25_fusion_weight", spy)
    monkeypatch.setattr(sm, "BM25_WEIGHT", 1.5)
    svc.search("alpha", "docs", mode="hybrid", top_k=3)
    assert seen[-1] == 1.5
    svc.set_bm25_weight("docs", 0.5)
    svc.search("alpha", "docs", mode="hybrid", top_k=3)
    assert seen[-1] == 0.5
