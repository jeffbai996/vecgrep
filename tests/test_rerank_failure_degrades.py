"""A reranker that breaks at predict time must not take the search with it.

Readiness only proves the model loaded. `predict()` can fail afterwards, and
on 2026-08-20 it did: the host GPU began reporting a nonsense VRAM figure and
every rerank raised `CUDA error: unknown error`. The search endpoint returned
500 to every bot in the squad, for a corpus that had perfectly good hits sitting
in the fused result set. Reranking reorders an answer we already have — losing
it is a quality hit, not a failure.
"""
import pytest

from vecgrep.backend import service as svc_mod


def test_a_rerank_that_raises_still_returns_results(svc, make_doc, monkeypatch):
    svc.index(str(make_doc("d.md", "Alpha topic. Beta topic. Gamma topic.")), "c")
    monkeypatch.setattr(svc, "_rerank_ready", lambda name: True)

    def _boom(*a, **kw):
        raise RuntimeError("CUDA error: unknown error")

    monkeypatch.setattr(svc, "_apply_rerank", _boom)
    hits = svc.search("alpha", "c", top_k=5, rerank=True)
    assert hits, "a broken reranker must not empty the result set"


def test_results_are_not_marked_reranked_when_it_failed(svc, make_doc, monkeypatch):
    svc.index(str(make_doc("d.md", "Alpha topic. Beta topic. Gamma topic.")), "c")
    monkeypatch.setattr(svc, "_rerank_ready", lambda name: True)
    monkeypatch.setattr(svc, "_apply_rerank",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    hits = svc.search("alpha", "c", top_k=5, rerank=True)
    assert all("rerank" not in h.matched_by for h in hits)


def test_a_working_reranker_is_untouched(svc, make_doc, monkeypatch):
    svc.index(str(make_doc("d.md", "Alpha topic. Beta topic. Gamma topic.")), "c")
    monkeypatch.setattr(svc, "_rerank_ready", lambda name: True)
    seen = {}

    def _ok(query, candidates, top_k, model_name, explain=False):
        seen["called"] = True
        return candidates[:top_k]

    monkeypatch.setattr(svc, "_apply_rerank", _ok)
    hits = svc.search("alpha", "c", top_k=5, rerank=True)
    assert seen.get("called") and hits
