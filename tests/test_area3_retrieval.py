"""Audit reproduction tests for Area 3: Retrieval."""
from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from vecgrep.backend.config import Settings
from vecgrep.backend.embed.base import EmbedBackend, EmbedBackendError
from vecgrep.backend.embed.ollama import OllamaBackend
from vecgrep.backend.service import VecgrepService


class _DeterministicEmbedder(EmbedBackend):
    name = "ollama"
    model = "nomic-embed-text"
    dim = 16

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            # Make vectors containing "target_keyword" identical so cosine == 1.0
            key = "target_keyword" if "target_keyword" in t.lower() else t
            h = hashlib.sha256(key.encode("utf-8")).digest()
            vec = [(b / 127.5) - 1.0 for b in h[: self.dim]]
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out


def _make_service(tmp_path: Path) -> VecgrepService:
    settings = Settings(home=tmp_path / "vg_home", qdrant_url=None)
    svc = VecgrepService(settings=settings, ephemeral=False)
    svc._backend_cache["auto"] = _DeterministicEmbedder()
    return svc


def test_f_retr_01a_qdrant_store_search_swallows_backend_failure_as_empty_results(tmp_path: Path):
    """F-RETR-01a: QdrantStore.search has bare `except Exception: return []`, turning a closed
    or broken Qdrant connection into a silent 0-hit result with no warning or error."""
    svc = _make_service(tmp_path)
    doc = tmp_path / "doc.md"
    doc.write_text("# Target\n\nThis document contains target_keyword for vector search.\n", encoding="utf-8")
    svc.index(str(doc), "notes")

    # Simulate Qdrant client failure / closed client during search
    svc.store.client.close()

    # BUG: Instead of raising an error or emitting a SearchWarning, vector search silently returns []!
    from vecgrep.backend.store.qdrant_store import SearchBackendError
    with pytest.raises(SearchBackendError, match="unavailable"):
        svc.search_with_diagnostics("target_keyword", "notes", mode="vector")


def test_f_retr_01b_ollama_5xx_returns_zero_vector_for_query_instead_of_raising():
    """F-RETR-01b: OllamaBackend.embed_one returns [0.0]*dim on HTTP 500/503 instead of raising
    EmbedBackendError, preventing failover and silently zeroing out query vector similarity."""
    backend = OllamaBackend(base_url="http://127.0.0.1:9", model="nomic-embed-text")

    def _mock_post(url: str, json: dict):
        return httpx.Response(503, text="model runner crashed / overloaded")

    backend._client.post = _mock_post  # type: ignore[method-assign]

    # BUG: embed_one("search query") returns [0.0] * 768 instead of raising EmbedBackendError!
    with pytest.raises(EmbedBackendError):
        backend.embed_one("search query")


def test_f_retr_01c_ollama_unknown_model_probe_5xx_raises_attribute_error():
    """F-RETR-01c: When probing dim for an unknown model in OllamaBackend.__init__, a 5xx response
    reaches `return [0.0] * self.dim` before `self.dim` is assigned, raising AttributeError."""
    orig_make = OllamaBackend._make_client

    def _failing_client(connect: float = 2.0, read: float = 60.0):
        client = orig_make(connect=connect, read=read)
        client.post = lambda url, json: httpx.Response(500, text="internal server error")  # type: ignore[method-assign]
        return client

    OllamaBackend._make_client = staticmethod(_failing_client)  # type: ignore[assignment]
    try:
        # Should raise EmbedBackendError, not AttributeError: 'OllamaBackend' object has no attribute 'dim'
        with pytest.raises(EmbedBackendError):
            OllamaBackend(base_url="http://127.0.0.1:9", model="custom-unknown-embed-model")
    finally:
        OllamaBackend._make_client = staticmethod(orig_make)  # type: ignore[assignment]


def test_f_retr_02_superseded_chunks_starve_active_results_without_qdrant_pushdown(tmp_path: Path):
    """F-RETR-02: Default `meta.status=active` filter is not passed to _build_qdrant_filter (which
    only receives user_filters and doesn't push down meta.* anyway). When 55 superseded docs have
    matching vectors, fetch_k=50 only retrieves superseded docs and post-filtering drops all of them,
    returning 0 results even though an active matching doc exists in the corpus!"""
    svc = _make_service(tmp_path)
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    # Create 55 superseded docs and 1 active doc, all matching "target_keyword"
    for i in range(55):
        (docs_dir / f"old_{i:02d}.md").write_text(
            f"---\nstatus: superseded\n---\n\nSuperseded note {i} with target_keyword in body.\n",
            encoding="utf-8",
        )
    (docs_dir / "active_doc.md").write_text(
        "---\nstatus: active\n---\n\nActive current truth document with target_keyword in body.\n",
        encoding="utf-8",
    )

    svc.index(str(docs_dir), "notes")
    # In vector mode with default top_k=5 (fetch_k=50), Qdrant returns 50 points without filtering
    # metadata.status == "active", and Python post-filtering drops all 50 superseded hits!
    hits = svc.search("target_keyword", "notes", top_k=5, mode="vector")
    assert len(hits) == 1
    assert "active_doc.md" in hits[0].source_id


def test_f_retr_03_rank_weight_ignored_in_vector_and_bm25_modes(tmp_path: Path):
    """F-RETR-03: corpus.rank_weight is applied in hybrid mode but completely ignored in vector and bm25 modes."""
    svc = _make_service(tmp_path)
    d1 = tmp_path / "low.md"
    d2 = tmp_path / "high.md"
    d1.write_text("# Doc\n\nShared target_keyword content for ranking test.\n", encoding="utf-8")
    d2.write_text("# Doc\n\nShared target_keyword content for ranking test.\n", encoding="utf-8")

    svc.index(str(d1), "low_corpus")
    svc.index(str(d2), "high_corpus")
    svc.set_rank_weight("low_corpus", 0.1)
    svc.set_rank_weight("high_corpus", 10.0)

    vec_hits = svc.search("target_keyword", corpus_names=["low_corpus", "high_corpus"], mode="vector")
    by_corpus = {h.corpus: h.score for h in vec_hits}
    # BUG: In vector mode, r.score is raw cosine (1.0 for both), ignoring rank_weight (10.0 vs 0.1)!
    assert by_corpus["high_corpus"] > by_corpus["low_corpus"]


def test_f_retr_04_incident_time_range_inverted_on_single_digit_hour(tmp_path: Path):
    """F-RETR-04: incident() computes time_range start/end via lexicographical min/max on 'YYYY-MM-DD H:MM',
    where '2026-10-01 9:05' > '2026-10-01 10:15', inverting start and end."""
    svc = _make_service(tmp_path)
    chat = tmp_path / "2026-10-01-general.md"
    chat.write_text(
        "**alice** · 9:05\n"
        "> Outage started on the primary target_keyword database cluster.\n\n"
        "**bob** · 10:15\n"
        "> Failover complete and target_keyword database cluster recovered.\n",
        encoding="utf-8",
    )
    svc.index(str(chat), "chats")

    inc = svc.incident("target_keyword", "chats")
    assert inc is not None
    # Chronologically, 9:05 is the start and 10:15 is the end.
    # BUG: min(["2026-10-01 9:05", "2026-10-01 10:15"]) == "2026-10-01 10:15"!
    assert inc["time_range"]["start"] == "2026-10-01 9:05" or inc["time_range"]["start"] == "2026-10-01 09:05"
    assert inc["time_range"]["end"] == "2026-10-01 10:15"


def test_f_retr_05_related_returns_superseded_and_expired_chunks(tmp_path: Path):
    """F-RETR-05: related() does not filter out superseded (status=superseded) or expired (expires_at) chunks."""
    svc = _make_service(tmp_path)
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    seed = docs_dir / "seed.md"
    superseded = docs_dir / "superseded.md"
    expired = docs_dir / "expired.md"

    seed.write_text("---\nstatus: active\n---\n\nSeed note with target_keyword.\n", encoding="utf-8")
    superseded.write_text("---\nstatus: superseded\n---\n\nSuperseded note with target_keyword.\n", encoding="utf-8")
    expired.write_text(
        "---\nstatus: active\nexpires_at: 2020-01-01T00:00:00+00:00\n---\n\nExpired note with target_keyword.\n",
        encoding="utf-8",
    )
    svc.index(str(docs_dir), "notes")

    seed_hits = svc.search("target_keyword", "notes")
    assert len(seed_hits) == 1  # search() properly filters out superseded and expired
    seed_cid = seed_hits[0].chunk_id

    # BUG: related() returns both the superseded and expired chunks!
    neighbours = svc.related(seed_cid, "notes", top_k=5)
    assert neighbours == []
