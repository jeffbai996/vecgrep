"""BM25 sidecar write amplification.

`BM25Store.upsert` re-pickled the WHOLE corpus index after every source. A
directory index of N documents therefore wrote the growing pickle N times --
O(N^2) bytes; on a 1,000-transcript corpus with a 250 MB sidecar that is
~125 GB of writes for one rebuild, and it is why a warm-cache side-by-side
build took 45 minutes. In bulk mode the store persists once at the end and
leaves a dirty marker while the in-memory index is ahead of the file, so a
crash mid-bulk is repaired (rebuilt from qdrant, the canonical store) the
next time a service comes up instead of silently drifting.
"""
from __future__ import annotations

from pathlib import Path

from vecgrep.backend.store.bm25_store import BM25Store


def _seed(tmp_path: Path, n: int = 6) -> Path:
    root = tmp_path / "src"
    root.mkdir()
    for i in range(n):
        (root / f"doc-{i}.md").write_text(
            f"document number {i} about topic-{i} and shared-word.\n", encoding="utf-8")
    return root


def test_directory_index_persists_bm25_once_and_stays_searchable(svc, tmp_path, monkeypatch) -> None:
    root = _seed(tmp_path)
    calls = []
    real = BM25Store._persist_now

    def counting(self, corpus):
        calls.append(corpus)
        return real(self, corpus)

    monkeypatch.setattr(BM25Store, "_persist_now", counting)
    docs, chunks, _ = svc.index(str(root), "docs")
    assert docs == 6
    assert calls.count("docs") == 1, f"expected ONE persist for a 6-doc directory, got {calls.count('docs')}"
    assert not (svc.settings.home / "bm25" / "docs.pkl.dirty").exists()
    for i in range(6):
        hits = svc.search(f"topic-{i}", "docs", mode="bm25", top_k=3)
        assert hits and hits[0].source_id.endswith(f"doc-{i}.md")


def test_single_source_index_still_persists_immediately(svc, tmp_path, monkeypatch) -> None:
    root = _seed(tmp_path, n=1)
    calls = []
    real = BM25Store._persist_now
    monkeypatch.setattr(BM25Store, "_persist_now", lambda self, c: (calls.append(c), real(self, c))[1])
    svc.index(str(root / "doc-0.md"), "docs")
    assert calls.count("docs") == 1


def test_interrupted_bulk_leaves_marker_and_next_service_rebuilds_from_qdrant(svc, tmp_path) -> None:
    root = _seed(tmp_path)
    svc.index(str(root), "docs")
    bm25_dir = svc.settings.home / "bm25"
    # Simulate a crash mid-bulk: the marker is down and the on-disk pickle is
    # stale (we delete it outright -- worst case).
    (bm25_dir / "docs.pkl.dirty").write_text("crash", encoding="utf-8")
    (bm25_dir / "docs.pkl").unlink()
    svc.bm25.evict("docs")
    inner = svc._backend_cache["auto"]._inner
    svc.store.client.close()   # embedded qdrant admits one process at a time

    from vecgrep.backend.service import VecgrepService
    from vecgrep.backend.embed.cache import CachedBackend
    svc2 = VecgrepService(ephemeral=False)
    svc2._backend_cache["auto"] = CachedBackend(inner, svc2._embed_cache)
    try:
        assert not (bm25_dir / "docs.pkl.dirty").exists(), "startup must clear the marker"
        assert (bm25_dir / "docs.pkl").exists(), "startup must rebuild the sidecar"
        expected = svc2.store.count("vecgrep__docs")
        assert svc2.bm25.count("docs") == expected
        hits = svc2.search("topic-3", "docs", mode="bm25", top_k=3)
        assert hits and hits[0].source_id.endswith("doc-3.md")
    finally:
        try:
            svc2.store.client.close()
        except Exception:
            pass
