"""Service-level orphan sweep: the keep-set is derived from what the LIVE
corpora actually hold, per embedding identity, so a sweep never evicts a
vector a re-index of any registered corpus would ask for."""
from __future__ import annotations

from pathlib import Path

from vecgrep.backend.embed.cache import EmbedCache


def _seed(tmp_path: Path) -> Path:
    root = tmp_path / "src"
    root.mkdir()
    (root / "a.md").write_text("alpha beta gamma. delta epsilon zeta.\n", encoding="utf-8")
    (root / "b.md").write_text("one two three. four five six.\n", encoding="utf-8")
    return root


def test_cache_sweep_keeps_every_live_chunk_and_drops_the_rest(svc, tmp_path: Path) -> None:
    root = _seed(tmp_path)
    svc.index(str(root), "docs")
    cache: EmbedCache = svc._embed_cache
    identity = svc._backend_cache["auto"]._inner.identity
    before = sum(cache.stats().values())
    assert before > 0
    # an orphan under the live identity and a whole identity no corpus uses
    cache.put_many(identity, ["orphaned chunk text"], [[0.0] * 16])
    cache.put_many("ollama:retired-model", ["anything"], [[0.0] * 16])
    plan = svc.cache_sweep(dry_run=True)
    assert plan["deleted"] == {identity: 1, "ollama:retired-model": 1}
    assert plan["kept"][identity] == before
    assert sum(cache.stats().values()) == before + 2, "dry run deletes nothing"
    done = svc.cache_sweep()
    assert done["deleted"] == {identity: 1, "ollama:retired-model": 1}
    assert cache.stats() == {identity: before}
    # every live chunk text is still a hit
    texts = [pl["text"] for _pid, pl in svc.store.iter_payloads("vecgrep__docs", include_payload_fields={"text"})]
    assert texts and len(cache.get_many(identity, texts)) == len(set(texts))
