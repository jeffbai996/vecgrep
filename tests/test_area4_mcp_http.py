"""Audit reproduction tests for Area 4: MCP and HTTP layers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vecgrep.backend.api import routes
from vecgrep.backend.embed.base import EmbedBackend, EmbedBackendError
from vecgrep.backend.main import create_app
from vecgrep.backend.service import VecgrepService
from vecgrep.mcp import server as mcp_server


class _FakeEmbedder(EmbedBackend):
    name = "ollama"
    model = "nomic-embed-text"
    dim = 16
    fail_now = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.fail_now:
            raise EmbedBackendError("simulated embedding backend outage")
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec = [(b / 127.5) - 1.0 for b in h[: self.dim]]
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / norm for x in vec])
        return out


def _setup_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[VecgrepService, _FakeEmbedder]:
    from vecgrep.backend import config as cfg_mod

    home = tmp_path / "vg_home"
    monkeypatch.setenv("VECGREP_HOME", str(home))
    monkeypatch.setenv("VECGREP_QDRANT_URL", "")
    cfg_mod.reset_settings()
    settings = cfg_mod.get_settings()
    embedder = _FakeEmbedder()
    svc = VecgrepService(settings=settings, ephemeral=False)
    svc.store.client.close()
    svc.store = VecgrepService(settings=settings, ephemeral=True).store
    svc._backend_cache["auto"] = embedder
    monkeypatch.setattr(routes, "_SERVICE", svc)
    monkeypatch.setattr(mcp_server, "_SVC", (id(settings), svc))
    return svc, embedder


def test_f_api_01_mcp_direct_edit_fails_when_corpus_has_existing_external_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """F-API-01: _commit_direct_edit validates target in _write_dir(corpus) and reads it via
    _read_nofollow(target), then calls _doc_body(corpus, doc_id) which resolves _corpus_doc_dir(corpus)
    to Path(corpus.sources[0]).parent (and also instantiates a brand-new VecgrepService()!).
    When corpus.sources[0] lives outside _write_dir(corpus), _doc_body returns None and direct edit fails!"""
    svc, _ = _setup_service(tmp_path, monkeypatch)
    monkeypatch.setenv("VECGREP_DIRECT_WRITE_CORPUS", "notes")

    # Index an initial file into "notes" from an external directory
    ext_dir = tmp_path / "external_docs"
    ext_dir.mkdir()
    ext_file = ext_dir / "seed.md"
    ext_file.write_text("# Seed\n\nInitial external note in the notes corpus.\n", encoding="utf-8")
    svc.index(str(ext_file), "notes")

    # Now use MCP direct `write` tool to create a new entry in _write_dir("notes")
    write_res = json.loads(mcp_server._run_direct_write("Original body text to be patched.", title="Test"))
    assert write_res.get("committed") is True
    doc_id = write_res["doc_id"]

    # Now use MCP direct `edit` tool to patch the entry we just created
    edit_res = json.loads(
        mcp_server._run_direct_edit(
            doc_id,
            old_str="Original body text",
            new_str="Updated body text",
        )
    )
    # BUG: _commit_direct_edit fails with {"error": "could not read the body of '...'."}
    assert edit_res.get("committed") is True, f"Direct edit failed: {edit_res}"


def test_f_api_02_http_compare_unhandled_500_on_embed_outage_and_bad_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """F-API-02: POST /api/compare does not catch EmbedBackendError (returning 500 instead of 503)
    and accepts raw dict without validating query/top_k types (raising AttributeError/ValueError -> 500)."""
    svc, embedder = _setup_service(tmp_path, monkeypatch)
    doc = tmp_path / "doc.md"
    doc.write_text("# Doc\n\nSome indexed text for compare endpoint.\n", encoding="utf-8")
    svc.index(str(doc), "notes")

    client = TestClient(create_app(), base_url="http://127.0.0.1:8765", raise_server_exceptions=False)

    # 1. Malformed top_k -> unhandled ValueError -> 500 instead of 400/422
    r_bad = client.post("/api/compare", json={"query": "indexed", "corpus": "notes", "top_k": "not-an-int"})
    assert r_bad.status_code in (400, 422), f"Expected 400/422, got {r_bad.status_code}"

    # 2. EmbedBackendError during /api/compare -> unhandled EmbedBackendError -> 500 instead of 503
    embedder.fail_now = True
    r_outage = client.post("/api/compare", json={"query": "indexed", "corpus": "notes", "mode": "vector"})
    assert r_outage.status_code == 503, f"Expected 503, got {r_outage.status_code}"


def test_f_api_03_negative_full_k_and_negative_timeline_padding_invert_slices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """F-API-03: Negative numeric arguments (full_k=-1, padding=-5000, max_groups=-1) are not rejected
    and slice Python lists from the tail or invert slice_start > slice_end."""
    svc, _ = _setup_service(tmp_path, monkeypatch)
    d1 = tmp_path / "a.md"
    d2 = tmp_path / "b.md"
    d1.write_text("# Alpha\n\nFirst document matching query keyword.\n", encoding="utf-8")
    d2.write_text("# Beta\n\nSecond document matching query keyword.\n", encoding="utf-8")
    svc.index(str(d1), "notes")
    svc.index(str(d2), "notes")

    client = TestClient(create_app(), base_url="http://127.0.0.1:8765", raise_server_exceptions=False)

    # 1. Budget search with full_k=-1 slices results[:-1] into hits and results[-1:] into stubs!
    r_search = client.post(
        "/api/search",
        json={"query": "document matching query keyword", "corpus": "notes", "budget": True, "full_k": -1},
    )
    # Should either reject full_k < 0 with 400/422 or return 0 full hits, not len(results)-1 full hits!
    assert r_search.status_code in (400, 422) or len(r_search.json()["hits"]) == 0

    # 2. Timeline with negative padding (-5000) produces slice_start > slice_end and empty slice_text!
    r_tl = client.post(
        "/api/timeline",
        json={"query": "document matching query keyword", "corpus": "notes", "padding": -5000},
    )
    assert r_tl.status_code in (400, 422) or all(
        g["slice_start"] <= g["slice_end"] and g["slice_text"] for g in r_tl.json()
    )


def test_f_api_04_ollama_read_timeout_retries_on_fallback_path_multiplying_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    """F-API-04: When _embed_batch hits httpx.ReadTimeout, it returns None instead of raising
    EmbedBackendError, causing OllamaBackend.embed() to fall back to _embed_one_resilient()
    and make a second full-timeout request (or up to 64 requests on a batch)."""
    import httpx

    from vecgrep.backend.embed.ollama import OllamaBackend

    backend = OllamaBackend(base_url="http://127.0.0.1:9", model="nomic-embed-text")
    calls = 0

    def _timeout_post(url: str, json: dict):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("simulated read timeout")

    monkeypatch.setattr(backend._client, "post", _timeout_post)

    with pytest.raises(EmbedBackendError):
        backend.embed(["chunk one", "chunk two"])

    # BUG: _embed_batch swallowed the ReadTimeout and returned None, so _embed_one_resilient
    # made a second call (calls == 2 instead of failing fast after 1 timeout).
    assert calls == 1


def test_f_api_05_admin_backup_crashes_with_500_when_live_service_holds_embedded_qdrant_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """F-API-05: POST /api/admin/backups instantiates a brand-new VecgrepService() via
    BackupManager.service_factory while routes._SERVICE already holds the embedded Qdrant
    storage lock, crashing with an unhandled RuntimeError (HTTP 500)."""
    from vecgrep.backend import config as cfg_mod

    home = tmp_path / "vg_home"
    monkeypatch.setenv("VECGREP_HOME", str(home))
    monkeypatch.setenv("VECGREP_QDRANT_URL", "")
    cfg_mod.reset_settings()
    settings = cfg_mod.get_settings()
    embedder = _FakeEmbedder()
    svc = VecgrepService(settings=settings, ephemeral=False)
    svc._backend_cache["auto"] = embedder
    monkeypatch.setattr(routes, "_SERVICE", svc)

    try:
        doc = tmp_path / "doc.md"
        doc.write_text("# Doc\n\nContent to back up while embedded Qdrant is open.\n", encoding="utf-8")
        svc.index(str(doc), "notes")

        client = TestClient(
            create_app(),
            base_url="http://127.0.0.1:8765",
            client=("127.0.0.1", 50000),
            raise_server_exceptions=False,
        )
        resp = client.post("/api/admin/backups")
        # BUG: BackupManager.create() calls VecgrepService() while routes._SERVICE holds the
        # embedded Qdrant lock, raising an unhandled RuntimeError -> 500 Internal Server Error!
        assert resp.status_code == 200, f"Expected 200 OK, got {resp.status_code}: {resp.text}"
    finally:
        svc.store.client.close()


def test_f_api_01_propose_edit_on_external_corpus_with_embedded_qdrant_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """F-API-01b: _corpus_doc_dir() must use _svc().list_corpora() rather than instantiating
    VecgrepService(), which fails on the embedded Qdrant lock and falls back to _write_dir."""
    from vecgrep.backend import config as cfg_mod

    home = tmp_path / "vg_home"
    monkeypatch.setenv("VECGREP_HOME", str(home))
    monkeypatch.setenv("VECGREP_QDRANT_URL", "")
    monkeypatch.setenv("VECGREP_PROPOSE_ALLOWED_CORPORA", "mirror_notes")
    cfg_mod.reset_settings()
    settings = cfg_mod.get_settings()
    svc = VecgrepService(settings=settings, ephemeral=False)
    svc._backend_cache["auto"] = _FakeEmbedder()
    monkeypatch.setattr(mcp_server, "_SVC", (id(settings), svc))

    try:
        ext_dir = tmp_path / "mirror_source"
        ext_dir.mkdir()
        doc = ext_dir / "entry-01.md"
        doc.write_text(
            "---\nid: entry-01\nstatus: active\n---\n\nOriginal mirror body text.\n",
            encoding="utf-8",
        )
        svc.index(str(doc), "mirror_notes")

        res = json.loads(
            mcp_server._run_propose(
                "mirror_notes",
                None,
                edit_id="entry-01",
                old_str="Original mirror body text.",
                new_str="Patched mirror body text.",
            )
        )
        assert "proposal_id" in res, f"Expected proposal_id, got {res}"
    finally:
        svc.store.client.close()


def test_f_api_03_direct_python_negative_bounds_clamped_and_full_k_zero_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """F-API-03b: split_full_and_stubs and build_timeline must clamp negative bounds when
    called from Python/MCP, and SearchRequest must allow full_k=0 for stub-only budget responses."""
    svc, _ = _setup_service(tmp_path, monkeypatch)
    d1 = tmp_path / "a.md"
    d2 = tmp_path / "b.md"
    d1.write_text("# Alpha\n\nFirst document matching query keyword.\n", encoding="utf-8")
    d2.write_text("# Beta\n\nSecond document matching query keyword.\n", encoding="utf-8")
    svc.index(str(d1), "notes")
    svc.index(str(d2), "notes")

    full, stubs = svc.search_budgeted("document matching query keyword", "notes", full_k=-1)
    assert full == []
    assert len(stubs) == 2

    groups = svc.timeline("document matching query keyword", "notes", padding=-5000, max_groups=4)
    assert all(g["slice_start"] <= g["slice_end"] and g["slice_text"] for g in groups)
    assert svc.timeline("document matching query keyword", "notes", max_groups=-1) == []

    client = TestClient(create_app(), base_url="http://127.0.0.1:8765", raise_server_exceptions=False)
    r_zero = client.post(
        "/api/search",
        json={"query": "document matching query keyword", "corpus": "notes", "budget": True, "full_k": 0},
    )
    assert r_zero.status_code == 200
    assert r_zero.json()["hits"] == []
    assert len(r_zero.json()["stubs"]) == 2


def test_f_api_04_openai_backend_configures_bounded_timeout(monkeypatch: pytest.MonkeyPatch):
    """F-API-04b: OpenAIBackend must pass a bounded timeout and max_retries to the OpenAI client."""
    import sys
    import types

    from vecgrep.backend.embed.openai import OpenAIBackend

    captured: dict = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_mod = types.ModuleType("openai")
    monkeypatch.setattr(fake_mod, "OpenAI", _FakeOpenAI, raising=False)
    monkeypatch.setitem(sys.modules, "openai", fake_mod)

    OpenAIBackend(api_key="sk-test", model="text-embedding-3-small")
    assert captured.get("timeout") is not None and captured["timeout"] > 0
    assert captured.get("max_retries") is not None and captured["max_retries"] >= 0




