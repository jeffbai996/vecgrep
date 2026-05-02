"""Shared fixtures.

Tests use a deterministic fake embedding backend instead of Ollama/OpenAI
so they're hermetic and fast. Each test gets a fresh VECGREP_HOME under
tmp_path, isolated from any real ~/.vecgrep on the dev box.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest

from vecgrep.backend import config as cfg_mod
from vecgrep.backend.embed.base import EmbedBackend
from vecgrep.backend.service import VecgrepService


class StubEmbed(EmbedBackend):
    """Tiny deterministic embed backend with a configurable identity.

    Vector is derived from text content so semantically related strings
    score together, but the function is stable across runs (no RNG).
    """

    name = "stub"
    model = "stub-1"
    dim = 16
    calls = 0

    def __init__(self, name: str = "stub", model: str = "stub-1", dim: int = 16) -> None:
        type(self).calls = 0
        self.name = name
        self.model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        type(self).calls += 1
        out: list[list[float]] = []
        for t in texts:
            v = [0.0] * self.dim
            # Codepoint-based: similar prefixes -> similar vectors.
            for i, ch in enumerate(t.lower()[: self.dim]):
                v[i] = (ord(ch) - 96) / 26.0 if ch.isalpha() else (ord(ch) % 26) / 26.0
            out.append(v)
        return out


@pytest.fixture
def vg_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Isolate VECGREP_HOME per test. Resets the config singleton so the
    new env var actually wins.
    """
    home = tmp_path / "vg"
    monkeypatch.setenv("VECGREP_HOME", str(home))
    monkeypatch.setattr(cfg_mod, "_settings", None)
    yield home


@pytest.fixture
def stub_backend() -> StubEmbed:
    return StubEmbed()


@pytest.fixture
def svc(vg_home: Path, stub_backend: StubEmbed) -> Iterator[VecgrepService]:
    """A VecgrepService with the stub embedding pre-injected."""
    from vecgrep.backend.embed.cache import CachedBackend

    s = VecgrepService(ephemeral=False)
    s._backend_cache["auto"] = CachedBackend(stub_backend, s._embed_cache)
    yield s
    # Best-effort qdrant cleanup so the next test's fixture doesn't trip
    # the single-process lock when running in the same pytest session.
    try:
        s.store.client.close()
    except Exception:
        pass


@pytest.fixture
def make_doc(tmp_path: Path):
    """Factory for creating temp source files inside tmp_path."""

    def _make(name: str, content: str) -> Path:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    return _make
