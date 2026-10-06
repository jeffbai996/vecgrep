"""Overflow Ollama endpoint for bulk embedding.

A deployment can point bulk embedding (indexing batches) at a second GPU so
the primary stays free for queries and other work. The overflow host may be
switched off at any time (for example while its GPU is needed elsewhere), so
it must never be load-bearing:

- queries (a single text) always go to the primary;
- a batch goes to the overflow first, and anything short of a clean answer
  sends that batch to the primary instead;
- a failing overflow is benched for a cooldown so a down host does not cost
  every batch a round trip;
- an overflow failure is never turned into zero vectors, which would store
  chunks that can no longer be found by meaning and raise no error.
"""
from __future__ import annotations

import json

import httpx
import pytest

from vecgrep.backend.config import ConfigError, Settings, load_settings, validate_settings
from vecgrep.backend.embed import factory
from vecgrep.backend.embed.ollama import OllamaBackend

DIM = 1024
PRIMARY_VEC = [0.1] * DIM
OVERFLOW_VEC = [0.2] * DIM


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _rows(req: httpx.Request, vec: list[float]) -> httpx.Response:
    inp = json.loads(req.content)["input"]
    n = len(inp) if isinstance(inp, list) else 1
    return httpx.Response(200, json={"embeddings": [vec] * n})


def _backend(overflow_handler, *, clock: Clock | None = None, num_batch=None):
    calls = {"primary": [], "overflow": []}

    def primary(req):
        calls["primary"].append(json.loads(req.content))
        return _rows(req, PRIMARY_VEC)

    def overflow(req):
        calls["overflow"].append(json.loads(req.content))
        return overflow_handler(req)

    b = OllamaBackend(
        base_url="http://primary",
        model="bge-m3",
        num_batch=num_batch,
        overflow_url="http://overflow",
        clock=clock or Clock(),
    )
    b._client = httpx.Client(transport=httpx.MockTransport(primary))
    b._overflow_client = httpx.Client(transport=httpx.MockTransport(overflow))
    return b, calls


def test_batch_goes_to_overflow_when_it_answers_cleanly() -> None:
    b, calls = _backend(lambda req: _rows(req, OVERFLOW_VEC))
    out = b.embed(["a", "b", "c"])
    assert out == [OVERFLOW_VEC] * 3
    assert calls["primary"] == []


def test_single_text_query_stays_on_primary() -> None:
    b, calls = _backend(lambda req: _rows(req, OVERFLOW_VEC))
    assert b.embed_one("what did we decide") == PRIMARY_VEC
    assert calls["overflow"] == []


def test_overflow_refusal_sends_the_batch_to_primary_not_zero_vectors() -> None:
    """A switched-off overflow answers 503. That must not become zero vectors."""
    b, calls = _backend(lambda req: httpx.Response(503, text="gpu in use"))
    out = b.embed(["a", "b"])
    assert out == [PRIMARY_VEC] * 2
    assert [0.0] * DIM not in out


def test_unreachable_overflow_falls_back_to_primary() -> None:
    def down(req):
        raise httpx.ConnectError("no route", request=req)

    b, _ = _backend(down)
    assert b.embed(["a", "b"]) == [PRIMARY_VEC] * 2


@pytest.mark.parametrize("bad", [
    lambda req: httpx.Response(200, json={"embeddings": [OVERFLOW_VEC]}),          # wrong row count
    lambda req: httpx.Response(200, json={"embeddings": [[float("nan")] * DIM] * 2}),
    lambda req: httpx.Response(200, text="not json"),
    lambda req: httpx.Response(500, text="unsupported value: NaN"),
])
def test_any_unclean_overflow_answer_uses_primary(bad) -> None:
    b, _ = _backend(bad)
    assert b.embed(["a", "b"]) == [PRIMARY_VEC] * 2


def test_failed_overflow_is_benched_for_the_cooldown() -> None:
    clock = Clock()
    b, calls = _backend(lambda req: httpx.Response(503), clock=clock)
    b.embed(["a", "b"])
    b.embed(["c", "d"])
    assert len(calls["overflow"]) == 1, "a benched overflow is not retried every batch"
    assert len(calls["primary"]) == 2


def test_overflow_is_retried_after_the_cooldown() -> None:
    clock = Clock()
    state = {"up": False}

    def flaky(req):
        return _rows(req, OVERFLOW_VEC) if state["up"] else httpx.Response(503)

    b, calls = _backend(flaky, clock=clock)
    b.embed(["a", "b"])
    state["up"] = True
    clock.now += b.overflow_cooldown_s + 1
    assert b.embed(["c", "d"]) == [OVERFLOW_VEC] * 2


def test_overflow_request_carries_the_same_model_and_batch_options() -> None:
    b, calls = _backend(lambda req: _rows(req, OVERFLOW_VEC), num_batch=4096)
    b.embed(["a", "b"])
    sent = calls["overflow"][0]
    assert sent["model"] == "bge-m3"
    assert sent["options"] == {"num_batch": 4096}
    assert sent["truncate"] is True


def test_no_overflow_configured_keeps_primary_only() -> None:
    seen = []

    def primary(req):
        seen.append(req.url.host)
        return _rows(req, PRIMARY_VEC)

    b = OllamaBackend(base_url="http://primary", model="bge-m3")
    b._client = httpx.Client(transport=httpx.MockTransport(primary))
    assert b.embed(["a", "b"]) == [PRIMARY_VEC] * 2
    assert set(seen) == {"primary"}


# ── configuration ─────────────────────────────────────────────────────────

def test_overflow_url_comes_from_the_environment(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VECGREP_HOME", str(tmp_path))
    monkeypatch.setenv("VECGREP_OLLAMA_OVERFLOW_URL", "http://overflow:11434")
    assert load_settings().ollama_overflow_url == "http://overflow:11434"


def test_empty_overflow_env_means_unset(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VECGREP_HOME", str(tmp_path))
    monkeypatch.setenv("VECGREP_OLLAMA_OVERFLOW_URL", "")
    assert load_settings().ollama_overflow_url is None


def test_invalid_overflow_url_is_rejected() -> None:
    s = Settings()
    s.ollama_overflow_url = "not a url"
    with pytest.raises(ConfigError):
        validate_settings(s)


def test_factory_hands_the_overflow_url_to_the_backend(monkeypatch) -> None:
    s = Settings()
    s.ollama_url = "http://primary:11434"
    s.ollama_overflow_url = "http://overflow:11434"
    monkeypatch.setattr(factory, "_ollama_alive", lambda url: url == "http://primary:11434")
    b = factory.get_embed_backend(s)
    assert b.base_url == "http://primary:11434"
    assert b.overflow_url == "http://overflow:11434"
