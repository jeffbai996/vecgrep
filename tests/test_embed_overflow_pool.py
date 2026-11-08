"""Several overflow Ollama endpoints for bulk embedding.

`ollama_overflow_url` may name more than one endpoint, comma-separated, each
serving the same model as the primary. A bulk batch that spans several
windows is spread across every endpoint that is not benched, in parallel, so
two hosts embed a backlog faster than one. Each endpoint keeps its own
cooldown: one host going to sleep does not stop the others. Every rule of the
single-overflow contract still holds per window.
"""
from __future__ import annotations

import json
import threading

import httpx
import pytest

from vecgrep.backend.config import ConfigError, Settings, validate_settings
from vecgrep.backend.embed import factory
from vecgrep.backend.embed.ollama import _MAX_BATCH, OllamaBackend

DIM = 8


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _vec(text: str, tag: float) -> list[float]:
    # Encodes the input so a test can tell which text a vector belongs to and
    # which host produced it.
    return [float(len(text)), tag] + [0.5] * (DIM - 2)


def _answer(req: httpx.Request, tag: float) -> httpx.Response:
    inp = json.loads(req.content)["input"]
    inp = inp if isinstance(inp, list) else [inp]
    return httpx.Response(200, json={"embeddings": [_vec(t, tag) for t in inp]})


def _pool(handlers: dict[str, object], *, clock: Clock | None = None):
    """A backend whose overflow endpoints answer through the given handlers."""
    seen: dict[str, int] = {"primary": 0, **{name: 0 for name in handlers}}
    lock = threading.Lock()

    def counted(name, fn):
        def handler(req):
            with lock:
                seen[name] += 1
            return fn(req)
        return handler

    b = OllamaBackend(
        base_url="http://primary",
        model="bge-m3",
        overflow_url=",".join(f"http://{name}" for name in handlers),
        clock=clock or Clock(),
    )
    b.dim = DIM
    b._client = httpx.Client(transport=httpx.MockTransport(
        counted("primary", lambda req: _answer(req, 0.0))))
    for endpoint, (name, fn) in zip(b._overflows, handlers.items()):
        endpoint.client = httpx.Client(transport=httpx.MockTransport(counted(name, fn)))
    return b, seen


def _texts(n: int) -> list[str]:
    return ["x" * (i + 1) for i in range(n)]


def test_comma_separated_overflow_urls_become_separate_endpoints() -> None:
    b = OllamaBackend("http://primary", "bge-m3",
                      overflow_url=" http://a:11434/ , http://b:11434,, ")
    assert [o.url for o in b._overflows] == ["http://a:11434", "http://b:11434"]


def test_windows_of_one_batch_run_on_two_endpoints_at_once() -> None:
    # Each host holds its request until the other host has one too. Run one
    # after the other, the barrier times out, both answer 503, and the batch
    # lands on the primary.
    barrier = threading.Barrier(2, timeout=5)

    def gated(tag):
        def handler(req):
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                return httpx.Response(503)
            return _answer(req, tag)
        return handler

    b, seen = _pool({"a": gated(1.0), "b": gated(2.0)})
    texts = _texts(2 * _MAX_BATCH)
    out = b.embed(texts)
    assert seen["primary"] == 0
    assert seen["a"] == 1 and seen["b"] == 1
    assert [v[0] for v in out] == [float(len(t)) for t in texts], "rows stay in input order"


def test_faster_host_takes_more_windows() -> None:
    slow_started = threading.Event()
    release_slow = threading.Event()

    def slow(req):
        slow_started.set()
        release_slow.wait(5)
        return _answer(req, 2.0)

    def fast(req):
        slow_started.wait(5)
        return _answer(req, 1.0)

    b, seen = _pool({"fast": fast, "slow": slow})
    texts = _texts(4 * _MAX_BATCH)
    done = threading.Event()
    result: list[list[list[float]]] = []

    def run():
        result.append(b.embed(texts))
        done.set()

    t = threading.Thread(target=run)
    t.start()
    # The slow host is stuck on its first window; the fast one drains the rest.
    deadline = threading.Event()
    for _ in range(500):
        if seen["fast"] >= 3:
            break
        deadline.wait(0.01)
    release_slow.set()
    t.join(5)
    assert done.is_set()
    assert seen["fast"] == 3 and seen["slow"] == 1
    assert len(result[0]) == len(texts)


def test_one_host_failing_leaves_its_work_to_the_other_and_benches_it() -> None:
    clock = Clock()
    down_called = threading.Event()

    def down(req):
        down_called.set()
        return httpx.Response(503)

    def up(req):
        # Hold the first answer until the failing host has taken a window,
        # so the healthy one cannot drain the whole batch first.
        down_called.wait(5)
        return _answer(req, 1.0)

    b, seen = _pool({"down": down, "up": up}, clock=clock)
    texts = _texts(3 * _MAX_BATCH)
    out = b.embed(texts)
    assert [v[0] for v in out] == [float(len(t)) for t in texts]
    assert [0.0] * DIM not in out
    assert seen["down"] == 1, "a refusing host is benched after one try"
    # Its one window went to the primary; the healthy host took the rest.
    assert seen["up"] + seen["primary"] == 3
    before = dict(seen)
    b.embed(_texts(2 * _MAX_BATCH))
    assert seen["down"] == before["down"], "still benched inside the cooldown"


def test_each_endpoint_has_its_own_cooldown() -> None:
    clock = Clock()
    state = {"a_up": False}
    b, seen = _pool({
        "a": lambda req: _answer(req, 1.0) if state["a_up"] else httpx.Response(503),
        "b": lambda req: _answer(req, 2.0),
    }, clock=clock)
    b.embed(_texts(2 * _MAX_BATCH))
    state["a_up"] = True
    clock.now += b.overflow_cooldown_s + 1
    a_before = seen["a"]
    b.embed(_texts(2 * _MAX_BATCH))
    assert seen["a"] > a_before


def test_all_endpoints_benched_uses_the_primary() -> None:
    b, seen = _pool({"a": lambda req: httpx.Response(503),
                     "b": lambda req: httpx.Response(503)})
    out = b.embed(_texts(2 * _MAX_BATCH))
    assert all(v[1] == 0.0 for v in out)
    calls_after_first = seen["a"] + seen["b"]
    b.embed(_texts(2))
    assert seen["a"] + seen["b"] == calls_after_first


def test_single_window_goes_to_the_first_listed_endpoint() -> None:
    b, seen = _pool({"a": lambda req: _answer(req, 1.0),
                     "b": lambda req: _answer(req, 2.0)})
    for _ in range(3):
        b.embed(["one", "two"])
    assert seen["a"] == 3 and seen["b"] == 0


def test_single_window_skips_a_busy_endpoint_for_an_idle_one() -> None:
    a_entered = threading.Event()
    release_a = threading.Event()

    def held(req):
        a_entered.set()
        release_a.wait(5)
        return _answer(req, 1.0)

    b, _seen = _pool({"a": held, "b": lambda req: _answer(req, 2.0)})
    t = threading.Thread(target=b.embed, args=(["first", "batch"],))
    t.start()
    assert a_entered.wait(5)
    out = b.embed(["second", "batch"])
    release_a.set()
    t.join(5)
    assert all(v[1] == 2.0 for v in out)


def test_queries_still_use_only_the_primary() -> None:
    b, seen = _pool({"a": lambda req: _answer(req, 1.0),
                     "b": lambda req: _answer(req, 2.0)})
    b.embed_one("a query")
    assert seen["a"] == seen["b"] == 0


# ── configuration ─────────────────────────────────────────────────────────

def test_a_list_of_overflow_urls_validates() -> None:
    s = Settings()
    s.ollama_overflow_url = "http://a:11434,http://b:11434"
    validate_settings(s)


def test_one_bad_url_in_the_list_is_rejected() -> None:
    s = Settings()
    s.ollama_overflow_url = "http://a:11434,not a url"
    with pytest.raises(ConfigError):
        validate_settings(s)


def test_factory_passes_every_overflow_endpoint(monkeypatch) -> None:
    s = Settings()
    s.ollama_url = "http://primary:11434"
    s.ollama_overflow_url = "http://a:11434,http://b:11434"
    monkeypatch.setattr(factory, "_ollama_alive", lambda url: url == "http://primary:11434")
    b = factory.get_embed_backend(s)
    assert isinstance(b, OllamaBackend)
    assert [o.url for o in b._overflows] == ["http://a:11434", "http://b:11434"]


# ── prewarm ───────────────────────────────────────────────────────────────

def test_prewarm_embeds_many_small_documents_in_shared_batches(svc, stub_backend, make_doc) -> None:
    """Prewarm used to embed one document per call, so a backlog of small
    files never produced a multi-window batch for the pool to spread."""
    for i in range(6):
        make_doc(f"src/doc{i}.md", f"Document number {i} says something about topic {i}.")
    root = make_doc("src/doc0.md", "Document number 0 says something about topic 0.").parent
    stub_backend.embed_sizes = []
    original = stub_backend.embed

    def recording(texts):
        stub_backend.embed_sizes.append(len(texts))
        return original(texts)

    stub_backend.embed = recording
    svc._prewarm_index_embeddings(str(root), "notes", "sentence_window")
    assert len(stub_backend.embed_sizes) == 1
    assert stub_backend.embed_sizes[0] >= 6
