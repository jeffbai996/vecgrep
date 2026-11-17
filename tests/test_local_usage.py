from __future__ import annotations

import json
import sqlite3

import httpx
import pytest

from vecgrep.backend.embed import local_usage
from vecgrep.backend.embed.local_usage import record
from vecgrep.backend.embed.ollama import OllamaBackend

# ── test_embed_local_usage ──

@pytest.fixture
def receipt_db(tmp_path, monkeypatch):
    path = tmp_path / "usage.sqlite3"
    monkeypatch.setenv("LOCAL_LLM_USAGE_DB", str(path))
    monkeypatch.setenv("VECGREP_HOME", str(tmp_path / "vecgrep"))
    return path


def rows(path):
    with sqlite3.connect(path) as db:
        return db.execute("SELECT request_id,at,host,model,input_tokens,output_tokens FROM usage").fetchall()


@pytest.mark.parametrize("lane", ["batch", "single", "overflow"])
def test_each_embedding_lane_records_reported_input_tokens(receipt_db, lane):
    backend = OllamaBackend("http://primary.invalid", "bge-m3", overflow_url="http://overflow.invalid")
    def answer(request):
        return httpx.Response(200, json={"model": "bge-test", "embeddings": [[0.1] * 1024],
                                        "prompt_eval_count": 17})
    backend._client.close()
    assert backend._overflow_client is not None
    backend._overflow_client.close()
    backend._client = httpx.Client(transport=httpx.MockTransport(answer))
    backend._overflow_client = httpx.Client(transport=httpx.MockTransport(answer))
    try:
        if lane == "batch": backend._embed_batch(["private text"])
        elif lane == "single": backend._embed_one_resilient("private text")
        else: backend._embed_overflow(backend._overflows[0], ["private text"])
    finally:
        backend._client.close()
        backend._overflow_client.close()
    row, = rows(receipt_db)
    assert row[0].startswith("vecgrep:") and row[1] > 0
    assert row[2:] == ("vecgrep", "bge-test", 17, 0)
    assert b"private text" not in receipt_db.read_bytes()
    assert receipt_db.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("data", [{}, {"prompt_eval_count": True},
    {"prompt_eval_count": -1}, {"prompt_eval_count": 2**63}])
def test_unknown_or_invalid_counts_are_not_estimated(receipt_db, data):
    record(data, model="bge-test")
    assert not receipt_db.exists()


def test_unavailable_receipt_storage_does_not_fail_embedding(receipt_db):
    receipt_db.mkdir()
    record({"prompt_eval_count": 12}, model="bge-test")


def test_concurrent_token_receipt_writes_do_not_drop_rows_or_lock(receipt_db):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda i: record({"prompt_eval_count": i + 1}, model="bge-test", route="primary"),
                range(32),
            )
        )
    with sqlite3.connect(receipt_db) as db:
        mode = db.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    assert len(rows(receipt_db)) == 32

# ── test_embed_route_receipts ──

DIM = 1024


@pytest.fixture
def receipt_db_embed_route_receipts(tmp_path, monkeypatch):
    path = tmp_path / "usage.sqlite3"
    monkeypatch.setenv("LOCAL_LLM_USAGE_DB", str(path))
    monkeypatch.setenv("VECGREP_HOME", str(tmp_path / "vecgrep"))
    return path


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _answer(req: httpx.Request, *, tokens: int = 9) -> httpx.Response:
    inp = json.loads(req.content)["input"]
    n = len(inp) if isinstance(inp, list) else 1
    return httpx.Response(200, json={"model": "bge-m3", "embeddings": [[0.1] * DIM] * n,
                                     "prompt_eval_count": tokens})


def _backend(overflow, *, primary=_answer, clock=None):
    b = OllamaBackend("http://primary.invalid", "bge-m3", overflow_url="http://overflow.invalid",
                      clock=clock or Clock())
    b._client.close()
    assert b._overflow_client is not None
    b._overflow_client.close()
    b._client = httpx.Client(transport=httpx.MockTransport(primary))
    b._overflow_client = httpx.Client(transport=httpx.MockTransport(overflow))
    return b


def routes(path):
    with sqlite3.connect(path) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='embed_route'").fetchone():
            return []
        return db.execute("SELECT route, outcome, reason FROM embed_route ORDER BY rowid").fetchall()


def test_a_batch_the_overflow_answers_is_recorded_as_overflow(receipt_db_embed_route_receipts):
    b = _backend(_answer)
    b.embed(["a", "b"])
    assert routes(receipt_db_embed_route_receipts) == [("overflow", "ok", "")]
    with sqlite3.connect(receipt_db_embed_route_receipts) as db:
        linked = db.execute("SELECT u.input_tokens FROM embed_route r JOIN usage u"
                            " ON u.request_id = r.request_id").fetchall()
    assert linked == [(9,)]


def test_a_query_is_not_a_routed_batch(receipt_db_embed_route_receipts):
    b = _backend(_answer)
    b.embed(["one query"])
    assert routes(receipt_db_embed_route_receipts) == []


@pytest.mark.parametrize("overflow,reason", [
    (lambda req: httpx.Response(503, json={"error": "residency unknown"}), "refused"),
    (lambda req: httpx.Response(500, text="boom"), "http_500"),
    (lambda req: httpx.Response(200, json={"embeddings": [[0.1] * DIM]}), "bad_vectors"),
    (lambda req: httpx.Response(200, text="not json"), "bad_response"),
])
def test_a_fallback_records_why_and_the_primary_batch(receipt_db_embed_route_receipts, overflow, reason):
    b = _backend(overflow)
    b.embed(["a", "b"])
    assert routes(receipt_db_embed_route_receipts) == [("overflow", "fallback", reason), ("primary", "ok", "")]


@pytest.mark.parametrize("error,reason", [
    (httpx.ConnectTimeout("slow"), "timeout"),
    (httpx.ConnectError("down"), "unreachable"),
])
def test_a_transport_failure_records_its_reason(receipt_db_embed_route_receipts, error, reason):
    def overflow(req):
        raise error
    b = _backend(overflow)
    b.embed(["a", "b"])
    assert routes(receipt_db_embed_route_receipts)[0] == ("overflow", "fallback", reason)


def test_a_benched_overflow_records_primary_batches_only(receipt_db_embed_route_receipts):
    clock = Clock()
    b = _backend(lambda req: httpx.Response(503), clock=clock)
    b.embed(["a", "b"])
    b.embed(["c", "d"])
    assert routes(receipt_db_embed_route_receipts) == [("overflow", "fallback", "refused"), ("primary", "ok", ""),
                                  ("primary", "ok", "")]


def test_the_shared_usage_table_keeps_its_six_columns(receipt_db_embed_route_receipts):
    """Sibling services sharing the usage table insert six positional values."""
    with sqlite3.connect(receipt_db_embed_route_receipts) as db:
        db.execute("CREATE TABLE usage (request_id TEXT PRIMARY KEY, at REAL NOT NULL,"
                   " host TEXT NOT NULL, model TEXT NOT NULL, input_tokens INTEGER NOT NULL,"
                   " output_tokens INTEGER NOT NULL)")
        db.execute("INSERT INTO usage VALUES (?,?,?,?,?,?)", ("other:1", 1.0, "other", "m", 3, 4))
    _backend(_answer).embed(["a", "b"])
    with sqlite3.connect(receipt_db_embed_route_receipts) as db:
        db.execute("INSERT INTO usage VALUES (?,?,?,?,?,?)", ("other:2", 2.0, "other", "m", 5, 6))
        columns = [row[1] for row in db.execute("PRAGMA table_info(usage)")]
        others = db.execute("SELECT input_tokens FROM usage WHERE host='other' ORDER BY at").fetchall()
    assert columns == ["request_id", "at", "host", "model", "input_tokens", "output_tokens"]
    assert others == [(3,), (5,)]


def test_route_rows_carry_no_endpoint_or_text(receipt_db_embed_route_receipts):
    _backend(lambda req: httpx.Response(503)).embed(["private text", "more"])
    raw = receipt_db_embed_route_receipts.read_bytes()
    assert b"private text" not in raw and b"overflow.invalid" not in raw and b"primary.invalid" not in raw


def test_old_route_rows_are_pruned(receipt_db_embed_route_receipts, monkeypatch):
    local_usage.record_route(route="primary", outcome="ok")
    with sqlite3.connect(receipt_db_embed_route_receipts) as db:
        db.execute("UPDATE embed_route SET at = at - ?", (local_usage.ROUTE_RETENTION_S + 60,))
    local_usage.record_route(route="primary", outcome="ok")
    assert len(routes(receipt_db_embed_route_receipts)) == 1


def test_unavailable_storage_does_not_fail_embedding(receipt_db_embed_route_receipts):
    receipt_db_embed_route_receipts.mkdir()
    assert len(_backend(_answer).embed(["a", "b"])) == 2


def test_no_test_writes_the_real_receipt_database(tmp_path):
    """Every test's receipts land in its own directory.

    The overflow tests build backends without asking for a receipt fixture, and
    each bulk batch now writes a route row, so an unisolated run put test rows
    in the operator's real database.
    """
    assert local_usage._database().resolve().is_relative_to(tmp_path.resolve())


def test_an_overflow_connection_that_breaks_mid_answer_records_error(receipt_db_embed_route_receipts):
    def overflow(req):
        raise httpx.ReadError("connection reset")
    _backend(overflow).embed(["a", "b"])
    assert routes(receipt_db_embed_route_receipts)[0] == ("overflow", "fallback", "error")


def _primary_only(primary):
    b = OllamaBackend("http://primary.invalid", "bge-m3", clock=Clock())
    b._client.close()
    b._client = httpx.Client(transport=httpx.MockTransport(primary))
    return b


def _bulk_only(bad):
    """`bad` answers the multi-text batch; single-chunk retries succeed."""
    def handler(req):
        return bad(req) if isinstance(json.loads(req.content)["input"], list) else _answer(req)
    return handler


@pytest.mark.parametrize("bad,reason", [
    (lambda req: httpx.Response(200, json={"embeddings": [[0.1] * DIM]}), "bad_vectors"),
    (lambda req: httpx.Response(200, content=b'{"embeddings": [[NaN, 0.1], [0.1, 0.1]]}'), "bad_vectors"),
    (lambda req: httpx.Response(200, text="not json"), "bad_response"),
    (lambda req: httpx.Response(200, json=[1, 2]), "bad_response"),
    (lambda req: httpx.Response(500, text="boom"), "http_500"),
])
def test_a_primary_batch_is_ok_only_after_its_answer_validates(receipt_db_embed_route_receipts, bad, reason):
    """A degraded primary batch goes to per-chunk requests; its route row
    says so instead of reporting success or nothing."""
    vectors = _primary_only(_bulk_only(bad)).embed(["a", "b"])
    assert len(vectors) == 2
    assert routes(receipt_db_embed_route_receipts) == [("primary", "fallback", reason)]


@pytest.mark.parametrize("error,reason", [
    (httpx.ReadTimeout("slow"), "timeout"),
    (httpx.ReadError("reset"), "error"),
])
def test_a_primary_batch_transport_failure_records_its_reason(receipt_db_embed_route_receipts, error, reason):
    def primary(req):
        if isinstance(json.loads(req.content)["input"], list):
            raise error
        return _answer(req)
    from vecgrep.backend.embed.base import EmbedBackendError
    with pytest.raises(EmbedBackendError):
        _primary_only(primary).embed(["a", "b"])
    assert routes(receipt_db_embed_route_receipts) == [("primary", "fallback", reason)]


def test_an_unreachable_primary_records_the_batch_before_failing(receipt_db_embed_route_receipts):
    from vecgrep.backend.embed.base import EmbedBackendError

    def primary(req):
        raise httpx.ConnectError("down")
    with pytest.raises(EmbedBackendError):
        _primary_only(primary).embed(["a", "b"])
    assert routes(receipt_db_embed_route_receipts) == [("primary", "fallback", "unreachable")]


def test_a_clean_primary_batch_is_ok_and_linked_to_its_tokens(receipt_db_embed_route_receipts):
    _primary_only(_answer).embed(["a", "b"])
    assert routes(receipt_db_embed_route_receipts) == [("primary", "ok", "")]
    with sqlite3.connect(receipt_db_embed_route_receipts) as db:
        linked = db.execute("SELECT u.input_tokens FROM embed_route r JOIN usage u"
                            " ON u.request_id = r.request_id").fetchall()
    assert linked == [(9,)]
